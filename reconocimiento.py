# file: airi_smart_home_face.py
import os
import re
import cv2
import json
import time
import pytz
import torch
import shutil
import datetime
import requests
import numpy as np
import collections
import sounddevice as sd
import speech_recognition as sr
from pathlib import Path
from llama_cpp import Llama

# =========================
# ===== CONFIGURACIÓN =====
# =========================
MODEL_PATH = "modelos/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
ESP_HOST = "http://192.168.43.150"
ESP_TIMEOUT = 3
MEM_FILE = "memoria_airi.json"

# Reconocimiento facial
DATASET_DIR = Path("data_faces")
MODEL_DIR = Path("modelos")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

FACE_MODEL_FILE = MODEL_DIR / "face_lbph.xml"
FACE_LABELS_FILE = MODEL_DIR / "face_labels.json"

CAM_INDEX = 0
SAMPLES_PER_PERSON = 25
FACE_SIZE = (200, 200)
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_THRESHOLD = 65.0
AUTH_REQUIRED_MATCHES = 5

# =========================
# ===== UTILIDADES JSON ===
# =========================
def cargar_memoria():
    if os.path.exists(MEM_FILE):
        with open(MEM_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    data.setdefault("nombre", None)
    data.setdefault("emocion", None)
    data.setdefault("ultima_fecha", None)
    data.setdefault("autorizados", [])
    return data


def guardar_memoria(mem):
    with open(MEM_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)


def cargar_labels():
    if FACE_LABELS_FILE.exists():
        with open(FACE_LABELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"name_to_id": {}, "id_to_name": {}}


def guardar_labels(labels):
    with open(FACE_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


# =========================
# ===== VOZ (Silero) ======
# =========================
def hablar(texto: str):
    print(f"Airi: {texto}")
    try:
        global silero_model, silero_loaded
        if "silero_loaded" not in globals():
            torch.set_num_threads(2)
            silero_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="es",
                speaker="v3_es",
            )
            silero_loaded = True
        audio = silero_model.apply_tts(text=texto, speaker="es_1", sample_rate=48000)
        sd.play(audio.numpy(), 48000)
        sd.wait()
    except Exception as e:
        print(f"[Error voz] {e}")


# =========================
# == CONEXIÓN ESP8266 =====
# =========================
def enviar_a_esp(accion, lugar=None, valor=None):
    try:
        if accion == "on":
            url = f"{ESP_HOST}/on?lugar={lugar}"
            if valor is not None:
                url += f"&valor={valor}"
        elif accion == "off":
            url = f"{ESP_HOST}/off?lugar={lugar}"
        elif accion == "estado":
            url = f"{ESP_HOST}/estado"
        else:
            print("⚠️ Acción no reconocida")
            return None

        print(f"🌐 Enviando → {url}")
        r = requests.get(url, timeout=ESP_TIMEOUT)
        if accion == "estado":
            return r.json()
        else:
            print("💬 Respuesta:", r.text)
    except Exception as e:
        print(f"❌ Error al contactar con el ESP8266: {e}")
        return None


# =========================
# ======== AUDIO ==========
# =========================
r = sr.Recognizer()
try:
    mic = sr.Microphone()
except Exception as e:
    raise RuntimeError("Micrófono no disponible") from e


def escuchar(timeout: float | None = None, phrase_time_limit: float | None = None):
    with mic as source:
        print("🎤 Escuchando...")
        r.adjust_for_ambient_noise(source, duration=0.4)
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        texto = r.recognize_google(audio, language="es-ES")
        print(f"Tú: {texto}")
        return texto
    except Exception:
        hablar("No te entendí, repite por favor.")
        return None


# =========================
# ======= FECHA ===========
# =========================
dias_es = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}
meses_es = {
    "January": "enero",
    "February": "febrero",
    "March": "marzo",
    "April": "abril",
    "May": "mayo",
    "June": "junio",
    "July": "julio",
    "August": "agosto",
    "September": "septiembre",
    "October": "octubre",
    "November": "noviembre",
    "December": "diciembre",
}
zona_peru = pytz.timezone("America/Lima")


def fecha_hoy_str():
    ahora = datetime.datetime.now(zona_peru)
    return f"{dias_es[ahora.strftime('%A')]} {ahora.day:02d} de {meses_es[ahora.strftime('%B')]} de {ahora.year}"


# =========================
# === MAPA DE LUGARES =====
# =========================
LUGARES = {
    "cocina": "ledCocina",
    "cochera": "ledCochera",
    "sala": "ledSala",
    "oficina": "ledOficina",
    "pasillo": "ledPasillo",
    "cuarto principal": "ledCuartoPrincipal",
    "cuarto de visita": "ledCuartoVisita",
}

# =========================
# == MODELO DE LENGUAJE ===
# =========================
print("Cargando modelo LLM...")
llm = Llama(model_path=MODEL_PATH, n_ctx=768, n_threads=8, verbose=False)
print("✅ Modelo LLM cargado\n")

# =========================
# === OpenCV Face/Video ===
# =========================
if not hasattr(cv2, "face"):
    raise RuntimeError("OpenCV no tiene 'cv2.face'. Instala 'opencv-contrib-python'.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def abrir_camara(index: int = CAM_INDEX) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def detectar_caras(gray: np.ndarray):
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    return faces


def capturar_muestras(nombre: str, muestras: int = SAMPLES_PER_PERSON):
    persona_dir = DATASET_DIR / nombre.lower().strip().replace(" ", "_")
    if persona_dir.exists():
        shutil.rmtree(persona_dir)  # por qué: evitar datos viejos
    persona_dir.mkdir(parents=True, exist_ok=True)

    hablar(f"Mirando a la cámara. Registrando a {nombre}. Mantén el rostro centrado.")
    cap = abrir_camara()
    guardadas = 0
    consecutivas_sin_rostro = 0
    try:
        while guardadas < muestras:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detectar_caras(gray)
            if len(faces) != 1:
                consecutivas_sin_rostro += 1
                if consecutivas_sin_rostro % 30 == 0:
                    hablar("Asegúrate de que solo aparezca un rostro.")
                continue
            (x, y, w, h) = faces[0]
            crop = gray[y : y + h, x : x + w]
            crop = cv2.resize(crop, FACE_SIZE)
            out_path = persona_dir / f"img_{guardadas:03d}.png"
            cv2.imwrite(str(out_path), crop)
            guardadas += 1
            if guardadas in (5, 15):
                hablar(f"{guardadas} muestras.")
        hablar("Registro completado.")
    finally:
        cap.release()


def _recorrer_dataset():
    images = []
    labels = []
    name_to_id = {}
    id_to_name = {}
    current_id = 0
    for person_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        name = person_dir.name.replace("_", " ")
        name = name.lower()
        name_to_id[name] = current_id
        id_to_name[str(current_id)] = name
        for img_path in person_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            images.append(img)
            labels.append(current_id)
        current_id += 1
    return images, np.array(labels, dtype=np.int32), name_to_id, id_to_name


def entrenar_modelo():
    images, labels, name_to_id, id_to_name = _recorrer_dataset()
    if not images or len(np.unique(labels)) == 0:
        raise RuntimeError("Dataset vacío. Registra al menos una persona.")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )
    recognizer.train(images, labels)
    recognizer.write(str(FACE_MODEL_FILE))
    guardar_labels({"name_to_id": name_to_id, "id_to_name": id_to_name})


def cargar_modelo():
    if not FACE_MODEL_FILE.exists() or not FACE_LABELS_FILE.exists():
        return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )
    recognizer.read(str(FACE_MODEL_FILE))
    labels = cargar_labels()
    return recognizer, labels


def asegurar_duenio(memoria):
    recognizer, labels = cargar_modelo()
    if recognizer is not None and labels is not None:
        return
    hablar("No hay rostros registrados. Vamos a registrar al dueño: aaron.")
    capturar_muestras("aaron", SAMPLES_PER_PERSON)
    entrenar_modelo()
    memoria.setdefault("autorizados", [])
    if "aaron" not in memoria["autorizados"]:
        memoria["autorizados"].append("aaron")
    guardar_memoria(memoria)
    hablar("Listo. Dueño registrado.")


def reconocer_frame(gray: np.ndarray, recognizer, labels):
    faces = detectar_caras(gray)
    if len(faces) != 1:
        return None, None
    (x, y, w, h) = faces[0]
    crop = gray[y : y + h, x : x + w]
    crop = cv2.resize(crop, FACE_SIZE)
    label_id, confidence = recognizer.predict(crop)
    nombre = labels["id_to_name"].get(str(label_id))
    return nombre, confidence


def autenticar_usuario():
    recognizer, labels = cargar_modelo()
    if recognizer is None:
        return None, None
    cap = abrir_camara()
    hablar("Mirando a la cámara para autenticar.")
    try:
        last_name = None
        streak = 0
        last_conf = 999.0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nombre, conf = reconocer_frame(gray, recognizer, labels)
            if nombre is not None and conf is not None and conf < LBPH_THRESHOLD:
                if nombre == last_name:
                    streak += 1
                else:
                    last_name = nombre
                    streak = 1
                last_conf = conf
                if streak >= AUTH_REQUIRED_MATCHES:
                    return nombre, last_conf
            else:
                last_name = None
                streak = 0
            if time.time() - t0 > 30:
                return None, None
    finally:
        cap.release()


def registrar_nueva_persona(nombre: str, memoria):
    nombre = normalizar_nombre(nombre)
    if not nombre:
        hablar("Nombre inválido.")
        return False
    hablar(f"Iniciando registro de {nombre}.")
    capturar_muestras(nombre, SAMPLES_PER_PERSON)
    entrenar_modelo()
    if nombre not in memoria["autorizados"]:
        memoria["autorizados"].append(nombre)
    guardar_memoria(memoria)
    hablar(f"{nombre} registrado y autorizado.")
    return True


def normalizar_nombre(nombre: str) -> str:
    n = nombre.strip().lower()
    n = re.sub(r"\s+", " ", n)
    n = n.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    n = n.strip()
    return n


def eliminar_persona(nombre: str, memoria) -> bool:
    """
    Elimina dataset y acceso de 'nombre'. Reentrena o borra modelo si no quedan personas.
    """
    nombre = normalizar_nombre(nombre)
    if not nombre:
        hablar("Nombre inválido.")
        return False
    if nombre == "aaron":
        hablar("No puedo eliminar al dueño aaron para evitar bloqueo.")
        return False

    persona_dir = DATASET_DIR / nombre.replace(" ", "_")
    if not persona_dir.exists():
        hablar(f"No encuentro datos de {nombre}.")
        # aún así quita de autorizados por seguridad
        if nombre in memoria.get("autorizados", []):
            memoria["autorizados"].remove(nombre)
            guardar_memoria(memoria)
        return False

    shutil.rmtree(persona_dir)

    # Quitar de autorizados
    if nombre in memoria.get("autorizados", []):
        memoria["autorizados"].remove(nombre)
    guardar_memoria(memoria)

    # ¿Queda alguien?
    quedan = [p for p in DATASET_DIR.iterdir() if p.is_dir()]
    if not quedan:
        # por qué: modelo obsoleto si no hay dataset
        if FACE_MODEL_FILE.exists():
            FACE_MODEL_FILE.unlink()
        if FACE_LABELS_FILE.exists():
            FACE_LABELS_FILE.unlink()
        hablar("Usuario eliminado. No quedan rostros. Se limpió el modelo.")
        return True

    # Reentrenar con el resto
    try:
        entrenar_modelo()
        hablar(f"{nombre} eliminado y modelo actualizado.")
    except Exception as e:
        # Si falla el reentrenamiento, dejar el sistema en estado limpio
        if FACE_MODEL_FILE.exists():
            FACE_MODEL_FILE.unlink()
        if FACE_LABELS_FILE.exists():
            FACE_LABELS_FILE.unlink()
        hablar("Se eliminó el usuario, pero hubo un error al reentrenar. Modelo reiniciado.")
    return True


# =========================
# ======== DIALOGO ========
# =========================
def responder_llm(contexto_deque, texto_usuario):
    contexto_deque.append(f"Tú: {texto_usuario}")
    historial = "\n".join(contexto_deque)
    prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"Eres Airi, asistente local creada por Aaron Yair Torres Novella.\n"
        f"Hablas español natural, cálido y breve.\n"
        f"Fecha: {fecha_hoy_str()}.\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{historial}\n"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )
    output = llm(prompt, max_tokens=100, temperature=0.7)
    respuesta = output["choices"][0]["text"].strip()
    respuesta = re.sub(r"<\|.*?\|>", "", respuesta).replace("\n", " ").strip()
    return respuesta


def parsear_registro(texto: str):
    t = texto.lower().strip()
    m = re.search(
        r"(?:registra|registrar|agrega|agregar|añade|añadir)\s+(?:el\s+rostro\s+de\s+|a\s+)?([a-záéíóúñ\s]+)$",
        t,
    )
    if m:
        nombre = m.group(1).strip()
        nombre = re.sub(r"^(a\s+|al\s+|la\s+|el\s+)", "", nombre)
        return nombre
    return None


def parsear_borrado(texto: str):
    """
    Soporta: elimina/borra/quitar permiso/revoca acceso de <nombre>
    """
    t = texto.lower().strip()
    patron = re.compile(
        r"(?:elimina|eliminar|borra|borrar|quita|quitar)\s+(?:permiso\s+de\s+|acceso\s+de\s+|rostro\s+de\s+|a\s+)?([a-záéíóúñ\s]+)$"
    )
    m = patron.search(t)
    if m:
        nombre = m.group(1).strip()
        nombre = re.sub(r"^(a\s+|al\s+|la\s+|el\s+)", "", nombre)
        return nombre
    # variantes: "revoca acceso a juan"
    m2 = re.search(r"(?:revoca|revocar)\s+(?:acceso|permiso)\s+(?:a\s+|de\s+)?([a-záéíóúñ\s]+)$", t)
    if m2:
        return m2.group(1).strip()
    return None


def bucle_voz(memoria):
    contexto = collections.deque(maxlen=6)
    while True:
        texto = escuchar()
        if not texto:
            continue
        texto_limpio = texto.lower()

        # Salida
        if any(p in texto_limpio for p in ["adiós", "salir", "chao", "apagar sistema"]):
            hablar("Hasta pronto, me desconecto.")
            return "exit"

        # Cambiar usuario (reautenticación)
        if "cambiar usuario" in texto_limpio or "bloquear" in texto_limpio:
            hablar("Bloqueando. Se requiere autenticación nuevamente.")
            return "relock"

        # Lista de usuarios
        if "lista usuarios" in texto_limpio or "lista de usuarios" in texto_limpio:
            autorizados = memoria.get("autorizados", [])
            if autorizados:
                hablar("Usuarios autorizados: " + ", ".join(a.capitalize() for a in autorizados))
            else:
                hablar("No hay usuarios autorizados.")
            continue

        # Registro de nuevos rostros (solo aaron)
        nombre_nuevo = parsear_registro(texto_limpio)
        if nombre_nuevo:
            if memoria.get("nombre", "") != "aaron":
                hablar("Solo aaron puede registrar nuevos rostros.")
            else:
                registrar_nueva_persona(nombre_nuevo, memoria)
            continue

        # Borrado de rostros (solo aaron)
        nombre_borrar = parsear_borrado(texto_limpio)
        if nombre_borrar:
            if memoria.get("nombre", "") != "aaron":
                hablar("Solo aaron puede eliminar accesos.")
            else:
                eliminar_persona(nombre_borrar, memoria)
            continue

        # Detectar lugar
        lugar_detectado = None
        for palabra, led in LUGARES.items():
            if palabra in texto_limpio:
                lugar_detectado = led
                palabra_lugar = palabra
                break
        if not lugar_detectado:
            lugar_detectado = "ledSala"
            palabra_lugar = "sala"

        # Control de luces
        if any(p in texto_limpio for p in ["enciende", "prende", "activa", "ilumina"]):
            porcentaje = re.findall(r"\d+", texto_limpio)
            valor = 100 if not porcentaje else max(0, min(int(porcentaje[0]), 100))
            enviar_a_esp("on", lugar_detectado, valor)
            hablar(f"He encendido la luz de la {palabra_lugar} al {valor} por ciento.")
            continue

        elif any(p in texto_limpio for p in ["apaga", "desactiva", "oscurece"]):
            enviar_a_esp("off", lugar_detectado)
            hablar(f"He apagado la luz de la {palabra_lugar}.")
            continue

        elif "todas" in texto_limpio and "apaga" in texto_limpio:
            for l in LUGARES.values():
                enviar_a_esp("off", l)
            hablar("He apagado todas las luces de la casa.")
            continue

        elif any(p in texto_limpio for p in ["baja", "sube", "intensidad", "por ciento", "%", "brillo"]):
            porcentaje = re.findall(r"\d+", texto_limpio)
            valor = 100 if not porcentaje else max(0, min(int(porcentaje[0]), 100))
            enviar_a_esp("on", lugar_detectado, valor)
            hablar(f"He ajustado la luz de la {palabra_lugar} al {valor}% de intensidad.")
            continue

        elif "estado" in texto_limpio or "luces" in texto_limpio or "encendidas" in texto_limpio:
            estado = enviar_a_esp("estado")
            if estado:
                luces_encendidas = [k for k, v in estado.items() if v > 0]
                if luces_encendidas:
                    luces = ", ".join(luces_encendidas).replace("led", "")
                    hablar(f"Las luces encendidas son: {luces}.")
                else:
                    hablar("Todas las luces están apagadas.")
            continue

        # Hora y fecha
        if any(p in texto_limpio for p in ["qué hora", "dime la hora", "hora actual", "hora es"]):
            ahora_pe = datetime.datetime.now(zona_peru)
            hora_str = (
                ahora_pe.strftime("%I:%M %p")
                .lstrip("0")
                .replace("AM", "de la mañana")
                .replace("PM", "de la tarde")
            )
            hablar(f"La hora actual en Perú es {hora_str}.")
            continue

        if any(p in texto_limpio for p in ["qué día", "fecha de hoy", "día actual"]):
            ahora_pe = datetime.datetime.now(zona_peru)
            fecha_str = f"{dias_es[ahora_pe.strftime('%A')]} {ahora_pe.day:02d} de {meses_es[ahora_pe.strftime('%B')]} de {ahora_pe.year}"
            hablar(f"Hoy es {fecha_str}.")
            continue

        # Respuesta general
        respuesta = responder_llm(contexto, texto)
        hablar(respuesta)
        contexto.append(f"Airi: {respuesta}")
        memoria["ultima_fecha"] = fecha_hoy_str()
        guardar_memoria(memoria)


# =========================
# ========= MAIN ==========
# =========================
def main():
    memoria = cargar_memoria()
    asegurar_duenio(memoria)

    while True:
        nombre, conf = autenticar_usuario()
        if not nombre:
            hablar("No pude autenticar. Inténtalo de nuevo.")
            continue

        memoria["nombre"] = normalizar_nombre(nombre)
        if memoria["nombre"] not in memoria.get("autorizados", []):
            hablar("Rostro reconocido pero no autorizado. Acceso denegado.")
            continue

        hablar(f"Hola {memoria['nombre'].capitalize()}, soy Airi, tu asistente de tu casa inteligente.")
        accion = bucle_voz(memoria)
        if accion == "exit":
            break
        # si 'relock', vuelve a autenticar


if __name__ == "__main__":
    main()
