import os, re, time, datetime, json, collections, requests
import speech_recognition as sr
import pytz
from llama_cpp import Llama
import torch
import numpy as np
import sounddevice as sd

# === CONFIGURACION ===
MODEL_PATH = "modelos/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
ESP_HOST = "http://192.168.43.150" 
ESP_TIMEOUT = 3
MEM_FILE = "memoria_airi.json"

# === MEMORIA ===
def cargar_memoria():
    if os.path.exists(MEM_FILE):
        with open(MEM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"nombre": None, "emocion": None, "ultima_fecha": None}

def guardar_memoria(mem):
    with open(MEM_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

memoria = cargar_memoria()

# === MAPA DE LUGARES ===
LUGARES = {
    "cocina": "ledCocina",
    "cochera": "ledCochera",
    "sala": "ledSala",
    "oficina": "ledOficina",
    "pasillo": "ledPasillo",
    "cuarto principal": "ledCuartoPrincipal",
    "cuarto de visita": "ledCuartoVisita"
}

# === CONEXIÓN CON ESP8266 ===
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

# === VOZ (Silero) ===
def hablar(texto):
    print(f"Airi: {texto}")
    try:
        global silero_model, silero_loaded
        if 'silero_loaded' not in globals():
            torch.set_num_threads(2)
            silero_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='es',
                speaker='v3_es'
            )
            silero_loaded = True

        audio = silero_model.apply_tts(text=texto, speaker='es_1', sample_rate=48000)
        sd.play(audio.numpy(), 48000)
        sd.wait()
    except Exception as e:
        print(f"[Error voz] {e}")

# === MODELO ===
print("Cargando modelo...")
llm = Llama(model_path=MODEL_PATH, n_ctx=768, n_threads=8, verbose=False)
print("✅ Modelo cargado correctamente\n")

# === AUDIO ===
r = sr.Recognizer()
mic = sr.Microphone()

def escuchar():
    with mic as source:
        print("🎤 Escuchando...")
        r.adjust_for_ambient_noise(source, duration=0.4)
        audio = r.listen(source)
    try:
        texto = r.recognize_google(audio, language="es-ES")
        print(f"Tú: {texto}")
        return texto
    except:
        hablar("No te entendí, repite por favor.")
        return None

# === FECHA ===
dias_es = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo"
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
    "December": "diciembre"
}

zona_peru = pytz.timezone("America/Lima")
ahora = datetime.datetime.now(zona_peru)
fecha_hoy = f"{dias_es[ahora.strftime('%A')]} {ahora.day:02d} de {meses_es[ahora.strftime('%B')]} de {ahora.year}"

# === SALUDO ===
hablar(f"Hola {memoria['nombre']}, soy Airi, tu asistente virtual de tu casa inteligente.")

# === CONTEXTO ===
contexto = collections.deque(maxlen=6)

# === LOOP PRINCIPAL ===
while True:
    texto = escuchar()
    if not texto:
        continue

    texto_limpio = texto.lower()

    # === Cierre ===
    if any(p in texto_limpio for p in ["adiós", "salir", "chao", "apagar sistema"]):
        hablar("Hasta pronto, me desconecto.")
        break

    # === DETECTAR LUGAR ===
    lugar_detectado = None
    for palabra, led in LUGARES.items():
        if palabra in texto_limpio:
            lugar_detectado = led
            palabra_lugar = palabra
            break
    if not lugar_detectado:
        lugar_detectado = "ledSala"
        palabra_lugar = "sala"

    # === CONTROL DE LUCES ===
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

    # === HORA Y FECHA ===
    if any(p in texto_limpio for p in ["qué hora", "dime la hora", "hora actual", "hora es"]):
        ahora_pe = datetime.datetime.now(zona_peru)
        hora_str = ahora_pe.strftime("%I:%M %p").lstrip("0").replace("AM", "de la mañana").replace("PM", "de la tarde")
        hablar(f"La hora actual en Perú es {hora_str}.")
        continue

    if any(p in texto_limpio for p in ["qué día", "fecha de hoy", "día actual"]):
        ahora_pe = datetime.datetime.now(zona_peru)
        fecha_str = f"{dias_es[ahora_pe.strftime('%A')]} {ahora_pe.day:02d} de {meses_es[ahora_pe.strftime('%B')]} de {ahora_pe.year}"
        hablar(f"Hoy es {fecha_str}.")
        continue

    # === RESPUESTA GENERAL ===
    contexto.append(f"Tú: {texto}")
    historial = "\n".join(contexto)

    prompt = f"""<|start_header_id|>system<|end_header_id|>
Eres Airi, asistente local creada por Aaron Yair Torres Novella.
Hablas español natural, cálido y breve.
Fecha: {fecha_hoy}.
<|start_header_id|>user<|end_header_id|>
{historial}
<|start_header_id|>assistant<|end_header_id|>"""

    output = llm(prompt, max_tokens=100, temperature=0.7)
    respuesta = output["choices"][0]["text"].strip()
    respuesta = re.sub(r"<\|.*?\|>", "", respuesta).replace("\n", " ").strip()

    hablar(respuesta)
    contexto.append(f"Airi: {respuesta}")
    memoria["ultima_fecha"] = fecha_hoy
    guardar_memoria(memoria)
