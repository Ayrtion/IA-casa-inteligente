import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("=== Voces disponibles ===")
for v in voices:
    print(v.id, "-", v.name)

engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 170)
engine.say("Hola, esta es una prueba de voz de Airi.")
engine.runAndWait()
