import threading
import pyttsx3
import queue

# Inicializa el motor y estructuras de control
engine = pyttsx3.init()
tts_queue = queue.Queue()
stop_flag = threading.Event()

def tts_worker():
    """Hilo principal que maneja la cola de voz (único run loop activo)."""
    while True:
        text = tts_queue.get()
        if text is None:
            break

        if not stop_flag.is_set():
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Error TTS: {e}")
        tts_queue.task_done()

# Iniciar el hilo del motor una sola vez
threading.Thread(target=tts_worker, daemon=True).start()

def _speak(text: str):
    """Envía texto a la cola de voz, solo si no está silenciado."""
    if not stop_flag.is_set():
        tts_queue.put(str(text))

def stop_speaking():
    """Detiene inmediatamente el habla actual."""
    stop_flag.set()
    try:
        engine.stop()
        # Limpia la cola (para no leer mensajes futuros)
        while not tts_queue.empty():
            tts_queue.get_nowait()
            tts_queue.task_done()
    except Exception:
        pass
    finally:
        # Restablecer flag para poder volver a hablar luego
        stop_flag.clear()
