import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()  # carga variables desde el .env al entorno

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY no encontrada en .env")

# === CONFIGURACIÓN DE OPENROUTER ===
# API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    # Usa un dominio real o al menos algo válido (requerido por OpenRouter)
    "HTTP-Referer": "https://openrouter.ai",  
    "X-Title": "Chat Flet Llama3",
}

def obtener_respuesta(texto):
    try:
        data = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [
                {"role": "system", "content": "Eres un asistente útil que responde con claridad."},
                {"role": "user", "content": texto}
            ],
            "max_tokens": 512,
            "temperature": 0.8
        }

        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))

        # Verificar si hubo error HTTP
        if response.status_code != 200:
            print(f"⚠️ Error HTTP {response.status_code}: {response.text}")
            return "Ocurrió un error al consultar el modelo."

        # Procesar la respuesta JSON
        result = response.json()

        # Extraer texto de respuesta
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        import traceback

        print("⚠️ Error al consultar el modelo:")
        traceback.print_exc()
        return "Lo siento, ocurrió un error al procesar tu solicitud. Inténtalo de nuevo."
