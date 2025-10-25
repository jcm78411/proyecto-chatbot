import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY no encontrada en .env")

# === CONFIGURACIÓN DE OPENROUTER ===
API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://openrouter.ai",
    "X-Title": "Chat Flet Llama3",
}


def obtener_respuesta(texto):
    try:
        data = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un asistente útil, experto en ingenieria de sistemas, que responde con claridad exclusivamente sobre patrones de diseño singleton en java.",
                },
                {"role": "user", "content": texto},
            ],
            "max_tokens": 512,
            "temperature": 0.8,
        }

        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))

        # Verificar si hubo error HTTP
        if response.status_code != 200:
            print(f"Error HTTP {response.status_code}: {response.text}")
            return "Ocurrió un error al consultar el modelo."

        # Procesar la respuesta JSON
        result = response.json()

        # Extraer texto de respuesta
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        import traceback

        print("Error al consultar el modelo:")
        traceback.print_exc()
        return (
            "Lo siento, ocurrió un error al procesar tu solicitud. Inténtalo de nuevo."
        )
