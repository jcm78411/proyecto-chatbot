import numpy as np
from .theory_generator import faiss_index, theory_data, embedding_model
from .data_loader import singleton_data
from .api_client import obtener_respuesta

def generar_respuesta_api(pregunta, contexto):
    prompt = f"Pregunta: {pregunta}\n\nContexto:\n{contexto}\n\nRespuesta:"
    return obtener_respuesta(prompt)

def clasificar_pregunta(pregunta):
    pregunta = pregunta.lower()
    if any(p in pregunta for p in ["código", "ejemplo", "implementación", "mostrar", "ver"]):
        return "codigo"
    elif any(p in pregunta for p in ["qué", "qué es", "definición", "significa", "consiste, hola"]):
        return "teoria"
    elif "tipos de singleton" in pregunta:
        return "tipos"
    else:
        return "mixta"

def chat_rag(pregunta_usuario):
    tipo_pregunta = clasificar_pregunta(pregunta_usuario)
    pregunta = pregunta_usuario.strip()

    query_embedding = embedding_model.encode([pregunta])[0]
    D, I = faiss_index.search(np.array([query_embedding]), k=1)

    if I[0][0] >= len(theory_data):
        tipo = "unknown"
    else:
        tipo = theory_data[I[0][0]]["type"]

    teoria = next((d["content"] for d in theory_data if d["type"] == tipo), "")
    ejemplo_codigo = next(
        (item["code"] for item in singleton_data if "implementation_type" in item and item["implementation_type"] == tipo),
        "",
    )

    if tipo_pregunta == "teoria":
        contexto = teoria
    elif tipo_pregunta == "codigo":
        contexto = ejemplo_codigo
    elif tipo_pregunta == "tipos":
        tipos_theoria = set(d["type"] for d in theory_data if d["type"] != "unknown")
        tipos_codigo = set(d["implementation_type"] for d in singleton_data if "implementation_type" in d)
        tipos_disponibles = sorted(tipos_theoria.union(tipos_codigo))
        return "Los tipos de implementación del patrón Singleton que conozco son:\n- " + "\n- ".join(tipos_disponibles)
    else:
        contexto = f"{teoria}\n\nEjemplo:\n{ejemplo_codigo}"

    return generar_respuesta_api(pregunta, contexto)
