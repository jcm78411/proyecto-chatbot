import os
import json
import nltk
import keras
import pickle
import random
import itertools
import flet as ft
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sentence_transformers import SentenceTransformer
import faiss
import tf_keras
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Cargar T5 (puedes usar otras variantes más grandes si tu RAM lo permite)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

nltk.download("punkt")
stemmer = LancasterStemmer()

dir_path = os.path.dirname(os.path.realpath(__file__))
enriched_data_path = os.path.join(
    dir_path, "data_bot", "data_bot-main", "singleton_dataset_extended.json"
)

with open(enriched_data_path, "r", encoding="utf-8") as f:
    singleton_data = json.load(f)

with open("storage/data/singleton_dataset_enriched.json", "r", encoding="utf-8") as f:
    singleton_data = json.load(f)

intents = {}
for item in singleton_data:
    tag = item.get("implementation_type", "unknown")
    code = item["code"]
    example = f"Ejemplo de código ({tag}):\n\n{code.strip()}"
    if tag not in intents:
        intents[tag] = {
            "tag": tag,
            "patterns": [
                f"¿Cómo se implementa un singleton tipo {tag}?",
                f"¿Tienes una implementación {tag}?",
                f"¿Me puedes mostrar un singleton con el estilo {tag}?",
                f"¿Cuál es el código de singleton versión {tag}?",
                f"¿Cómo hago una clase singleton usando el método {tag}?",
                f"Dame un ejemplo de singleton con enfoque {tag}",
                f"Quiero un singleton tipo {tag}, ¿puedes ayudarme?",
                f"¿Hay alguna forma de implementar singleton como {tag}?",
                f"¿Puedes escribirme una clase singleton tipo {tag}?",
                f"Explícame un singleton con patrón {tag}",
                f"¿En qué consiste una implementación {tag} del patrón singleton?",
                f"¿Tienes código {tag} de singleton?",
                f"¿Puedes darme una muestra del singleton {tag}?",
                f"¿Cómo luce una clase singleton con el estilo {tag}?",
                f"¿Puedes generar un ejemplo {tag} para el patrón singleton?",
            ],
            "responses": [example],
        }
    else:
        intents[tag]["responses"].append(example)

intents_list = {"intents": list(intents.values())}

implementation_theory = {}


theory_dataset = []
for intent in intents_list["intents"]:
    tag = intent["tag"]
    content = implementation_theory.get(
        tag, f"No se encontró teoría definida para el tipo {tag}."
    )
    theory_dataset.append({"type": tag, "content": content})

with open(
    os.path.join(dir_path, "data_bot", "data_bot-main", "theory_dataset.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(theory_dataset, f, indent=2, ensure_ascii=False)

print("Archivo 'theory_dataset.json' generado correctamente ✅")

with open(
    os.path.join(dir_path, "data_bot", "data_bot-main", "theory_dataset.json"),
    "r",
    encoding="utf-8",
) as f:
    theory_data = json.load(f)

# Inicializar modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Crear lista de textos y tipos
texts = [doc["content"] for doc in theory_data]
types = [doc["type"] for doc in theory_data]

# Crear embeddings
theory_embeddings = embedding_model.encode(texts, convert_to_numpy=True)

# Crear índice FAISS
embedding_dim = theory_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(theory_embeddings)  # type: ignore

with open(
    os.path.join(dir_path, "data_bot", "data_bot-main", "data.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(intents_list, f, indent=2, ensure_ascii=False)

words, all_words, tags, aux, auxA, auxB, training, exit_data = ()

try:
    if not os.path.exists("Entrenamiento"):
        os.makedirs("Entrenamiento")

    with open("Entrenamiento/brain.words.pickle", "rb") as pickle_Brain:
        all_words, tags, training, exit_data = pickle.load(pickle_Brain)
except:
    for intent in intents_list["intents"]:
        for pattern in intent["patterns"]:
            auxWords = word_tokenize(pattern)
            auxA.append(auxWords)
            auxB.append(auxWords)
            aux.append(intent["tag"])

    ignore_words = []
    for w in auxB:
        words.append(w)

    words = sorted(set(list(itertools.chain.from_iterable(words))))
    tags = sorted(set(aux))
    all_words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    all_words = sorted(set(all_words))
    null_exit = [0 for _ in range(len(tags))]

    for i, document in enumerate(auxA):
        bucket = []
        auxWords = [stemmer.stem(w.lower()) for w in document if w not in ignore_words]
        for w in all_words:
            bucket.append(1 if w in auxWords else 0)
        exit_row = null_exit[:]
        exit_row[tags.index(aux[i])] = 1
        training.append(bucket)
        exit_data.append(exit_row)

    training = np.array(training)
    exit_data = np.array(exit_data)

    if not os.path.exists("EntrenamientoPickle"):
        os.makedirs("EntrenamientoPickle")

    with open("EntrenamientoPickle/brain.words.pickle", "wb") as pickle_Brain:
        pickle.dump([all_words, tags, training, exit_data], pickle_Brain)

# ============================
# Red neuronal con TensorFlow
# ============================

model_path = os.path.join(dir_path, "EntrenamientoPickle", "brain_model.h5")
if os.path.isfile(model_path):
    model = keras.models.load_model(model_path)
    print("=" * 40)
    print("Modelo cargado desde el disco 💽")
    print("=" * 40)
else:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(len(training[0]),)),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(exit_data[0]), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # epoch=1700
    model.fit(training, exit_data, epochs=1700, batch_size=128, validation_split=0.1, verbose=1)  # type: ignore
    model.save(model_path)
    print("=" * 40)
    print("Modelo guardado en disco 💾")
    print("=" * 40)
    print("=" * 40)


def generar_respuesta_t5(pregunta, contexto):
    prompt = f"Pregunta: {pregunta}\n\nContexto:\n{contexto}\n\nRespuesta:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model_t5.generate(
        **inputs, max_length=256, num_beams=5, early_stopping=True
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return respuesta


def clasificar_pregunta(pregunta):
    pregunta = pregunta.lower()
    if any(
        p in pregunta for p in ["código", "ejemplo", "implementación", "mostrar", "ver"]
    ):
        return "codigo"
    elif any(
        p in pregunta for p in ["qué", "qué es", "definición", "significa", "consiste"]
    ):
        return "teoria"
    elif "tipos de singleton" in pregunta:
        return "tipos"
    else:
        return "mixta"


def chat_rag(pregunta_usuario):
    tipo_pregunta = clasificar_pregunta(pregunta_usuario)
    pregunta = pregunta_usuario.strip()

    # Embedding y búsqueda semántica
    query_embedding = embedding_model.encode([pregunta])[0]
    D, I = faiss_index.search(np.array([query_embedding]), k=1)

    if I[0][0] >= len(theory_data):
        tipo = "unknown"
    else:
        tipo = theory_data[I[0][0]]["type"]

    teoria = next((d["content"] for d in theory_data if d["type"] == tipo), "")
    ejemplo_codigo = next(
        (
            item["code"]
            for item in singleton_data
            if "implementation_type" in item and item["implementation_type"] == tipo
        ),
        "",
    )

    # Generar contexto según tipo de pregunta
    if tipo_pregunta == "teoria":
        contexto = teoria
    elif tipo_pregunta == "codigo":
        contexto = ejemplo_codigo
    elif tipo_pregunta == "tipos":
        tipos_disponibles = sorted(
            set([d["type"] for d in theory_data if d["type"] != "unknown"])
        )
        return (
            "Los tipos de implementación del patrón Singleton que conozco son:\n- "
            + "\n- ".join(tipos_disponibles)
        )
    else:  # mixta o desconocida
        contexto = f"{teoria}\n\nEjemplo:\n{ejemplo_codigo}"

    return generar_respuesta_t5(pregunta, contexto)


# ====================
# Función de chatbot


def main(page: ft.Page):

    win_width = 1366
    win_height = 768

    screen_width = 1366
    screen_height = 768

    page.window_left = (screen_width - win_width) // 2  # type: ignore
    page.window_top = (screen_height - win_height) // 2  # type: ignore

    page.window_width = win_width  # type: ignore
    page.window_height = win_height  # type: ignore
    page.window_resizable = True  # type: ignore

    page.title = "Chat con IA"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    chat_display = ft.Column(scroll="always", expand=True)  # type: ignore
    user_input = ft.TextField(label="Mensaje", expand=True, autofocus=True)
    send_btn = ft.ElevatedButton("Enviar", icon=ft.Icons.SEND)
    clear_btn = ft.ElevatedButton("Limpiar chat", icon=ft.Icons.DELETE)

    espera_label = ft.Text("🤖 Pensando...", visible=False)
    spinner = ft.ProgressRing(visible=False)

    def send_message(e):
        text = user_input.value.strip()  # type: ignore
        if not text:
            return

        chat_display.controls.append(
            ft.Row(
                [
                    ft.Container(
                        ft.Text(f"🧑‍💻 Usuario: {text}"),
                        padding=10,
                        bgcolor=ft.Colors.BLUE,
                        border_radius=10,
                        alignment=ft.Alignment(1, 0),
                    )
                ],
                alignment=ft.MainAxisAlignment.END,
            )
        )
        user_input.value = ""
        user_input.focus()
        user_input.update()
        page.update()

        espera_label.visible = True
        spinner.visible = True
        espera_label.update()
        spinner.update()

        respuesta = chat_rag(text)  # Reemplaza con tu lógica de respuesta

        espera_label.visible = False
        spinner.visible = False
        espera_label.update()
        spinner.update()
        chat_display.controls.append(
            ft.Row(
                [
                    ft.Container(
                        ft.Text(
                            f"🤖 IA:\n{respuesta}",
                            size=14,
                            font_family="Courier New",  # Fuente monoespaciada
                            no_wrap=False,  # Permite saltos de línea
                            expand=True,
                        ),
                        padding=10,
                        bgcolor=ft.Colors.GREEN,
                        border_radius=10,
                        alignment=ft.Alignment(1, 0),
                        width=700,
                    )
                ],
                alignment=ft.MainAxisAlignment.START,
            )
        )

        page.update()

    def clear_chat(e):
        chat_display.controls.clear()
        page.update()

    send_btn.on_click = send_message
    clear_btn.on_click = clear_chat
    user_input.on_submit = send_message

    page.add(
        ft.Column(
            [
                ft.Text("🤖 Bienvenido al Chat IA", size=22, weight="bold", text_align="center"),  # type: ignore
                espera_label,
                spinner,
                ft.Container(
                    chat_display,
                    height=500,
                    bgcolor=ft.Colors.GREY,
                    border_radius=10,
                    padding=ft.Padding(left=50, right=50, top=10, bottom=10),
                    width=1365,
                    margin=ft.Margin(top=30, bottom=20, left=50, right=50),
                ),
                ft.Row(
                    [user_input, ft.Column([send_btn, clear_btn])],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )


if __name__ == "__main__":
    ft.app(target=main)
