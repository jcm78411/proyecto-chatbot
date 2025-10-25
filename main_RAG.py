import os
import json
import nltk
import keras
import pickle
import itertools
import flet as ft
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
import json
from dotenv import load_dotenv

########################################
########################################
"""VOZ CON TTSX3"""
########################################
########################################

import pyttsx3
import threading

# Inicializar engine de pyttsx3 (una sola instancia global)
engine = pyttsx3.init()

# Configuración básica
try:
    engine.setProperty("rate", 150)      # velocidad de voz
    engine.setProperty("volume", 1.0)    # volumen (0.0 a 1.0)
except Exception:
    pass

# Intentar seleccionar una voz en español si está disponible
try:
    voices = engine.getProperty("voices")
    chosen = None
    for v in voices:
        name = (v.name or "").lower()
        vid = (v.id or "").lower()
        # buscar pistas de "spanish" o "es"
        if "spanish" in name or "spanish" in vid or "es_" in vid or "es-" in vid or "es" == vid:
            chosen = v
            break
    if chosen:
        engine.setProperty("voice", chosen.id)
except Exception:
    pass

# Función interna que habla (bloqueante)
def _speak(text: str):
    try:
        engine.say(str(text))
        engine.runAndWait()
    except Exception:
        # proteger contra errores de TTS en tiempo de ejecución
        pass

# Lanza la voz en un hilo para no bloquear la UI
def speak_async(text: str):
    threading.Thread(target=_speak, args=(text,), daemon=True).start()

# Permite detener la reproducción si es necesario
def stop_speaking():
    try:
        engine.stop()
    except Exception:
        pass

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

nltk.download('punkt_tab')
nltk.download("punkt")
stemmer = LancasterStemmer()

dir_path = os.path.dirname(os.path.realpath(__file__))
extended_data_path = os.path.join(
    dir_path, "storage", "data", "singleton_dataset_extended.json"
)

with open(extended_data_path, "r", encoding="utf-8") as f:
    singleton_data = json.load(f)

with open("storage/data/singleton_dataset_extended.json", "r", encoding="utf-8") as f:
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

implementation_theory = {
    "classic": (
        "El patrón Singleton clásico crea una única instancia de una clase y la almacena en una variable estática dentro de la propia clase. "
        "Se proporciona un método de acceso (normalmente llamado `getInstance()` o similar) que devuelve esta instancia. "
        "Este enfoque es simple y funciona bien en entornos de un solo hilo, pero **no es seguro en entornos multihilo**, ya que dos hilos podrían crear instancias simultáneamente. "
        "No incluye control sobre la inicialización diferida ni medidas contra reflexión o serialización."
    ),
    "lazy": (
        "El Singleton lazy (inicialización diferida) retrasa la creación de la instancia hasta que esta es solicitada por primera vez. "
        "Esto reduce el uso innecesario de recursos si la instancia nunca llega a utilizarse. "
        "Sin embargo, **no es seguro en entornos multihilo a menos que se agregue sincronización** explícita. "
        "Es adecuado cuando la instancia es costosa de construir o puede no ser necesaria en todos los casos de uso."
    ),
    "thread_safe": (
        "El Singleton thread-safe garantiza que solo un hilo pueda crear la instancia, utilizando mecanismos como `synchronized`, `Lock`, o semáforos. "
        "Esto evita condiciones de carrera, pero **puede generar cuellos de botella por la sobrecarga de sincronización** si se accede frecuentemente al método. "
        "Es una solución correcta en aplicaciones concurrentes, pero puede no ser la más eficiente."
    ),
    "double_checked": (
        "La doble verificación mejora la eficiencia del thread-safe clásico. Primero verifica si la instancia ya fue creada sin sincronización. "
        "Solo si no existe, se entra en una sección crítica sincronizada y se vuelve a verificar. "
        "Esto evita sincronizar cada llamada después de la primera. "
        "**Requiere declarar la instancia como `volatile`** en Java para evitar problemas con el orden de ejecución en entornos multihilo."
    ),
    "enum": (
        "El patrón Singleton con `enum` en Java es considerado el más robusto y seguro. "
        "Permite al compilador garantizar que solo existe una instancia, incluso frente a ataques por reflexión, clonación o deserialización. "
        "Es muy conciso (`INSTANCE;`) y **recomendado por Joshua Bloch en Effective Java**. "
        "No permite lazy initialization, pero a cambio ofrece simplicidad y protección avanzada."
    ),
    "bill_pugh": (
        "Utiliza una clase interna estática (`static inner class`) que contiene la instancia singleton. "
        "La clase no se carga hasta que se llama al método `getInstance()`, lo que garantiza inicialización diferida y seguridad en múltiples hilos. "
        "**No requiere sincronización manual**, ya que la JVM asegura la carga del `classloader` de manera segura. "
        "Es considerado uno de los enfoques más elegantes y eficientes en Java moderno."
    ),
    "synchronized": (
        "En este patrón, el método `getInstance()` está completamente sincronizado, lo que significa que solo un hilo puede ejecutarlo a la vez. "
        "Esto resuelve los problemas de concurrencia, pero **introduce una penalización de rendimiento**, ya que incluso cuando la instancia ya existe, se sigue sincronizando. "
        "Es fácil de implementar pero no ideal para escenarios con muchas llamadas a `getInstance()`."
    ),
    "reflection_safe": (
        "Previene ataques por reflexión que podrían permitir la creación de múltiples instancias. "
        "El constructor privado lanza una excepción si ya existe una instancia, validando una variable de control antes de permitir la construcción. "
        "**Importante en Java**, donde es posible acceder a constructores privados mediante APIs de reflexión. "
        "No es infalible, pero añade una capa importante de seguridad a la implementación Singleton."
    ),
    "eager": (
        "La instancia se crea en el momento de la carga de la clase, ya sea al declarar directamente el campo o mediante un bloque estático. "
        "Esto **garantiza la disponibilidad inmediata y seguridad en multihilo**, pero puede generar uso innecesario de memoria si la instancia nunca se usa. "
        "Es útil cuando se sabe con certeza que la instancia será requerida."
    ),
    "static_block": (
        "Similar al eager, pero permite envolver la creación de la instancia en un bloque `static {}` para manejar posibles excepciones. "
        "Esto es útil cuando la instancia depende de operaciones que puedan fallar (por ejemplo, lectura de archivos de configuración o acceso a recursos externos). "
        "Combina seguridad de carga temprana con manejo de errores."
    ),
    "volatile": (
        "Usa la palabra clave `volatile` para garantizar la visibilidad inmediata de la instancia entre hilos. "
        "Suele combinarse con la doble verificación para lograr **eficiencia, seguridad y correcta propagación del valor** en entornos multihilo. "
        "Sin `volatile`, un hilo podría ver una versión parcialmente construida del objeto debido a reordenamientos del compilador o CPU."
    ),
    "registry": (
        "Este patrón extiende el Singleton tradicional permitiendo múltiples instancias únicas, cada una asociada a una clave (nombre, tipo, contexto). "
        "Todas las instancias se almacenan en un `map` o `diccionario` central. "
        "Es útil en aplicaciones grandes donde se requiere más de un singleton especializado. "
        "Permite controlar y desacoplar el acceso a diferentes instancias desde un único punto de gestión."
    ),
    "inner_static_class": (
        "Es una forma más precisa de referirse al patrón Bill Pugh. Utiliza una clase estática interna para mantener la instancia singleton. "
        "Aprovecha la carga diferida del `classloader`, lo cual garantiza inicialización perezosa y segura en múltiples hilos sin sincronización explícita. "
        "Ideal en Java, ya que combina eficiencia, simplicidad y robustez."
    ),
    "metaclass": (
        "En Python, una metaclase puede usarse para modificar el comportamiento de creación de clases. "
        "Al sobreescribir el método `__call__`, se puede interceptar cada intento de instanciación y retornar siempre la misma instancia. "
        "Permite aplicar Singleton de forma genérica a múltiples clases, ideal en arquitecturas extensibles o frameworks personalizados."
    ),
    "module": (
        "En Python, los módulos ya son objetos singleton por naturaleza. Una vez importado, el módulo se almacena en `sys.modules` y se reutiliza en cada importación posterior. "
        "Esto hace innecesario implementar un patrón Singleton explícito en muchos casos, especialmente para recursos compartidos o servicios globales. "
        "Es una solución muy simple y efectiva en scripts y proyectos pequeños."
    ),
    "borg": (
        "El patrón Borg o Monostate, propuesto por Alex Martelli, permite crear múltiples instancias de una clase, pero todas comparten el mismo estado. "
        "Esto se logra haciendo que todas las instancias compartan el mismo diccionario interno (`__dict__`), o una estructura compartida como `__shared_state`. "
        "Es una alternativa flexible al Singleton cuando se desea desacoplar la identidad del objeto del estado compartido."
    ),
    "unknown": (
        "Esta categoría se reserva para implementaciones atípicas o que no encajan claramente en las categorías conocidas. "
        "Puede incluir combinaciones híbridas, enfoques dependientes de framework, o técnicas aún no clasificadas. "
        "Requiere análisis manual para determinar su viabilidad y propósito real."
    ),
}

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

words, all_words, tags, aux, auxA, auxB, training, exit_data = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)

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

    ignore_words = [
        "?",
        "¿",
        "!",
        ".",
        ",",
        ":",
        ";",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "'",
        '"',
        "`",
        "~",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "-",
        "_",
        "+",
        "=",
    ]
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
    model.fit(training, exit_data, epochs=2000, batch_size=128, validation_split=0.1, verbose=1)  # type: ignore
    model.save(model_path)
    print("=" * 40)
    print("Modelo guardado en disco 💾")
    print("=" * 40)
    print("=" * 40)


load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY no encontrada en .env")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://tusitio.com",
    "X-Title": "Chat Flet Llama3",
}

def obtener_respuesta(texto):
    try:
        # Construcción del cuerpo del mensaje
        data = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [
                {"role": "system", "content": "Eres un asistente útil, experto en ingenieria de sistemas, que responde con claridad exclusivamente sobre patrones de diseño singleton en java"},
                {"role": "user", "content": texto}
            ],
            "max_tokens": 512,
            "temperature": 0.8
        }

        # Petición a la API de OpenRouter
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "<YOUR_SITE_URL>",  # (Opcional)
                "X-Title": "<YOUR_SITE_NAME>",       # (Opcional)
            },
            data=json.dumps(data)
        )

        # Comprobación de errores HTTP
        if response.status_code != 200:
            print(f"⚠️ Error HTTP {response.status_code}: {response.text}")
            return "Ocurrió un error al consultar el modelo."

        # Procesar la respuesta JSON
        result = response.json()

        # Extraer el contenido del mensaje del modelo
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        import traceback

        print("⚠️ Error al consultar el modelo:")
        traceback.print_exc()
        return "Lo siento, ocurrió un error al procesar tu solicitud. Inténtalo de nuevo."


def generar_respuesta_api(pregunta, contexto):
    prompt = f"Pregunta: {pregunta}\n\nContexto:\n{contexto}\n\nRespuesta:"
    return obtener_respuesta(prompt)

def clasificar_pregunta(pregunta):
    pregunta = pregunta.lower()
    if any(
        p in pregunta for p in ["código", "ejemplo", "implementación", "mostrar", "ver"]
    ):
        return "codigo"
    elif any(
        p in pregunta for p in ["qué", "qué es", "definición", "significa", "consiste, hola"]
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
    D, I = faiss_index.search(np.array([query_embedding]), k=1) # type: ignore

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

    if tipo_pregunta == "teoria":
        contexto = teoria
    elif tipo_pregunta == "codigo":
        contexto = ejemplo_codigo
    elif tipo_pregunta == "tipos":
        tipos_theoria = set(d["type"] for d in theory_data if d["type"] != "unknown")
        tipos_codigo = set(
            d["implementation_type"] for d in singleton_data if "implementation_type" in d
        )

        tipos_disponibles = sorted(tipos_theoria.union(tipos_codigo))
        return (
            "Los tipos de implementación del patrón Singleton que conozco son:\n- "
            + "\n- ".join(tipos_disponibles)
        )

    else:
        contexto = f"{teoria}\n\nEjemplo:\n{ejemplo_codigo}"

    return generar_respuesta_api(pregunta, contexto)


# ====================
# Función de chatbot
# ====================

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

        respuesta = chat_rag(text)
        speak_async(respuesta)

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
                            font_family="Courier New",
                            no_wrap=False,
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
