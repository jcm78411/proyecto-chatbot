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

nltk.download('punkt')
stemmer = LancasterStemmer()    

dir_path = os.path.dirname(os.path.realpath(__file__))
enriched_data_path = os.path.join(dir_path, "data_bot", "data_bot-main", "singleton_dataset_extended.json")

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
            "patterns" : [
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
                f"¿Puedes generar un ejemplo {tag} para el patrón singleton?"
            ],
            "responses": [example]
        }
    else:
        intents[tag]["responses"].append(example)

intents_list = {"intents": list(intents.values())}

implementation_theory = {
    "classic": (
        "El patrón Singleton clásico crea una única instancia de una clase y la almacena en una variable estática dentro de la propia clase. "
        "Cada vez que se solicita, retorna la misma instancia. Esta implementación no es segura para entornos multihilo."
    ),
    "lazy": (
        "El Singleton lazy (o de inicialización diferida) crea la instancia solo cuando se llama por primera vez al método de acceso. "
        "Esto permite ahorrar recursos si la instancia nunca llega a usarse, pero requiere mecanismos de sincronización si se usa en entornos concurrentes."
    ),
    "thread_safe": (
        "El Singleton thread-safe está diseñado para funcionar correctamente en entornos multihilo. Usa sincronización (por ejemplo, `synchronized` en Java) "
        "para garantizar que solo un hilo cree la instancia, evitando condiciones de carrera. Puede ser menos eficiente por la sobrecarga de sincronización."
    ),
    "double_checked": (
        "Este patrón mejora la eficiencia del Singleton thread-safe mediante doble verificación: primero comprueba si la instancia existe sin bloqueo, "
        "y solo sincroniza cuando es necesario. Es eficiente pero requiere cuidado, especialmente en versiones antiguas de Java (pre-Java 5)."
    ),
    "enum": (
        "El patrón Singleton usando `enum` en Java es la forma más sencilla y segura de crear un singleton. "
        "Protege contra problemas de serialización, clonación y reflexión. El compilador asegura que solo exista una instancia del enum."
    ),
    "bill_pugh": (
        "Este patrón utiliza una clase interna estática (`static inner class`) para mantener la instancia única. "
        "La clase interna no se carga hasta que se invoca el método `getInstance()`, lo que permite inicialización perezosa y es seguro en entornos multihilo."
    ),
    "synchronized": (
        "El patrón Singleton sincronizado marca el método `getInstance()` como `synchronized`, lo que garantiza exclusión mutua durante la creación de la instancia. "
        "Es seguro para hilos, pero puede causar cuellos de botella al sincronizar incluso cuando la instancia ya ha sido creada."
    ),
    "reflection_safe": (
        "Un Singleton reflection-safe evita la creación de nuevas instancias mediante reflexión al lanzar una excepción desde el constructor si la instancia ya existe. "
        "Esto añade una capa de protección extra en entornos donde se pueda acceder a la clase por medios no convencionales."
    ),
    "eager": (
        "El Singleton eager (inicialización temprana) crea la instancia en el momento de la carga de la clase. "
        "Es simple y seguro en entornos multihilo, pero puede generar un gasto innecesario de recursos si la instancia nunca se utiliza."
    ),
    "static_block": (
        "El patrón Singleton con `static block` es similar al eager, pero permite manejar excepciones durante la creación de la instancia. "
        "Es útil cuando la creación puede lanzar errores que deben ser tratados en tiempo de carga."
    ),
    "volatile": (
        "El patrón Singleton con `volatile` combina inicialización perezosa, doble verificación y seguridad en hilos. "
        "La palabra clave `volatile` garantiza que la instancia sea visible para todos los hilos inmediatamente después de ser creada."
    ),
    "registry": (
        "El Singleton Registry permite mantener múltiples instancias singleton gestionadas en un mapa o registro. "
        "Es útil para gestionar varios objetos únicos (por tipo, nombre o contexto) en sistemas complejos y desacoplados."
    ),
    "inner_static_class": (
        "Este patrón utiliza una clase interna estática para almacenar la instancia singleton. "
        "Aprovecha la carga diferida del classloader de Java, lo que permite una inicialización perezosa y segura en múltiples hilos sin sincronización explícita."
    ),
    "metaclass": (
        "En Python, el patrón Singleton con metaclase redefine el método `__call__` de la metaclase para asegurarse de que solo se cree una instancia. "
        "Es una forma poderosa y flexible, adecuada cuando se necesita aplicar Singleton a múltiples clases desde una misma lógica centralizada."
    ),
    "module": (
        "En Python, los módulos son por naturaleza singletons. Una vez importado un módulo, se almacena en `sys.modules`, lo que garantiza que futuras importaciones "
        "usen la misma instancia. Esto hace innecesario implementar explícitamente el patrón Singleton en muchos casos."
    ),
    "borg": (
        "El patrón Borg (también conocido como Monostate) permite múltiples instancias de una clase, pero todas comparten el mismo estado interno. "
        "Esto se logra haciendo que todas las instancias compartan el mismo diccionario `__dict__` o un atributo común `__shared_state`."
    ),
    "unknown": (
        "Este tipo de singleton no está categorizado aún. Puede requerir revisión manual para determinar su propósito y mecanismo de implementación."
    ),
}

theory_dataset = []
for intent in intents_list["intents"]:
    tag = intent["tag"]
    content = implementation_theory.get(tag, f"No se encontró teoría definida para el tipo {tag}.")
    theory_dataset.append({
        "type": tag,
        "content": content
    })

with open(os.path.join(dir_path, "data_bot", "data_bot-main", "theory_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(theory_dataset, f, indent=2, ensure_ascii=False)

print("Archivo 'theory_dataset.json' generado correctamente ✅")

with open(os.path.join(dir_path, "data_bot", "data_bot-main", "theory_dataset.json"), "r", encoding="utf-8") as f:
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
faiss_index.add(theory_embeddings) # type: ignore

with open(os.path.join(dir_path, "data_bot", "data_bot-main", "data.json"), "w", encoding="utf-8") as f:
    json.dump(intents_list, f, indent=2, ensure_ascii=False)

words, all_words, tags, aux, auxA, auxB, training, exit_data = [], [], [], [], [], [], [], []

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

    ignore_words = ["?", "¿", "!", ".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "'", '"', "`", "~", "@", "#", "$", "%", "^", "&", "*", "-", "_", "+", "="]
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
    model = keras.Sequential([
        keras.layers.Input(shape=(len(training[0]),)),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(exit_data[0]), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # epoch=1700
    model.fit(training, exit_data, epochs=1700, batch_size=128, validation_split=0.1, verbose=1) # type: ignore
    model.save(model_path)
    print("=" * 40)
    print("Modelo guardado en disco 💾")
    print("=" * 40)
    print("=" * 40)


# ====================
# Función de chatbot
def chat(text):
    text_lower = text.lower().strip()
    
    tipo_singleton_preguntas = [
        "¿qué tipos de patrones de diseño singleton conoces?",
        "¿cuáles son los tipos de singleton?",
        "¿qué variantes del patrón singleton existen?",
        "tipos de singleton",
        "¿qué implementaciones de singleton hay?",
        "¿qué formas de hacer singleton existen?",
        "clases de singleton",
        "variantes de singleton",
        "¿me puedes listar los tipos de singleton?",
        "diferentes maneras de implementar singleton"
    ]
    
    if any(p in text_lower for p in tipo_singleton_preguntas):
        lista_tags = sorted(set(tags))
        return "📌 *Tipos de patrones Singleton conocidos:*\n- " + "\n- ".join(lista_tags)

    # === Parte del modelo de predicción ===
    bow = [0] * len(all_words)
    words_in_input = [stemmer.stem(w.lower()) for w in word_tokenize(text)]
    for idx, w in enumerate(all_words):
        if w in words_in_input:
            bow[idx] = 1

    res = model.predict(np.array([bow]))[0] # type: ignore
    tag_index = np.argmax(res)
    tag = tags[tag_index]

    if res[tag_index] < 0.6:
        return "No entendí tu pregunta. ¿Podrías reformularla?"

    query_embedding = embedding_model.encode([text])[0]
    D, I = faiss_index.search(np.array([query_embedding]), k=1) # type: ignore
    if I[0][0] >= len(theory_data):
        teoria_relacionada = "No se encontró teoría relacionada."
    else:
        teoria_relacionada = theory_data[I[0][0]]["content"]

    if tag not in intents:
        return "Lo siento, no tengo ejemplos para ese tipo de implementación."

    codigo = random.choice(intents[tag]["responses"])

    respuesta = f"📚 *Teoría relacionada ({tag}):*\n{teoria_relacionada}\n\n💻 *Ejemplo de código:*\n{codigo}"
    return respuesta
    
def main(page: ft.Page):
    
    win_width = 1366
    win_height = 768

    screen_width = 1366
    screen_height = 768

    page.window_left = (screen_width - win_width) // 2 # type: ignore
    page.window_top = (screen_height - win_height) // 2 # type: ignore

    page.window_width = win_width # type: ignore
    page.window_height = win_height # type: ignore
    page.window_resizable = True # type: ignore
    
    page.title = "Chat con IA"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    chat_display = ft.Column(scroll="always", expand=True) # type: ignore
    user_input = ft.TextField(label="Mensaje", expand=True, autofocus=True)
    send_btn = ft.ElevatedButton("Enviar", icon=ft.Icons.SEND)
    clear_btn = ft.ElevatedButton("Limpiar chat", icon=ft.Icons.DELETE)

    espera_label = ft.Text("🤖 Pensando...", visible=False)
    spinner = ft.ProgressRing(visible=False)

    def send_message(e):
        text = user_input.value.strip() # type: ignore
        if not text:
            return

        chat_display.controls.append(
            ft.Row([ 
                ft.Container(
                    ft.Text(f"🧑‍💻 Usuario: {text}"),
                    padding=10,
                    bgcolor=ft.Colors.BLUE,
                    border_radius=10,
                    alignment=ft.Alignment(1, 0)
                )
            ], alignment=ft.MainAxisAlignment.END)
        )
        user_input.value = ""
        user_input.focus()
        user_input.update()
        page.update()

        espera_label.visible = True
        spinner.visible = True
        espera_label.update()
        spinner.update()
        
        respuesta = chat(text)  # Reemplaza con tu lógica de respuesta

        espera_label.visible = False
        spinner.visible = False
        espera_label.update()
        spinner.update()
        chat_display.controls.append(
            ft.Row([ 
                ft.Container(
                    ft.Text(f"🤖 IA: {respuesta}"),
                    padding=10,
                    bgcolor=ft.Colors.GREEN,
                    border_radius=10,
                    alignment=ft.Alignment(1, 0)
                )
            ], alignment=ft.MainAxisAlignment.START)
        )
        page.update()

    def clear_chat(e):
        chat_display.controls.clear()
        page.update()

    send_btn.on_click = send_message
    clear_btn.on_click = clear_chat
    user_input.on_submit = send_message

    page.add(
        ft.Column([ 
            ft.Text("🤖 Bienvenido al Chat IA", size=22, weight="bold", text_align="center"), # type: ignore
            espera_label,
            spinner,
            ft.Container(
                chat_display,
                height=500,
                bgcolor=ft.Colors.GREY,
                border_radius=10,
                padding=ft.Padding(left=50, right=50, top=10, bottom=10),
                width=1365,
                margin=ft.Margin(top=30, bottom=20, left=50, right=50)
            ),
            ft.Row([
                user_input,
                ft.Column([send_btn, clear_btn])
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )
    
if __name__ == "__main__":
    ft.app(target=main)
