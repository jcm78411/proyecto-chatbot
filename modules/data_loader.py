import os
import json
import nltk
import pickle
import itertools
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt_tab')
nltk.download("punkt")
stemmer = LancasterStemmer()

dir_path = os.path.dirname(os.path.realpath(__file__))
extended_data_path = os.path.join(dir_path, "..", "storage", "data", "singleton_dataset_extended.json")

with open(extended_data_path, "r", encoding="utf-8") as f:
    singleton_data = json.load(f)

with open(os.path.join(dir_path, "..", "storage", "data", "singleton_dataset_extended.json"), "r", encoding="utf-8") as f:
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
