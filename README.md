# 🤖 Singleton Pattern Chatbot

Este proyecto implementa un **asistente inteligente** que responde preguntas sobre el patrón de diseño Singleton en diversos estilos de implementación, usando aprendizaje automático, embeddings semánticos y un buscador semántico con FAISS.

---

## 🧠 ¿Qué hace este proyecto?

- 📚 **Explica la teoría** de cada tipo de implementación Singleton.
- 💻 **Muestra ejemplos reales** y generados en Java para distintos estilos.
- 🔍 **Detecta automáticamente la intención** del usuario gracias a una red neuronal.
- 🧠 **Recupera teoría relacionada** usando **embeddings** con `sentence-transformers` y **búsqueda FAISS**.
- 🎓 Entrena un modelo de clasificación basado en Keras con frases comunes sobre Singleton.

---

## 📂 Estructura del proyecto

- data_bot/
- ├── data_bot-main/
- │ ├── singleton_dataset_enriched.json # Dataset enriquecido original
- │ ├── singleton_dataset_cleaned.json # Dataset con estructura normalizada y comentarios eliminados
- │ ├── singleton_dataset_extended.json # Dataset con ejemplos adicionales generados
- │ ├── theory_dataset.json # Teoría explicativa por tipo de singleton
- EntrenamientoPickle/
- ├── brain_model.h5 # Modelo de clasificación entrenado
- ├── brain.words.pickle # Palabras, etiquetas y vectores de entrenamiento
- main.py # Código principal del bot

---

## 🛠️ Tecnologías usadas

- **Python 3.10+**
- **NLTK** para tokenización y stemming.
- **Keras / TensorFlow** para entrenamiento del modelo.
- **SentenceTransformers** (`all-MiniLM-L6-v2`) para embeddings semánticos.
- **FAISS** para recuperación de teoría relacionada.
- **Flet** para futuras integraciones gráficas o interfaces de usuario.

---

## 🚀 Cómo usarlo

1. Clona el repositorio.

```bash
git clone https://github.com/jcm78411/proyecto-chatbot.git
```

```bash
cd proyecto-chatbot
```

```bash
code . #Si tienes Visual Studio Code
```

2. Crea el entorno virtual

```bash
python3.12.exe -m venv .venv
```

3. Instala los paquetes

```bash
pip install -r requirements.txt
```

4. Ejecuta el proyecto
* Nota: El bot se entrenará si no encuentra un modelo previamente guardado.
* Nota: Si haces cambios en las bases de datos, asegurate de eliminar las carpetas de Entrenamiento
* Nota: La primera vez que se ejecute, asegurate de estar conectado a internet
* Porque?: NLTK y Sentence pude que necesiten descargar contenido

```bash
python main.py
```
O también

```bash
flet run
```


💬 Ejemplo de uso

Usuario: ¿Cómo se implementa un singleton tipo lazy?
Bot:
📚 Teoría relacionada (lazy):
El Singleton lazy retrasa la creación de la instancia hasta que realmente se necesita...

💻 Ejemplo de código:
public class LazySingleton {
    private static LazySingleton instance;
    ...
}



📌 Tipos de Singleton soportados
- classic
- lazy
- thread_safe
- double_checked
- enum
- bill_pugh
- synchronized
- reflection_safe
- eager
- static_block
- volatile
- registry
- inner_static_class
- metaclass (Python)
- module (Python)
- borg (Python)

