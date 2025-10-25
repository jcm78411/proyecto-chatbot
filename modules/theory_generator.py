import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .data_loader import intents_list

dir_path = os.path.dirname(os.path.realpath(__file__))

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
    content = implementation_theory.get(tag, f"No se encontró teoría definida para el tipo {tag}.")
    theory_dataset.append({"type": tag, "content": content})

with open(os.path.join(dir_path, "..", "data_bot", "data_bot-main", "theory_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(theory_dataset, f, indent=2, ensure_ascii=False)

with open(os.path.join(dir_path, "..", "data_bot", "data_bot-main", "theory_dataset.json"), "r", encoding="utf-8") as f:
    theory_data = json.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc["content"] for doc in theory_data]
theory_embeddings = embedding_model.encode(texts, convert_to_numpy=True)
embedding_dim = theory_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(theory_embeddings)
