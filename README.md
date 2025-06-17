Okay, ¡aquí tienes el **README.md** completamente actualizado y listo para tu proyecto\! Incorpora todos los cambios que discutimos, destacando la integración con Ollama y las mejoras en la explicación del flujo de trabajo.

-----

# 🛡️ Fraud Detection RAG + LLM

Este proyecto implementa un sistema inteligente de **detección de fraudes financieros** combinando lo mejor del aprendizaje automático, recuperación de conocimiento (RAG) y generación de lenguaje natural (LLM).

> Desarrollado por **Rael Corrales** · Python Engineer & Data Specialist

-----

## 🚀 Características Principales

  * 🧠 **Embeddings Semánticos**: Representación vectorial avanzada para transacciones y reglas de fraude, utilizando **Ollama**.
  * ⚠️ **Detección de Anomalías con LLM**: Identifica transacciones sospechosas mediante el razonamiento de un Modelo de Lenguaje Grande.
  * 📚 **RAG Semántico**: Recupera reglas de fraude relevantes usando la similitud entre embeddings para un contexto preciso.
  * 💬 **Explicaciones Generadas por IA**: Ofrece razones claras y detalladas en lenguaje natural sobre por qué una transacción es considerada fraudulenta.
  * 🔌 **Diseño Modular y Extensible**: Componentes bien definidos para facilitar la adaptación y el crecimiento del sistema.

-----

## 📁 Estructura del Proyecto

```
src/
├── main_app.py                 # Punto de entrada principal (demostración/ejecución)
├── process_function.py         # Orquestador de la lógica central de detección
├── transaction.py              # Modelo de datos para las transacciones
└── fraud_detection_service/
    ├── embedding_manager.py    # Gestión de la generación de embeddings con Ollama
    ├── anomaly_detector.py     # Lógica de detección de fraude impulsada por LLM
    ├── rag_retriever.py        # Recuperación semántica de reglas de fraude
    ├── llm_explainer.py        # Generación de explicaciones en lenguaje natural
    ├── ollama_base.py          # Clase base para la interacción con la API de Ollama
    └── embedding_rule_utils.py # Carga y embebido de reglas desde JSON

resources/
└── fraud_rules.json            # Archivo JSON con las reglas estructuradas de fraude
```

-----

## 🧠 Tecnologías Utilizadas

  * **Python 3.9+**
  * **[NumPy](https://numpy.org/)**: Para operaciones numéricas eficientes con embeddings.
  * **[scikit-learn](https://scikit-learn.org/)**: Principalmente para el cálculo de similitud coseno.
  * **[Ollama](https://ollama.com/)**: Plataforma fundamental para la ejecución local de Modelos de Lenguaje Grandes (LLMs) y modelos de embeddings.

-----

## ⚙️ Configuración del Entorno

Asegúrate de tener **Ollama instalado y corriendo** en tu sistema. Luego, descarga los modelos necesarios.

1.  **Instala Ollama**: Sigue las instrucciones en [ollama.com](https://ollama.com/).

2.  **Descarga los modelos**:

      * Para embeddings (CRÍTICO: usa un modelo diseñado para embeddings):
        ```bash
        ollama pull nomic-embed-text
        ```
      * Para LLMs (detección y explicación):
        ```bash
        ollama pull llama3.1
        ```

3.  **Variables de Entorno**: Crea un archivo `.env` en la raíz de tu proyecto o define las siguientes variables de entorno:

    | Variable               | Descripción                                                                          | Valor por Defecto   |
    | :--------------------- | :----------------------------------------------------------------------------------- | :------------------ |
    | `llm_model_name`       | Nombre del modelo LLM de Ollama para la **generación de explicaciones**.             | `llama3.1`          |
    | `llm_model_temperature`| Temperatura de creatividad del LLM para explicaciones (0.0 a 1.0).                  | `0.5`               |
    | `chat_model_name`      | Nombre del modelo LLM de Ollama para la **detección inicial de anomalías**.         | `llama3.1`          |
    | `embedding_model_name` | Nombre del modelo de **embeddings** de Ollama (debe ser un modelo de embeddings).  | `nomic-embed-text`  |

-----

## ▶️ Ejecución del Proyecto

1.  **Clona el repositorio**:

    ```bash
    git clone https://github.com/raelcorrales/fraud-detection-rag-llm.git
    cd fraud-detection-rag-llm
    ```

2.  **Instala las dependencias de Python**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Verifica que Ollama esté corriendo**:

    ```bash
    ollama list
    ```

    (Deberías ver los modelos `nomic-embed-text` y `llama3.1` listados).

4.  **Ejecuta el script principal**:

    ```bash
    python src/main_app.py
    ```

### Integración en tu Aplicación

Puedes importar la función `process_transaction` y usarla en tu propia aplicación. **Es crucial inicializar los componentes de Ollama una sola vez** para optimizar el rendimiento y evitar inicializaciones redundantes.

```python
# Ejemplo de uso en tu script principal o controlador de API
import os
from src.process_function import process_transaction
from src.fraud_detection_service.embedding_manager import EmbeddingManager
from src.fraud_detection_service.anomaly_detector import AnomalyDetector
from src.fraud_detection_service.rag_retriever import RAGRetriever
from src.fraud_detection_service.llm_explainer import LLMExplainer
from src.fraud_detection_service.embedding_rule_utils import load_and_embed_rules

# --- Configuración y Carga GLOBAL (¡Ejecutar UNA ÚNICA VEZ al inicio de tu aplicación!) ---
RULES_FILE_PATH = "resources/fraud_rules.json" # Asegúrate de que esta ruta sea correcta
EMBEDDING_MODEL_NAME = os.environ.get('embedding_model_name', 'nomic-embed-text')
CHAT_MODEL_NAME = os.environ.get('chat_model_name', 'llama3.1')
LLM_MODEL_NAME = os.environ.get('llm_model_name', 'llama3.1')
LLM_MODEL_TEMPERATURE = float(os.environ.get('llm_model_temperature', '0.5'))

# Inicializa todos los componentes clave de tu sistema
global_embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
global_anomaly_detector = AnomalyDetector(model_name=CHAT_MODEL_NAME)
global_llm_explainer = LLMExplainer(model_name=LLM_MODEL_NAME, temperature=LLM_MODEL_TEMPERATURE)

# Pre-carga y embebe las reglas de fraude UNA ÚNICA VEZ al inicio
pre_embedded_rules = load_and_embed_rules(model_name=EMBEDDING_MODEL_NAME, rules_file_path=RULES_FILE_PATH)
global_rag_retriever = RAGRetriever(fraud_rules_data=pre_embedded_rules)

# --- Ejemplo de procesamiento de una transacción ---
# Este bloque se ejecutaría cada vez que recibas una nueva transacción
transaccion_ejemplo = {
    "TransactionID": "T-3981-XYZ",
    "TransactionAmount": 12000,
    "TransactionType": "compra internacional",
    "Location": "Estambul, Turquía",
    "DeviceID": "unknown_device",
    "IPAddress": "188.132.1.1",
    "CustomerAge": 35,
    "CustomerOccupation": "ingeniero",
    "AccountBalance": 10000,
    "LoginAttempts": 1,
    "TransactionDuration": 120,
    "Channel": "web",
    "Timestamp": "2024-06-15T10:30:00Z"
}

response = process_transaction(
    transaction_data=transaccion_ejemplo,
    embedding_manager=global_embedding_manager,
    anomaly_detector=global_anomaly_detector,
    rag_retriever=global_rag_retriever,
    llm_explainer=global_llm_explainer
)
print(response)
```

-----

## 📦 Ejemplo de Salida

Aquí tienes un ejemplo de cómo se vería la salida JSON para una transacción procesada:

```json
{
  "TransactionID": "T-3981-XYZ",
  "TransactionAmount": 12000,
  "is_fraudulent": true,
  "transaction_details_str_for_llm": "Transacción ID: T-3981-XYZ, Monto: 12000.0, Fecha: 2024-06-15, Tipo: compra internacional, Ubicación: Estambul, Turquía, Dispositivo: unknown_device,...",
  "transaction_embedding_shape": [768],
  "fraud_context": "Regla 1 (ID: RULE_001_HIGH_VALUE_UNUSUAL_CATEGORY): Transacciones de alto valor en categorías o ubicaciones inusuales. Keywords: monto alto, ubicación inusual, categorías sospechosas. Pasos de mitigación: Solicitar MFA; Contactar al cliente; Bloquear transacción si no se verifica.",
  "explanation": "La transacción fue detectada como fraudulenta (RIESGO CRÍTICO) por un monto inusualmente alto ($12000) en Estambul, Turquía, una ubicación inusual para el cliente, y realizada desde un dispositivo desconocido. Esto activó la regla de 'Transacciones de alto valor en categorías o ubicaciones inusuales'. Se recomienda aplicar Multi-Factor Authentication (MFA) y contactar al cliente de inmediato para verificar la transacción.",
  "detected_risk_level": "CRITICAL",
  "detected_rule_tags": ["Alto Valor", "Ubicación Inusual", "Dispositivo Sospechoso"]
}
```

-----

## 🧪 Tests

(Próximamente)

```bash
pytest tests/
```

-----

## 🧭 Roadmap

  * [x] Cargar y convertir reglas de fraude en embeddings.
  * [x] Procesar transacción con detección y explicación.
  * [x] Implementar matching semántico con RAG.
  * [x] Integrar con Ollama para Embeddings y LLMs locales.
  * [ ] Añadir soporte para múltiples proveedores de modelos de embeddings (más allá de Ollama).
  * [ ] Desarrollar visualizaciones de explicaciones (ej., con Dash o Streamlit).
  * [ ] Implementar integración con APIs RESTful (Flask / FastAPI) para servir el modelo.

-----

## 🙌 Conclusión

Este MVP demuestra de manera efectiva cómo la combinación de **RAG y LLM, potenciada por Ollama**, puede revolucionar la detección de fraude, ofreciendo un sistema:

  * **Inteligente y Contextual**: Va más allá de las reglas rígidas, comprendiendo la semántica de las transacciones y las reglas.
  * **Explicativo y Práctico**: Proporciona a los analistas de fraude las herramientas y la información que necesitan para tomar decisiones rápidas e informadas.

Este proyecto es una base sólida para auditores, analistas de fraude, instituciones financieras, fintechs y cualquier entidad que busque una solución de detección de fraude avanzada y transparente.

-----