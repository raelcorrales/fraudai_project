# MVP: Sistema Inteligente de Detección de Fraudes con RAG y LLM

## 1. 💡 Problema

* Los sistemas tradicionales de detección de fraudes se basan en reglas estáticas o modelos de Machine Learning opacos.
* Cuando se detecta una anomalía, **no explican el porqué**, dificultando la toma de decisiones humanas y generando desconfianza.
* Escenarios de fraude complejos requieren un análisis más profundo y explicaciones comprensibles, no solo un "sí" o un "no".

---

## 2. 🎯 Objetivo del MVP

> Crear un sistema capaz de:

* Detectar transacciones potencialmente fraudulentas.
* Recuperar **reglas de fraude relevantes** usando embeddings y similitud semántica (**RAG semántico**).
* Generar **explicaciones detalladas y contextualizadas en lenguaje natural** con un Modelo de Lenguaje Grande (LLM).

---

## 3. 🧠 Arquitectura Funcional

```
Transacción nueva
      ↓
→ Embedding de la transacción (con Ollama)
      ↓
→ Detección de anomalía (razonamiento del LLM)
      ↓
  ¿Es fraude?
      ↓
[SI] → RAG → Recuperar reglas similares (similitud de embeddings)
      ↓
→ LLM → Explicación contextual y mitigación sugerida
```

---

## 4. 🛠️ Componentes Clave

| Componente          | Descripción                                                                                               |
| :------------------ | :-------------------------------------------------------------------------------------------------------- |
| `Transaction`       | Clase que modela la transacción como un objeto de datos estructurado.                                    |
| `EmbeddingManager`  | Obtiene vectores de texto (embeddings) para transacciones y reglas usando **Ollama**.                     |
| `AnomalyDetector`   | Realiza una predicción inicial de fraude utilizando el **razonamiento de un LLM**, con un *fallback* heurístico. |
| `RAGRetriever`      | Recupera reglas de fraude relevantes usando la **similitud semántica entre embeddings** (cosine similarity). |
| `LLMExplainer`      | Genera explicaciones comprensibles en lenguaje natural, usando un **LLM** y el contexto recuperado.     |
| `OllamaBase`        | Clase base para gestionar la conexión, reintentos y comunicación con la API de Ollama.                  |

---

## 5. 📊 Ejemplo de Salida

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

---

## 6. ⚙️ Detalles Técnicos

* **Lenguaje**: Python 3.9+
* **Embeddings**: Generados con **Ollama** (`nomic-embed-text` recomendado)
* **RAG**: Recuperación de reglas por **similitud de embeddings** utilizando `sklearn.metrics.pairwise.cosine_similarity`.
* **LLM**: Modelos de lenguaje configurables vía variables de entorno (`llm_model_name`, `chat_model_name`, `llm_model_temperature`). **Integración con Ollama** (`llama3.1` recomendado para razonamiento y explicaciones).
* **Modularidad**: Diseño de componentes desacoplados para fácil extensión y mantenimiento.

---

## 7. 🧪 Flujo de Prueba (Ejemplo de Código)

```python
from src.process_function import process_transaction
from src.fraud_detection_service.embedding_manager import EmbeddingManager
from src.fraud_detection_service.anomaly_detector import AnomalyDetector
from src.fraud_detection_service.rag_retriever import RAGRetriever
from src.fraud_detection_service.llm_explainer import LLMExplainer
from src.fraud_detection_service.embedding_rule_utils import load_and_embed_rules
import os

# --- Configuración y Carga GLOBAL (¡ejecutar una sola vez al inicio de la aplicación!) ---
RULES_FILE_PATH = "resources/fraud_rules.json" # Ajusta la ruta real de tus reglas
EMBEDDING_MODEL_NAME = os.environ.get('embedding_model_name', 'nomic-embed-text')
CHAT_MODEL_NAME = os.environ.get('chat_model_name', 'llama3.1')
LLM_MODEL_NAME = os.environ.get('llm_model_name', 'llama3.1')
LLM_MODEL_TEMPERATURE = float(os.environ.get('llm_model_temperature', '0.5'))

# Inicialización de todos los componentes clave
global_embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
global_anomaly_detector = AnomalyDetector(model_name=CHAT_MODEL_NAME)
global_llm_explainer = LLMExplainer(model_name=LLM_MODEL_NAME, temperature=LLM_MODEL_TEMPERATURE)

# Carga y embebido de reglas de fraude (también una única vez)
pre_embedded_rules = load_and_embed_rules(model_name=EMBEDDING_MODEL_NAME, rules_file_path=RULES_FILE_PATH)
global_rag_retriever = RAGRetriever(fraud_rules_data=pre_embedded_rules)

# --- Ejemplo de procesamiento de una transacción (se ejecutaría por cada transacción entrante) ---
transaccion_dict = {
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
    transaction_data=transaccion_dict,
    embedding_manager=global_embedding_manager,
    anomaly_detector=global_anomaly_detector,
    rag_retriever=global_rag_retriever,
    llm_explainer=global_llm_explainer
)
print(response)
```

---

## 8. 🎯 Valor Agregado

* **Detección Automatizada**: Identifica patrones de fraude de manera eficiente.
* **Explicaciones Comprensibles**: Proporciona a los analistas razones claras para las detecciones, mejorando la confianza y la acción.
* **Modular y Portable**: Fácil de integrar en arquitecturas existentes y escalar.
* **Explicabilidad**: Transforma la caja negra del ML en un proceso transparente y auditable.
* **Base Robusta**: Punto de partida ideal para futuras integraciones con APIs, dashboards y sistemas de monitoreo.

---

## 10. 🙌 Conclusión

> Este MVP demuestra cómo combinar el poder de **RAG y LLM con la flexibilidad de Ollama** para crear un sistema de detección de fraude:

* **Inteligente y Contextual**: Va más allá de las reglas fijas, entendiendo el significado de las transacciones.
* **Explicativo y Práctico**: Empodera a los analistas con información procesable.
* Útil para auditores, analistas de fraude, instituciones financieras, fintechs y cualquier entidad que requiera una detección de fraude avanzada y transparente.
