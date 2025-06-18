# Sistema Inteligente de Detección de Fraudes con RAG

Este proyecto implementa un sistema inteligente de **detección de fraudes financieros** combinando lo mejor del aprendizaje automático, recuperación de conocimiento (RAG) y generación de lenguaje natural (LLM).

> Desarrollado por **Rael Corrales** · Python Engineer & Data Specialist

-----

## 🚀 Características Principales

- **Embeddings Semánticos**: Representación vectorial avanzada de transacciones y reglas de fraude utilizando modelos de Ollama.
- **Detección de Anomalías**: Identificación de comportamientos transaccionales sospechosos a través de modelos de lenguaje grande (LLM).
- **Recuperación Semántica de Reglas (RAG)**: Asociación contextual entre transacción y reglas relevantes a partir de embeddings.
- **Explicaciones Generadas por IA**: Justificaciones claras y adaptadas para analistas sobre las decisiones tomadas.
- **Pipeline Modular**: Componentes desacoplados y fácilmente extensibles.
- **Integración con Ollama**: Toda la inteligencia artificial (embeddings y LLM) corre de forma local, lo que maximiza la privacidad y la flexibilidad.

-----

## Dataset (Bank Transaction Dataset for Fraud Detection)

>Detailed Analysis of Transactional Behavior and Anomaly Detection

**URL**: [https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection?resource=download](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection?resource=download)

### Acerca del conjunto de datos
Este conjunto de datos ofrece una visión detallada del comportamiento transaccional y los patrones de actividad financiera, ideal para explorar la detección de fraudes y la identificación de anomalías. Contiene **2512** muestras de datos de transacciones, que abarcan diversos atributos de las transacciones, datos demográficos de los clientes y patrones de uso. Cada entrada ofrece información completa sobre el comportamiento de las transacciones, lo que permite el análisis para aplicaciones de seguridad financiera y detección de fraudes.

### Características principales:
- **TransactionID**: Identificador alfanumérico único para cada transacción.
- **AccountID**: Identificador único para cada cuenta, con múltiples transacciones por cuenta.
- **TransactionAmount**: Valor monetario de cada transacción, desde pequeños gastos cotidianos hasta compras más importantes.
- **TransactionDate**: Marca de tiempo de cada transacción, que captura la fecha y la hora.
- **TransactionType**: Campo categórico que indica transacciones de "Crédito" o "Débito".
- **Location**: Ubicación geográfica de la transacción, representada por nombres de ciudades de EE. UU.
- **DeviceID**: Identificador alfanumérico de los dispositivos utilizados para realizar la transacción.
- **IP Address**: Dirección IPv4 asociada a la transacción, con cambios ocasionales para algunas cuentas.
- **MerchantID**: Identificador único para comerciantes, que muestra los comerciantes preferidos y atípicos para cada cuenta.
- **AccountBalance**: Saldo de la cuenta después de la transacción, con correlaciones lógicas según el tipo y el importe de la transacción.
- **PreviousTransactionDate**: Marca de tiempo de la última transacción de la cuenta, que ayuda a calcular la frecuencia de las transacciones.
- **Channel**: Canal a través del cual se realizó la transacción (p. ej., en línea, cajero automático, sucursal).
- **CustomerAge**: Edad del titular de la cuenta, con agrupaciones lógicas según su ocupación.
- **CustomerOccupation**: Ocupación del titular de la cuenta (p. ej., médico, ingeniero, estudiante, jubilado), que refleja los patrones de ingresos.
- **TransactionDuration**: Duración de la transacción en segundos, que varía según el tipo de transacción.
- **LoginAttempts**: Número de intentos de inicio de sesión antes de la transacción; los valores más altos indican posibles anomalías.
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

- **Python 3.9+**
- **Ollama**: Motor de modelos LLM y generación de embeddings local.
- **FastAPI**: (opcional) Para servir el pipeline como API REST.
- **NumPy** **Pandas**: Manipulación de datos tabulares.

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

## 🏗️ Detalles de Implementación

### 1. **Embebido de Reglas de Fraude**
Las reglas de negocio del banco están estructuradas en un archivo JSON. Estas reglas se convierten en embeddings vectoriales usando el modelo configurado en Ollama. Esto permite hacer matching semántico, no solo por palabras clave.

### 2. **Procesamiento de una Transacción**
El pipeline sigue estos pasos:
- **Transformación y preprocesamiento** de la transacción.
- **Generación de embedding** de la transacción.
- **Detección de anomalía** vía LLM (razonamiento contextual).
- **Recuperación de reglas relevantes** (RAG), combinando heurística y similitud de embeddings.
- **Generación de explicación** con LLM para justificar el resultado.
- **Asignación de nivel de riesgo y tags** en base a reglas o inferencia del LLM.
- **Estructuración de la respuesta** para consumo humano o automatizado.

### 3. **Explicaciones de IA**
El sistema genera textos explicativos para cada caso de fraude detectado, fundamentados en el contexto, reglas y características de la transacción.

### 4. **Resiliencia y Robustez**
Las llamadas a Ollama están protegidas con lógica de reintentos exponenciales para maximizar la robustez en producción.

### 5. **API RESTful**
El sistema puede exponerse como un servicio web para procesar transacciones individuales vía HTTP POST, devolviendo todo el análisis contextual y explicativo en formato JSON.

### Integración en tu Aplicación

Puedes importar la función `process_transaction` y usarla en tu propia aplicación. **Es crucial inicializar los componentes de Ollama una sola vez** para optimizar el rendimiento y evitar inicializaciones redundantes.

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

## 🙌 Conclusión

Este MVP demuestra de manera efectiva cómo la combinación de **RAG y LLM, potenciada por Ollama**, puede revolucionar la detección de fraude, ofreciendo un sistema:

  * **Inteligente y Contextual**: Va más allá de las reglas rígidas, comprendiendo la semántica de las transacciones y las reglas.
  * **Explicativo y Práctico**: Proporciona a los analistas de fraude las herramientas y la información que necesitan para tomar decisiones rápidas e informadas.

Este proyecto es una base sólida para auditores, analistas de fraude, instituciones financieras, fintechs y cualquier entidad que busque una solución de detección de fraude avanzada y transparente.

-----