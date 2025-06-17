# FraudAI: Sistema de Detección y Explicación de Fraude con LLMs

## 🚀 Visión General del Proyecto

`FraudAI` es un sistema prototipo de detección y explicación de fraude en transacciones financieras, diseñado como un Producto Mínimo Viable (MVP). Combina técnicas de procesamiento de lenguaje natural (NLP) con Large Language Models (LLMs) para identificar transacciones sospechosas y proporcionar explicaciones claras y contextualizadas.

A diferencia de los sistemas tradicionales basados únicamente en reglas o modelos de Machine Learning convencionales, `FraudAI` integra un LLM (Llama 3.2 de Ollama) directamente en el proceso de detección de anomalías, lo que permite un "razonamiento" más flexible y una explicación más humana del porqué una transacción podría ser fraudulenta.

## ✨ Características Principales

* **Detección de Anomalías impulsada por LLM:** El módulo `AnomalyDetector` utiliza **Llama 3.2** para analizar los detalles de cada transacción y determinar si presenta patrones fraudulentos.
* **Recuperación Aumentada de Generación (RAG):** El `RAGRetriever` accede a una base de conocimiento de reglas de fraude (`fraud_rules.json`) para proporcionar contexto relevante.
* **Explicaciones de Fraude Generadas por LLM:** El `LLMExplainer` utiliza **Llama 3.2** para generar explicaciones claras y concisas del motivo del fraude, basándose en el contexto recuperado por RAG.
* **Gestión de Embeddings:** El `EmbeddingManager` (simulado en el MVP) prepara los datos de transacción para su procesamiento por los LLMs.
* **Interfaz de Usuario Interactiva con Streamlit:** Un dashboard intuitivo que permite visualizar el procesamiento de transacciones, identificar fraudes detectados y explorar las explicaciones generadas por la IA.
* **Escalabilidad para MVP:** Diseñado para manejar un volumen moderado de transacciones para demostraciones, con una estructura modular que facilita futuras expansiones.

## ⚙️ Arquitectura del Sistema (MVP)

El sistema sigue un flujo de procesamiento modular:

1.  **Datos de Transacción:** Las transacciones se cargan desde un conjunto de datos (ej. JSON Lines).
2.  **`Transaction` Model:** Los datos brutos de la transacción se transforman en un objeto estructurado.
3.  **`EmbeddingManager`:** (Simulado) Convierte los detalles de la transacción en embeddings (representaciones vectoriales), aunque para la detección LLM, el texto es la entrada principal.
4.  **`AnomalyDetector`:**
    * Recibe el objeto `Transaction`.
    * Envía los detalles de la transacción a **Ollama Llama 3.2** con un prompt específico para clasificarla como `FRAUDULENTO` o `NORMAL`. (Temperatura: **0.1**).
    * Incluye un mecanismo de fallback basado en reglas si la llamada al LLM falla.
5.  **`RAGRetriever`:**
    * Si el `AnomalyDetector` marca una transacción como fraudulenta.
    * Busca en un archivo `fraud_rules.json` (que contiene reglas de fraude estructuradas con descripciones y palabras clave) para encontrar el contexto más relevante para la transacción actual.
6.  **`LLMExplainer`:**
    * Si se detecta fraude.
    * Utiliza **Ollama Llama 3.2** (Temperatura: **0.5**) junto con los detalles de la transacción y el contexto recuperado por RAG para generar una explicación humana sobre por qué se considera fraudulenta la transacción.
7.  **Salida de Resultados:** Los resultados del procesamiento (incluyendo el veredicto de fraude, el contexto RAG y la explicación del LLM) se guardan en un archivo `.jsonl`.
8.  **Streamlit Dashboard:** Carga y visualiza los resultados del `.jsonl`, proporcionando métricas clave, una tabla de transacciones fraudulentas y un desglose detallado por transacción.

## 🛠️ Requisitos del Sistema

* Python 3.9+
* **Ollama:** Necesitas tener Ollama instalado y ejecutándose en tu máquina.
* **Modelo Llama 3.2:** El modelo `llama3.2` debe estar disponible en Ollama (`ollama pull llama3.2`).
* **GPU con al menos 6GB de VRAM:** Recomendado para un rendimiento óptimo con el modelo Llama 3.2 (8B parámetros).
* **Dependencias de Python:** Ver `requirements.txt` (se generará si aún no existe).

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

## 🚀 Configuración y Ejecución

### Crear Entorno Virtual e Instalar Dependencias

Es altamente recomendable usar un entorno virtual para gestionar las dependencias del proyecto.

```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
# Si no tienes un requirements.txt, instala manualmente:
pip install pandas numpy streamlit plotly ollama
```

### 3. Instalar y Configurar Ollama

Descarga Ollama desde [ollama.com](https://ollama.com/).
Una vez instalado, abre tu terminal y descarga el modelo Llama 3.2:

```bash
ollama pull llama3.2
```

Asegúrate de que el servidor Ollama esté ejecutándose (usualmente se inicia automáticamente después de la instalación). Puedes verificarlo con `ollama run llama3.2` o `ollama serve`.

### 4. Preparar la Estructura de Datos

Asegúrate de que tu proyecto tenga la siguiente estructura de directorios:

```
.
├── data/
│   ├── bank_transactions_data_2.csv    # Dataset para entrenar y evaluar el modelo.
│   └── fraud_rules.json                # Reglas de fraude para el módulo RAG.
├── src/
│   ├── __init__.py
│   ├── app_fraude.py                   # Dashboard interactivo de Streamlit para la visualización.
│   ├── main_app.py                     # Punto de entrada principal para la aplicación.
│   ├── process_function.py             # Script para el procesamiento de transacciones.
│   ├── transaction.py                  # Clase que define la estructura de una transacción bancaria.
│   └── fraud_detection_service/
│       ├── __init__.py
│       ├── anomaly_detector.py         # Módulo central para la detección de anomalías/fraude utilizando LLMs (via Ollama).
│       ├── embedding_manager.py        # Gestiona la creación y el almacenamiento de embeddings de transacciones.
│       ├── llm_explainer.py            # Genera explicaciones detalladas para las detecciones de fraude.
│       └── rag_retriever.py            # Módulo para recuperar contexto relevante (ej. reglas de fraude) para el LLM.
├── app_main.ipynb                      # Notebook principal para ejecutar la aplicación (desarrollo/pruebas).
├── fine-tuning.ipynb                   # Notebook para el proceso de fine-tuning del LLM.
├── README.md                           # Este archivo.
└── requirements.txt                    # Lista de dependencias del proyecto.
```

Coloca tu archivo `fraud_rules.json` (generado previamente) en la carpeta `data/`.

### 5. Ejecutar el Procesamiento de Transacciones

Este script procesará tus datos de transacciones y generará el archivo `processed_transactions_results.jsonl` que será consumido por el dashboard de Streamlit.

Este proceso puede tomar tiempo dependiendo del número de transacciones y la velocidad de tu GPU/CPU, ya que cada transacción implica una llamada al LLM.

### 6. Iniciar el Dashboard de Streamlit

Una vez que el archivo `.jsonl` se haya generado, puedes iniciar la aplicación Streamlit:

```bash
streamlit run src/app_fraude.py
```

Se abrirá una nueva pestaña en tu navegador web con el dashboard de `FraudAI`.

## 📂 Estructura del Proyecto

* `src/`: Contiene el código fuente principal del proyecto.
    * `app_fraude.py`: La aplicación web interactiva desarrollada con Streamlit.
    * `process_data_for_streamlit.py`: Script para cargar, procesar y guardar los resultados de las transacciones.
    * `fraud_detection_service/`: Módulos principales del sistema de detección.
        * `anomaly_detector.py`: Lógica de detección de fraude (usa Ollama Llama 3.2).
        * `embedding_manager.py`: Gestión de embeddings (simulada para MVP).
        * `llm_explainer.py`: Genera explicaciones de fraude (usa Ollama Llama 3.2).
        * `rag_retriever.py`: Recupera contexto de reglas de fraude.
    * `test_fraud_processing.py`: Contiene la definición de la clase `Transaction` y pruebas de la lógica de procesamiento.
* `data/`: Contiene los archivos de datos.
    * `fraud_rules.json`: Base de conocimiento de reglas de fraude.
    * `processed_transactions_results.jsonl`: Archivo de salida con los resultados procesados.
* `requirements.txt`: Lista de dependencias de Python.
* `README.md`: Este archivo.