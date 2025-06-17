import os
import logging
from typing import List, Dict, Any

from .transaction import Transaction

from .fraud_detection_service.embedding_manager import EmbeddingManager
from .fraud_detection_service.anomaly_detector import AnomalyDetector
from .fraud_detection_service.rag_retriever import RAGRetriever
from .fraud_detection_service.llm_explainer import LLMExplainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def process_transaction(transaction_data: Dict[str, Any], embedded_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Procesa una transacción y realiza la detección de fraude utilizando un diccionario de datos.
    """
    try:
        transaction = Transaction(transaction_data)
    except Exception as e:
        logger.error(f"Error al parsear los datos de la transacción: {e}")
        return {'error': f"Datos de transacción inválidos: {e}"}
    
    llm_model_name = os.environ.get('llm_model_name', 'llama3.1')               # Default to llama3.1 if not set
    llm_model_temperature = os.environ.get('llm_model_temperature', '0.5')      # Default temperature if not set
    chat_model_name = os.environ.get('chat_model_name', 'llama3.1')             # Default to llama3.1 if not set
    embedding_model_name = os.environ.get('embedding_model_name', 'all-minilm') # Default to all-minilm if not set

    embedding_manager = EmbeddingManager(model_name=embedding_model_name)
    anomaly_detector = AnomalyDetector(model_name=chat_model_name)
    rag_retriever = RAGRetriever(fraud_rules_data=embedded_rules)
    llm_explainer = LLMExplainer(model_name=llm_model_name, temperature=float(llm_model_temperature))

    logger.info("Iniciando procesamiento de transacción...")

    transaction_details_str = transaction.to_string_for_embedding()
    
    # PASO 1: Generar Embedding
    logger.info("Generando embedding de la transacción...")
    transaction_embedding = embedding_manager.get_embedding(transaction_details_str)

    # PASO 2: Detección de Anomalías
    logger.info("Detectando anomalías...")
    is_fraudulent = anomaly_detector.predict_fraud(transaction_embedding, transaction)

    return_data = transaction_data.copy()
    return_data.update({
        'is_fraudulent': is_fraudulent,
        'transaction_details_str_for_llm': transaction_details_str,
        'transaction_embedding_shape': transaction_embedding.shape, 
    })
    
     # Inicializar los campos que se llenarán condicionalmente
    fraud_context = None
    explanation = "No se detectó fraude en esta transacción."
    detected_risk_level = "NONE" # Valor por defecto si no es fraudulenta
    detected_rule_tags = [] # Lista vacía por defecto

    if is_fraudulent:
        logger.error(f"🚨 ¡ATENCIÓN! Transacción Potencialmente Fraudulenta: {transaction.TransactionID} 🚨")
        logger.warning(f"Monto: ${transaction.TransactionAmount}, Comercio: {transaction.MerchantID}, Ubicación: {transaction.Location}")

        # PASO 3: Recuperar Contexto RAG, Nivel de Riesgo y Tags (solo si es fraudulento)
        logger.info("Recuperando contexto de fraude y detalles de reglas...")
        # El RAGRetriever ahora devuelve 3 valores (asegúrate de que tu RAGRetriever devuelve LOW, MEDIUM, HIGH, CRITICAL si se activa una regla)
        fraud_context, detected_risk_level, detected_rule_tags = rag_retriever.retrieve_context(transaction, transaction_embedding)
        
        # PASO 4: Generar Explicación con LLM (solo si es fraudulento)
        logger.info("Generando explicación con LLM...")
        # Pasamos el nivel de riesgo y los tags al explicador, para que los pueda usar en su prompt
        explanation = llm_explainer.generate_explanation(
            transaction_details_str, 
            fraud_context, 
            detected_risk_level=detected_risk_level, 
            detected_rule_tags=detected_rule_tags
        )
        logger.info(f"Explicación Generada por la IA: {explanation}")
        
    else:
        logger.info(f"✅ Transacción Normal: {transaction.TransactionID}. No se detectaron anomalías.")
        logger.info("La transacción no presenta patrones sospechosos según nuestro análisis.")
        # Los valores por defecto ("NONE" y [] vacía) ya están establecidos para este caso.
    
    # Actualizar los datos de retorno con toda la información
    return_data.update({
        'fraud_context': fraud_context,
        'explanation': explanation,
        'detected_risk_level': detected_risk_level,
        'detected_rule_tags': detected_rule_tags,
    })
    
    logger.info(f"Procesamiento de transacción {transaction.TransactionID} completado.")
    
    return return_data
