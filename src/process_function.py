import os
import logging
from typing import List, Dict, Any

from src.transaction import Transaction

from src.fraud_detection_service.embedding_manager import EmbeddingManager
from src.fraud_detection_service.anomaly_detector import AnomalyDetector
from src.fraud_detection_service.rag_retriever import RAGRetriever
from src.fraud_detection_service.llm_explainer import LLMExplainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def process_transaction(transaction_data: Dict[str, Any], fraud_rules: List) -> Dict[str, Any]:
    """
    Procesa una transacción y realiza la detección de fraude utilizando un diccionario de datos.
    """
    try:
        transaction = Transaction(transaction_data)
    except Exception as e:
        logger.error(f"Error al parsear los datos de la transacción: {e}")
        return {'error': f"Datos de transacción inválidos: {e}"}
    
    model_name = os.environ.get('model_name', 'llama3.1')  # Default to llama3. if not set
    model_temperature = os.environ.get('temperature', '0.5')  # Default temperature if not set

    embedding_manager = EmbeddingManager()
    anomaly_detector = AnomalyDetector()
    rag_retriever = RAGRetriever(fraud_rules)
    llm_explainer = LLMExplainer(model_name=model_name, temperature=float(model_temperature))

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
        #'transaction_embedding': transaction_embedding.tolist()
    })
    result_data = {}
    if is_fraudulent:
        logger.error(f"🚨 ¡ATENCIÓN! Transacción Potencialmente Fraudulenta: {transaction.TransactionID} 🚨")
        logger.warning(f"Monto: ${transaction.TransactionAmount}, Comercio: {transaction.MerchantID}, Ubicación: {transaction.Location}")

        # PASO 3: Recuperar Contexto RAG (solo si es fraudulento)
        logger.info("Recuperando contexto de fraude para RAG...")
        fraud_context = rag_retriever.retrieve_context(transaction)
        detected_risk_level = fraud_context.get('risk_level', 'NONE')
        detected_rule_tags = fraud_context.get('tags', [])
        rule_descriptions = fraud_context.get('rule_descriptions', [])

        
        # PASO 4: Generar Explicación con LLM (solo si es fraudulento)
        logger.info("Generando explicación con LLM...")
        explanation = llm_explainer.generate_explanation(transaction_details_str, fraud_context)
        logger.info(f"Explicación Generada por la IA: {explanation}")
        result_data = {
            'fraud_context': rule_descriptions,
            'detected_risk_level': detected_risk_level,
            'detected_rule_tags': detected_rule_tags,
            'explanation': explanation,
        }
    else:
        logger.info(f"✅ Transacción Normal: {transaction.TransactionID}. No se detectaron anomalías.")
        logger.info("La transacción no presenta patrones sospechosos según nuestro análisis.")
        result_data = {
            'fraud_context': None,
            'detected_risk_level': "NONE",
            'detected_rule_tags': [],
            'explanation': "No se detectó fraude en esta transacción.",
        }
    return_data.update(result_data)
    logger.info(f"Procesamiento de transacción {transaction.TransactionID} completado.")
    
    return return_data