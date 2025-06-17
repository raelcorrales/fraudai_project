# src/fraud_detection_service/rag_retriever.py (modificado)

import numpy as np
from typing import List, Dict, Any, Tuple

# Asumo que esta es la forma en que tus reglas están cargadas y pre-embedidas
# Esto es esencial, ya que el RAGRetriever necesita las reglas completas para obtener risk_level y tags.
# fraud_rules_file debería ser una lista de diccionarios de reglas, cada una con 'id', 'description', 'keywords', 'risk_level', 'mitigation_steps', 'tags' y quizás 'embedding'
# from .embedding_rule_utils import load_and_embed_rules # Puedes usar esto en tu main para pasar las reglas ya cargadas y embebidas

class RAGRetriever:
    def __init__(self, fraud_rules_data: List[Dict[str, Any]]):
        # fraud_rules_data debe ser la lista de diccionarios de reglas,
        # cada una con sus campos, incluyendo 'risk_level' y 'tags'.
        self.fraud_rules = fraud_rules_data
        # Si usas embeddings para la recuperación, necesitas que las reglas también estén embebidas aquí
        # Por ejemplo, self.rule_embeddings = np.array([rule['embedding'] for rule in fraud_rules_data])

    def retrieve_context(self, transaction: Any, transaction_embedding: np.ndarray) -> Tuple[str, str, List[str]]:
        """
        Recupera el contexto relevante de las reglas de fraude para una transacción.
        También determina el nivel de riesgo detectado y los tags de las reglas activadas.

        Args:
            transaction: Objeto de transacción (asumo que tiene atributos como MerchantID, Location, etc.)
            transaction_embedding: El embedding de la transacción.

        Returns:
            Una tupla que contiene:
            - fraud_context (str): El contexto RAG para el LLM.
            - detected_risk_level (str): El nivel de riesgo más alto de las reglas activadas.
            - detected_rule_tags (List[str]): Una lista única de tags de las reglas activadas.
        """
        relevant_rules = []
        
        # --- Lógica de Recuperación de Reglas ---
        # Esta es la parte más importante donde decides qué reglas se "activan".
        # Puedes usar:
        # 1. Similitud de Embedding (más avanzado, recomendado)
        # 2. Coincidencia de palabras clave (más simple)
        # 3. Lógica condicional (si la transacción cumple ciertos criterios de regla)

        # EJEMPLO SIMPLIFICADO: Coincidencia por palabra clave y/o heurísticas básicas
        # En una implementación real con embeddings, harías una búsqueda de similitud.
        # Aquí, vamos a simular la "activación" de reglas para demostrar la extracción de riesgo y tags.

        activated_risk_levels = []
        activated_tags = []
        context_parts = []

        # Para este ejemplo, voy a simular la activación basada en algunas propiedades de la transacción
        # y luego buscar reglas que coincidan conceptualmente.
        # En un sistema real, usarías la similitud de embedding con 'transaction_embedding'
        # para encontrar las 'top-k' reglas más similares de 'self.fraud_rules'.

        # Lógica de ejemplo (Reemplazar con lógica de similitud de embeddings real)
        # Si tienes la capacidad de calcular la similitud:
        # from sklearn.metrics.pairwise import cosine_similarity
        # transaction_embedding_reshaped = transaction_embedding.reshape(1, -1)
        # similarities = cosine_similarity(transaction_embedding_reshaped, self.rule_embeddings)[0]
        # top_rule_indices = similarities.argsort()[-k:][::-1] # Obtener los índices de las k reglas más similares
        # relevant_rules = [self.fraud_rules[i] for i in top_rule_indices]

        # Para un ejemplo funcional sin embeddings complejos aquí, simulemos:
        for rule in self.fraud_rules:
            # Lógica simple para simular si una regla "se activa"
            # En tu implementación, esto se basaría en la similitud semántica o reglas explícitas.
            is_rule_activated = False
            
            # Ejemplo de activación heurística (reemplaza esto con tu lógica de detección real)
            if "monto alto" in rule["keywords"] and transaction.TransactionAmount > 5000:
                is_rule_activated = True
            if "ubicación inusual" in rule["keywords"] and transaction.Location not in ["New York", "Los Angeles"]: # Simplificado
                 is_rule_activated = True
            if "intentos fallidos" in rule["keywords"] and transaction.LoginAttempts > 3:
                is_rule_activated = True
            # ... añade más heurísticas o (preferiblemente) usa la similitud de embeddings real
            
            # Si la regla se activa (por heurística o similitud de embedding)
            if is_rule_activated:
                relevant_rules.append(rule)

        # Si no se recuperan reglas, proporcionar un contexto por defecto
        if not relevant_rules:
            detected_risk_level = "LOW" # O un default apropiado
            detected_rule_tags = []
            context_parts.append("No se encontraron reglas de fraude directamente aplicables con alta confianza.")
            fraud_context = ". ".join(context_parts)
            return fraud_context, detected_risk_level, detected_rule_tags

        # Procesar las reglas relevantes para construir el contexto y extraer riesgo/tags
        for i, rule in enumerate(relevant_rules):
            context_parts.append(f"Regla {i+1} (ID: {rule['id']}): {rule['description']}. Keywords: {', '.join(rule['keywords'])}. Pasos de mitigación: {'; '.join(rule['mitigation_steps'])}.")
            activated_risk_levels.append(rule['risk_level'])
            activated_tags.extend(rule['tags'])

        # Determinar el detected_risk_level más alto
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        detected_risk_level = "LOW"
        if activated_risk_levels:
            highest_risk_value = max([risk_order[level] for level in activated_risk_levels])
            detected_risk_level = [level for level, value in risk_order.items() if value == highest_risk_value][0]

        # Obtener tags únicos
        detected_rule_tags = list(set(activated_tags))
        
        fraud_context = " ".join(context_parts)

        return fraud_context, detected_risk_level, detected_rule_tags