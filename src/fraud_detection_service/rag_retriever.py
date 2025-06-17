import json
import os
from typing import Any, Dict, List

# Define la ruta al archivo JSON de reglas de fraude
# Asume que fraud_rules.json estará en la carpeta 'data/' en la raíz del proyecto.
# Ajusta la ruta si es diferente.
FRAUD_RULES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fraud_rules.json')

class RAGRetriever:
    """
    Recuperador de contexto para un sistema de detección de fraude basado en RAG (Retrieval-Augmented Generation).
    Este componente simula la recuperación de contexto relevante de una base de conocimiento
    de reglas de fraude, utilizando un enfoque simplificado para la demostración.
    En un sistema de producción, se utilizaría una base de datos vectorial para buscar
    por similitud de embeddings.
    """
    def __init__(self, fraud_rules_file: List):
        self.fraud_rules: List[Dict[str, Any]] = fraud_rules_file

    def retrieve_context(self, transaction: Any) -> str:
        """
        Simula la recuperación de contexto relevante de la base de conocimiento de reglas.
        En producción: Buscaría en una base de datos vectorial por similitud.
        Para la demo, busca palabras clave en los atributos de la transacción y devuelve
        la descripción de la regla más relevante.
        """
        transaction_str = transaction.to_string_for_embedding().lower()
        
        # Lógica simplificada para la demo: encontrar la primera regla cuyas palabras clave
        # estén presentes en la descripción de la transacción.
        # En un RAG real, se calcularían embeddings y se buscaría por similitud coseno.
        
        relevant_rule_description = "Patrón general: Esta transacción muestra características que la hacen atípica para el perfil del cliente."
        found_match = False

        for rule in self.fraud_rules:
            rule_keywords_lower = [kw.lower() for kw in rule.get("keywords", [])]
            # Verificar si alguna palabra clave de la regla está en la cadena de la transacción
            if any(keyword in transaction_str for keyword in rule_keywords_lower):
                relevant_rule_description = rule["description"]
                found_match = True
                break # En la demo, tomamos la primera coincidencia
        
        if not found_match:
            # Si no hay match directo por keywords, intenta con atributos específicos
            if transaction.TransactionAmount > 15000 and transaction.CustomerOccupation.lower() == "jubilado":
                 return self.get_rule_description_by_id("RULE_001_HIGH_VALUE_UNUSUAL_CATEGORY")
            if "turquía" in transaction.Location.lower() or transaction.IPAddress == "192.168.1.1":
                return self.get_rule_description_by_id("RULE_002_GEOGRAPHIC_LOCATION_MISMATCH")
            if transaction.LoginAttempts > 3:
                return self.get_rule_description_by_id("RULE_003_MULTIPLE_FAILED_LOGIN_ATTEMPTS")
            if transaction.TransactionDuration < 10 and transaction.LoginAttempts > 1:
                return self.get_rule_description_by_id("RULE_004_SHORT_TRANSACTION_DURATION_AFTER_LOGIN_SPIKE")
            if "unknown_device" in transaction.DeviceID.lower() or "anomalo" in transaction.Channel.lower():
                return self.get_rule_description_by_id("RULE_005_UNUSUAL_DEVICE_OR_CHANNEL")
            if transaction.TransactionAmount / transaction.AccountBalance > 0.8:
                return self.get_rule_description_by_id("RULE_006_LARGE_PERCENTAGE_OF_ACCOUNT_BALANCE")
            if "compra internacional" in transaction.TransactionType.lower() and "viaje reportado" not in transaction_str:
                return self.get_rule_description_by_id("RULE_007_INTERNATIONAL_TRANSACTION_NO_TRAVEL_NOTIFICATION")
            # Agrega más lógica aquí para cubrir otras reglas si las keywords no son suficientes

        return relevant_rule_description

    def get_rule_description_by_id(self, rule_id: str) -> str:
        """Helper para obtener la descripción de una regla por su ID."""
        for rule in self.fraud_rules:
            if rule.get("id") == rule_id:
                return rule.get("description", "Descripción de regla no encontrada.")
        return "Regla específica no encontrada."
