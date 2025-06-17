import numpy as np
import ollama
import logging 

# Asegúrate de ajustar la importación de OllamaBase según la estructura de tu proyecto
from .ollama_base import OllamaBase 
from src.transaction import Transaction # Asumo que Transaction está en src/transaction.py

class AnomalyDetector(OllamaBase): # Hereda de OllamaBase
    """
    Detector de anomalías utilizando Ollama Llama 3.1 para la detección de fraude.
    Este componente utiliza un modelo LLM para "razonar" sobre
    los detalles de la transacción y determinar si es potencialmente fraudulenta.
    En un sistema de producción, se utilizaría un modelo LLM entrenado específicamente
    para la detección de fraude, posiblemente con fine-tuning en un dataset de transacciones
    y reglas de fraude.
    Este es un MVP que utiliza Llama 3.1 para demostrar la capacidad de razonamiento del LLM.
    En un sistema real, se podría integrar con un servicio de embeddings para generar
    embeddings de transacciones y utilizar un modelo LLM especializado en detección de fraude.
    """
    def __init__(self, model_name: str = "llama3.1", max_retries: int = 3, base_delay: float = 1.0):
        # Llama al constructor de la clase base
        super().__init__(model_name=model_name, max_retries=max_retries, base_delay=base_delay)

    def _predict_fraud(self, embedding: np.ndarray, transaction: Transaction) -> bool:
        """
        Simula la detección de anomalías utilizando Ollama Llama 3.1 para "razonar"
        sobre los detalles de la transacción.
        
        Args:
            embedding (np.ndarray): El embedding generado de la transacción (puede ser ignorado
                                     directamente por el LLM en este MVP, pero se mantiene para
                                     compatibilidad con la firma).
            transaction (Transaction): El objeto Transaction con todos los detalles.
        
        Returns:
            bool: True si se detecta fraude, False en caso contrario.
        """
        transaction_details_for_llm = transaction.to_string_for_embedding()

        # Craft a precise prompt for Llama 3.1
        system_prompt = (
            "Eres un experto en detección de fraude bancario. Tu tarea es analizar los detalles de una transacción "
            "y determinar si es potencialmente fraudulenta. Responde SÓLO con 'FRAUDULENTO' o 'NORMAL'."
            "Considera indicadores como montos inusuales, ubicaciones geográficas sospechosas, múltiples intentos de login,"
            "dispositivos o canales no reconocidos, o transacciones que vacían la cuenta."
        )

        user_prompt = (
            f"Analiza la siguiente transacción:\n\n"
            f"{transaction_details_for_llm}\n\n"
            f"Basado en los detalles proporcionados, ¿es esta transacción FRAUDULENTA o NORMAL? "
            f"Responde SÓLO con 'FRAUDULENTO' o 'NORMAL'."
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

        print(f"Mensaje enviado a Ollama Llama 3.1: {messages}")  # Debugging output
        
        options = {
            'temperature': 0.1, # Keep temperature low for more deterministic output
            'num_predict': 20 # Limit output length to encourage concise answer
        }

        try:
            # Usamos el método chat_request de la clase base, que ya incluye la lógica de reintento
            response = self.chat_request(messages=messages, options=options)
            
            print(f"Respuesta de Ollama Llama 3.1: {response}")  # Debugging output
            
            llm_decision = response['message']['content'].strip().upper()
            
            if "FRAUDULENTO" in llm_decision:
                print(f"Ollama Llama 3.1 clasifica como: FRAUDULENTO")
                return True
            elif "NORMAL" in llm_decision:
                print(f"Ollama Llama 3.1 clasifica como: NORMAL")
                return False
            else:
                # Fallback if LLM doesn't give a clear answer
                print(f"Ollama Llama 3.1 respuesta ambigua: '{llm_decision}'. Asumiendo NORMAL.")
                return False

        except Exception as e:
            print(f"Error al llamar a Ollama Llama 3.1 (con reintentos): {e}", exc_info=True)
            print("Volviendo a la detección de fraude basada en reglas simples debido a un error de Ollama.")
            return self._fallback_rule_based_detection(transaction)
    
    def predict_fraud(self, embedding: np.ndarray, transaction: Transaction) -> bool:
        # En tu código original, `predict_fraud` solo llamaba al fallback.
        # Asumo que ahora quieres que intente el LLM primero.
        return self._predict_fraud(embedding, transaction)

    def _fallback_rule_based_detection(self, transaction: Transaction) -> bool:
        """
        Lógica de detección de fraude simplificada como fallback si el LLM falla.
        Esta es la lógica que tenías antes, ligeramente mejorada.
        """
        location_lower = transaction.Location.lower()
        ip_address_lower = transaction.IPAddress.lower()
        category_lower = transaction.TransactionType.lower()
        occupation_lower = transaction.CustomerOccupation.lower()
        channel_lower = transaction.Channel.lower()

        if transaction.TransactionAmount > 1500 and \
           ("estambul, turquía" in location_lower or ip_address_lower == "192.168.1.1"):
            return True
        if transaction.LoginAttempts >= 3 and transaction.TransactionDuration < 30:
            return True
        if transaction.AccountBalance > 0 and \
           transaction.TransactionAmount / transaction.AccountBalance > 0.8:
            return True
        if transaction.CustomerAge < 25 and transaction.TransactionAmount > 1000 and \
           ("inversión" in category_lower or "joyería" in category_lower) and \
           occupation_lower == "estudiante":
            return True
        if "unknown_device" in transaction.DeviceID.lower() or "call center anomalo" in channel_lower:
            return True
        if transaction.TransactionType.lower() == "compra internacional" and \
           transaction.Location not in ["guadalajara, mx", "monterrey, mx", "ciudad de méxico, mx"] and \
           "reportado viaje" not in transaction.to_string_for_embedding().lower():
            return True
        return False