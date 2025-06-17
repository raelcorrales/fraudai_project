# src/fraud_detection_service/embedding_rule_utils.py

import json
import numpy as np
import logging
from typing import List, Dict, Any, Optional

# Asumiendo que EmbeddingManager ya está definido en src/fraud_detection_service/embedding_manager.py
# y tiene un método get_embedding(text: str) -> np.ndarray
from src.fraud_detection_service.embedding_manager import EmbeddingManager

# Creamos una instancia global (o pasamos una instancia) del EmbeddingManager
# para evitar crearlo para cada llamada a load_and_embed_rules.
# Es crucial que el model_name aquí coincida con el modelo de embedding de Ollama que has descargado
# y que usas para generar embeddings de transacciones.
_embedding_manager_instance: Optional[EmbeddingManager] = None

def get_embedding_manager_instance(model_name: str) -> EmbeddingManager:
    """
    Retorna una instancia singleton de EmbeddingManager.
    """
    global _embedding_manager_instance
    if _embedding_manager_instance is None:
        _embedding_manager_instance = EmbeddingManager(model_name=model_name)
    return _embedding_manager_instance


def load_and_embed_rules(model_name: str, rules_file_path: str) -> List[Dict[str, Any]]:
    """
    Carga las reglas de fraude desde un archivo JSON, genera un embedding para cada regla
    utilizando Ollama y el EmbeddingManager, y añade el embedding al diccionario de la regla.

    Args:
        rules_file_path (str): Ruta al archivo JSON que contiene las reglas de fraude.

    Returns:
        List[Dict[str, Any]]: Una lista de diccionarios de reglas, cada uno con un
                               campo 'embedding' adicional.
    """
    if not isinstance(rules_file_path, str) or not rules_file_path:
        print("La ruta del archivo de reglas no es válida.")
        return []

    try:
        with open(rules_file_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        print(f"Reglas cargadas exitosamente desde: {rules_file_path}")
    except FileNotFoundError:
        print(f"Error: El archivo de reglas no se encontró en '{rules_file_path}'.")
        return []
    except json.JSONDecodeError:
        print(f"Error: No se pudo decodificar el JSON del archivo '{rules_file_path}'.")
        return []
    except Exception as e:
        print(f"Error inesperado al cargar las reglas: {e}")
        return []
    return rules