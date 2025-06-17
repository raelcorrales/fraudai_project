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

    embedding_manager = get_embedding_manager_instance(model_name=model_name)
    
    embedded_rules = []
    failed_embeddings_count = 0
    for rule in rules:
        try:
            # Aquí decides qué texto de la regla quieres embeber.
            # Una buena práctica es combinar la descripción y las palabras clave para un contexto más rico.
            # Considera que la descripción puede ser larga; los modelos de embedding tienen límites de tokens.
            text_to_embed = f"{rule.get('description', '')} {', '.join(rule.get('keywords', []))}"
            
            if not text_to_embed.strip():
                print(f"La regla '{rule.get('id', 'N/A')}' tiene un texto vacío para embeber. Saltando.")
                failed_embeddings_count += 1
                continue

            # Genera el embedding real usando el EmbeddingManager de Ollama
            rule_embedding = embedding_manager.get_embedding(text_to_embed)
            
            if rule_embedding is not None and rule_embedding.size > 0:
                # Añade el embedding a la regla
                # Convertir a lista si es necesario para JSON serialización, pero np.ndarray es preferible para cálculos
                rule['embedding'] = rule_embedding 
                embedded_rules.append(rule)
            else:
                print(f"No se pudo obtener un embedding válido para la regla '{rule.get('id', 'N/A')}'.")
                failed_embeddings_count += 1
                continue

        except Exception as e:
            print(f"Error inesperado al generar embedding para la regla '{rule.get('id', 'N/A')}': {e}")
            failed_embeddings_count += 1
            continue
            
    print(f"Se generaron embeddings para {len(embedded_rules)} de {len(rules)} reglas. Fallaron: {failed_embeddings_count}.")
    return embedded_rules