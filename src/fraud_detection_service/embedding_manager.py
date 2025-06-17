import numpy as np

from typing import Optional

from .ollama_base import OllamaBase


class EmbeddingManager(OllamaBase):
    """
    Gestor de embeddings para un sistema de detección de fraude.
    Este componente se conecta a un servicio de embeddings de Ollama (e.g., Llama 3.1)
    para generar embeddings de texto para transacciones y otros textos relevantes.
    """
    def __init__(self, model_name: str = 'llama3.1', max_retries: int = 3, base_delay: float = 1.0):
        """
        Inicializa el EmbeddingManager.

        Args:
            model_name (str): El nombre del modelo de Ollama a usar para embeddings (ej. 'llama3.1').
            max_retries (int): Número máximo de reintentos en caso de fallo de conexión.
            base_delay (float): Retardo base en segundos para el backoff exponencial.
        """
        # Llama al constructor de la clase base
        super().__init__(model_name=model_name, max_retries=max_retries, base_delay=base_delay)
        
        # Opcional: Probar la conexión al iniciar para verificar que el modelo existe y Ollama está corriendo.
        try:
            print(f"EmbeddingManager: Intentando verificar la disponibilidad del modelo '{self.model_name}' en Ollama...")
            test_embedding = self.get_embedding("test connection", is_initial_check=True)
            if test_embedding is not None and test_embedding.shape[0] > 0:
                print(f"EmbeddingManager: Modelo '{self.model_name}' disponible y respondió con un embedding de dimensión {test_embedding.shape[0]}.")
            else:
                print(f"EmbeddingManager: Advertencia - El modelo '{self.model_name}' parece no haber generado un embedding válido en la verificación inicial.")
        except Exception as e:
            print(f"EmbeddingManager: Error inicial al conectar con Ollama o modelo '{self.model_name}': {e}")
            print(f"Por favor, asegúrate de que Ollama esté corriendo y el modelo esté descargado (`ollama pull {self.model_name}`).")

    def get_embedding(self, text: str, is_initial_check: bool = False) -> Optional[np.ndarray]:
        """
        Obtiene el embedding de un texto utilizando el modelo de Ollama especificado.

        Args:
            text (str): El texto del cual obtener el embedding.
            is_initial_check (bool): Indica si la llamada es parte de la verificación inicial.
                                     Esto evita imprimir mensajes de reintento si solo es una verificación.

        Returns:
            np.ndarray: Un array de NumPy que representa el embedding del texto.
                        Retorna None o un array vacío en caso de fallo persistente.
        """
        try:
            # Usamos el método embeddings_request de la clase base
            response = self.embeddings_request(prompt=text)
            if 'embedding' in response:
                return np.array(response['embedding'])
            else:
                if not is_initial_check:
                    print("Embedding Response:", str(response))
                raise ValueError("La respuesta de Ollama no contiene la clave 'embedding'.")
        except Exception as e:
            if not is_initial_check:
                print(f"Error al obtener el embedding: {e}")
            return None