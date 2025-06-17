import ollama
import random
import time
from typing import List, Dict, Any, Callable, Union

class OllamaBase:
    def __init__(self, model_name: str, max_retries: int = 3, base_delay: float = 1.0):
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _resilience_wrapper(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Envuelve una función con lógica de reintento exponencial con jitter.

        Args:
            func (Callable): La función a ejecutar y reintentar.
            *args: Argumentos posicionales para la función.
            **kwargs: Argumentos de palabra clave para la función.

        Returns:
            Any: El resultado de la función si es exitosa.

        Raises:
            Exception: Si todos los reintentos fallan.
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Intento {attempt + 1}/{self.max_retries}: Error al ejecutar '{func.__name__}': {e}")
                if attempt == self.max_retries - 1:
                    print(f"Se agotaron los intentos para ejecutar '{func.__name__}'.")
                    raise
                
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"Reintentando en {delay:.2f} segundos...")
                time.sleep(delay)

    def generate_request(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una solicitud de generación a Ollama con resiliencia.
        Este método es genérico para llamadas a ollama.generate.
        """
        def _call_generate():
            return ollama.generate(model=self.model_name, prompt=prompt, options=options)
        
        return self._resilience_wrapper(_call_generate)

    def embeddings_request(self, prompt: str) -> Dict[str, Any]:
        """
        Realiza una solicitud de embeddings a Ollama con resiliencia.
        Este método es genérico para llamadas a ollama.embeddings.
        """
        def _call_embeddings():
            return ollama.embeddings(model=self.model_name, prompt=prompt)
        
        return self._resilience_wrapper(_call_embeddings)
    
    def chat_request(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una solicitud de chat a Ollama con resiliencia.
        Este método es genérico para llamadas a ollama.chat.
        """
        def _call_chat():
            return ollama.chat(model=self.model_name, messages=messages, options=options)
        
        return self._resilience_wrapper(_call_chat)