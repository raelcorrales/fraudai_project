from typing import List, Dict, Any
from .ollama_base import OllamaBase


class LLMExplainer(OllamaBase):
    """
    Generador de explicaciones de fraude utilizando un modelo LLM (Large Language Model).
    Este componente utiliza un modelo LLM para generar explicaciones sobre por qué
    una transacción es considerada sospechosa.
    """
    def __init__(self, model_name='llama3.2', temperature=0.5, max_retries=3, base_delay=1.0):
        super().__init__(model_name=model_name, max_retries=max_retries, base_delay=base_delay)
        self.temperature = temperature            # Temperatura del modelo, por defecto 0.5

    def generate_explanation(self, transaction_details_str: str, matched_rules: List[Dict[str, Any]]) -> str:
        """
        Genera una explicación del posible fraude usando un LLM.
        """
        rules_text = "\n\n".join([
            f"🔸 **{rule['id']}** ({rule['risk_level']}): {rule['description']}\n"
            f"🔹 *Sugerencias de mitigación:* {', '.join(rule['mitigation_steps'])}"
            for rule in matched_rules
        ])

        #prompt = (
        #    f"Eres un experto en detección de fraudes financieros.\n"
        #    f"Una transacción ha sido marcada como sospechosa. A continuación se muestran los detalles:\n\n"
        #    f"**Detalles de la Transacción**:\n{transaction_details_str}\n\n"
        #    f"**Reglas de fraude relevantes encontradas**:\n{rules_text}\n\n"
        #    f"Con base en las reglas y el contexto, genera una explicación concisa de por qué esta transacción es potencialmente fraudulenta. "
        #    f"Incluye el razonamiento y sugiere acciones a tomar."
        #)

        prompt = (
            f"Eres un analista experto en riesgo financiero y detección de anomalías transaccionales.\n"
            f"Tu tarea es evaluar la siguiente transacción que ha sido marcada como **de alto riesgo** o **anómala**.\n"
            f"Basándote en los detalles proporcionados y las reglas de comportamiento sospechoso, genera una explicación concisa del **porqué esta transacción es considerada anómala o potencialmente de riesgo**.\n"
            f"Incluye el razonamiento basado en las reglas aplicables y sugiere las **acciones de mitigación** más relevantes.\n\n"
            f"**Detalles de la Transacción**:\n{transaction_details_str}\n\n"
            f"**Reglas de comportamiento anómalo relevantes encontradas**:\n{rules_text}\n\n"
        )

        # Aquí llamas a tu LLM (ej. Ollama, OpenAI, etc.)
        return self._call_model(prompt)
    
    def _call_model(self, prompt: str) -> str:
        """
        Llama al modelo LLM para generar una respuesta basada en el prompt.
        Este método puede ser sobreescrito por subclases para personalizar la llamada al modelo.
        """
        options = {'temperature': self.temperature}
        try:
            response = self.generate_request(prompt=prompt, options=options)
            return response.get('response', "No se pudo obtener una respuesta válida del modelo.")
        except Exception as e:
            return f"No se pudo generar la explicación: {e}"

        