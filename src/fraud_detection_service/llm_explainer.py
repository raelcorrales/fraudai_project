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

    def generate_explanation(self, transaction_details, fraud_context):
        #prompt = f"""Eres un asistente experto en análisis de fraude bancario.
        #Basado en los siguientes detalles de la transacción y el contexto de fraude recuperado,
        #genera una explicación concisa y clara de por qué esta transacción es considerada sospechosa.
        #
        #Detalles de la transacción:
        #{transaction_details}
        #
        #Contexto de fraude relevante:
        #{fraud_context}
        #
        #Explicación:
        #"""
        prompt = f"Eres un analista de riesgos especializado en prevención de fraudes bancarios.\nTu tarea es revisar transacciones sospechosas y generar un informe técnico\nque detalle los posibles motivos de alerta, basados en reglas preestablecidas\nde detección de comportamiento anómalo.\n\n🧾 **Detalles de la transacción**:\n{transaction_details}\n\n📚 **Contexto de reglas activadas**:\n{fraud_context}\n\n🔍 **Informe de evaluación**:\nDescribe por qué esta transacción podría ser considerada atípica o de alto riesgo.\nIncluye un razonamiento claro y, si es posible, sugiere acciones de mitigación.\n"

        print("Prompt usado para generar la explicacion: ", prompt)

        options = {'temperature': self.temperature}

        try:
            # Usamos el método generate_request de la clase base OllamaBase
            # para realizar la solicitud de generación de texto.
            response = self.generate_request(prompt=prompt, options=options)
            print("Respuesta del modelo:", response)
            return response.get('response', "No se pudo obtener una respuesta válida del modelo.")
        except Exception as e:
            return f"No se pudo generar la explicación: {e}"
        