¡Perfecto! Vamos paso a paso. Abordaremos:

1. ✅ **Carga de `fraud_rules.json` + Embeddings**
2. ✅ **`RAGRetriever` mejorado con matching semántico (cosine similarity)**
3. ✅ **Prompt estructurado para `LLMExplainer`**

---

## 1. 📥 Carga y Embeddings de `fraud_rules.json`

Creamos una función `load_and_embed_rules()` que:

* Carga las reglas.
* Calcula el embedding del campo `description`.
* Guarda `description_embedding` como nuevo campo en cada regla.

### 📁 `embedding_rule_utils.py` (nuevo archivo sugerido)

```python
import json
import os
from typing import List, Dict, Any
from src.fraud_detection_service.embedding_manager import EmbeddingManager

def load_and_embed_rules(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)

    embedding_manager = EmbeddingManager()
    for rule in rules:
        text_to_embed = rule['description']
        rule['description_embedding'] = embedding_manager.get_embedding(text_to_embed)
    return rules
```

---

## 2. 🧠 RAGRetriever con Semantic Matching

Reemplazamos el keyword-matching por **cosine similarity** entre:

* `transaction_details_str` (embed)
* `description_embedding` de cada regla

### 📁 `rag_retriever.py` (modificado)

```python
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGRetriever:
    def __init__(self, embedded_rules: List[Dict[str, Any]]):
        self.embedded_rules = embedded_rules

    def retrieve_context(self, transaction: Any, transaction_embedding: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Recupera reglas más relevantes basadas en similitud semántica.
        """
        if transaction_embedding is None:
            raise ValueError("transaction_embedding no puede ser None")

        # Comparar con todos los embeddings
        similarities = []
        for rule in self.embedded_rules:
            rule_emb = np.array(rule['description_embedding']).reshape(1, -1)
            similarity = cosine_similarity(transaction_embedding.reshape(1, -1), rule_emb)[0][0]
            similarities.append((similarity, rule))

        # Ordenar por similitud descendente y tomar top-N (ej. 3)
        top_rules = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
        return [rule for _, rule in top_rules]
```

---

## 3. 💬 Prompt estructurado para `LLMExplainer`

Diseñamos un prompt en lenguaje natural con contexto real, aprovechando:

* `transaction_details_str`
* las reglas recuperadas (`description`, `risk_level`, `mitigation_steps`)

### 📁 `llm_explainer.py` (modificado)

```python
class LLMExplainer:
    def __init__(self, model_name: str, temperature: float = 0.5):
        self.model_name = model_name
        self.temperature = temperature

    def generate_explanation(self, transaction_details_str: str, matched_rules: List[Dict[str, Any]]) -> str:
        """
        Genera una explicación del posible fraude usando un LLM.
        """
        rules_text = "\n\n".join([
            f"🔸 **{rule['id']}** ({rule['risk_level']}): {rule['description']}\n"
            f"🔹 *Sugerencias de mitigación:* {', '.join(rule['mitigation_steps'])}"
            for rule in matched_rules
        ])

        prompt = (
            f"Eres un experto en detección de fraudes financieros.\n"
            f"Una transacción ha sido marcada como sospechosa. A continuación se muestran los detalles:\n\n"
            f"**Detalles de la Transacción**:\n{transaction_details_str}\n\n"
            f"**Reglas de fraude relevantes encontradas**:\n{rules_text}\n\n"
            f"Con base en las reglas y el contexto, genera una explicación concisa de por qué esta transacción es potencialmente fraudulenta. "
            f"Incluye el razonamiento y sugiere acciones a tomar."
        )

        # Aquí llamas a tu LLM (ej. Ollama, OpenAI, etc.)
        return self._call_model(prompt)

    def _call_model(self, prompt: str) -> str:
        # Aquí va tu integración real con LLM
        # Por ejemplo, con OpenAI:
        # return openai.ChatCompletion.create(...)
        return f"(Simulación) Explicación generada para:\n{prompt[:300]}..."  # Mock para pruebas
```

---

## 🔁 Cambios necesarios en `process_function.py`

### ⚠️ Ajustes clave:

* Cargar las reglas embebidas una sola vez (usa `load_and_embed_rules()`).
* Pasar `transaction_embedding` al `retrieve_context()`.

### Fragmento modificado:

```python
# Reemplaza tu carga de reglas y RAGRetriever por:
from src.fraud_detection_service.embedding_rule_utils import load_and_embed_rules

rules_path = os.path.join('resources', 'fraud_rules.json')
embedded_rules = load_and_embed_rules(rules_path)
rag_retriever = RAGRetriever(embedded_rules)
```

Y en el bloque de transacción fraudulenta:

```python
fraud_context = rag_retriever.retrieve_context(transaction, transaction_embedding)
explanation = llm_explainer.generate_explanation(transaction_details_str, fraud_context)
```

---

## ✅ Resultado: flujo semántico end-to-end

1. Calculas el embedding de la transacción.
2. Encuentras las reglas más cercanas por similitud semántica.
3. Usas esas reglas para explicarlo con un LLM en lenguaje natural.

---

¿Quieres que armemos un ejemplo de prueba para simular este flujo con una transacción dummy y respuesta del LLM (simulada)?
