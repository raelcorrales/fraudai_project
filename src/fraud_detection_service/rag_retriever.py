from typing import List, Dict, Any
from .ollama_base import OllamaBase

RISK_LEVEL_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

class RAGRetriever(OllamaBase):
    def __init__(self, fraud_rules: List[Dict[str, Any]], model_name: str = "llama3.1", max_retries: int = 3, base_delay: float = 1.0):
        super().__init__(model_name, max_retries, base_delay)
        self.fraud_rules = fraud_rules

    def retrieve_context(self, transaction) -> Dict[str, Any]:
        tx = transaction
        tx_str = tx.to_string_for_embedding().lower()
        matched_rules = []

        # 1. Matching heurístico simple por keywords y campos:
        for rule in self.fraud_rules:
            # Por keywords
            if any(kw.lower() in tx_str for kw in rule.get("keywords", [])):
                matched_rules.append(rule)
            # Ejemplo: por campo (puedes añadir más reglas aquí)
            if rule["id"] == "RULE_001_HIGH_VALUE_UNUSUAL_CATEGORY":
                if getattr(tx, "TransactionAmount", 0) > 500 and getattr(tx, "CustomerOccupation", "").lower() == "student":
                    matched_rules.append(rule)
            if rule["id"] == "RULE_006_LARGE_PERCENTAGE_OF_ACCOUNT_BALANCE":
                if getattr(tx, "AccountBalance", 1) > 0 and getattr(tx, "TransactionAmount", 0) / getattr(tx, "AccountBalance", 1) > 0.8:
                    matched_rules.append(rule)
            # Puedes añadir más matching heurístico por regla si lo deseas...

        # Eliminar duplicados
        matched_rules = list({r["id"]: r for r in matched_rules}.values())

        if matched_rules:
            max_rule = max(matched_rules, key=lambda r: RISK_LEVEL_ORDER.get(r.get("risk_level", "LOW"), 1))
            risk_level = max_rule["risk_level"]
            tags = list({tag for rule in matched_rules for tag in rule.get("tags", [])})
            rule_descriptions = [rule["description"] for rule in matched_rules]
            rule_ids = [rule["id"] for rule in matched_rules]
            return {
                "matched_rule_ids": rule_ids,
                "risk_level": risk_level,
                "tags": tags,
                "rule_descriptions": rule_descriptions,
                "explanation_base": "Se detectaron posibles patrones de fraude según las reglas coincidentes."
            }

        # 2. Si no hay match, consulta a Ollama:
        rules_summary = "\n".join([f"{r['id']}: {r['description']} (Nivel: {r['risk_level']}, Tags: {', '.join(r['tags'])})" for r in self.fraud_rules])
        prompt = (
            "Eres un sistema experto en fraude bancario. Dada la siguiente transacción y el resumen de reglas de fraude del banco, "
            "analiza la transacción y sugiere:\n"
            "1. El risk_level más relevante (CRITICAL, HIGH, MEDIUM, LOW, NONE).\n"
            "2. Los tags más relevantes de la lista de reglas (elige uno o varios si aplica).\n"
            "3. Una breve justificación.\n"
            "Devuelve la respuesta en formato JSON así:\n"
            "{ \"risk_level\":..., \"tags\": [...], \"justification\": ... }\n\n"
            f"Resumen de reglas:\n{rules_summary}\n\n"
            f"Transacción:\n{tx.to_string_for_embedding()}\n"
        )

        options = {"temperature": 0.1, "num_predict": 120}
        response = self.generate_request(prompt, options)
        import json
        content = response["response"] if "response" in response else response.get("message", {}).get("content", "")
        try:
            llm_json = json.loads(content)
            risk_level = llm_json.get("risk_level", "NONE")
            tags = llm_json.get("tags", [])
            justification = llm_json.get("justification", "")
        except Exception:
            risk_level = "NONE"
            tags = []
            justification = "No se pudo obtener información relevante del modelo."

        return {
            "matched_rule_ids": [],
            "risk_level": risk_level,
            "tags": tags,
            "rule_descriptions": [],
            "explanation_base": justification
        }
