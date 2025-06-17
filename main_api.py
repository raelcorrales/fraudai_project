import os
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict

from src.process_function import process_transaction
from src.fraud_detection_service.embedding_rule_utils import load_and_embed_rules

app = FastAPI(title="Fraud Detection API")

# LLM Configuration
os.environ['llm_model_name'] = 'llama3.1'
os.environ['llm_model_temperature'] = '0.5'
# RAG Configuration
os.environ['rag_model_name'] = 'llama3.1'
# Chat Model Configuration
os.environ['chat_model_name'] = 'llama3.1'
os.environ['embedding_model_name'] = 'nomic-embed-text'

# Cargamos las reglas embebidas al iniciar API (solo una vez)

FRAUD_RULES = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname('main_app.py')))), 'data', 'fraud_rules.json')

EMBEDDING_MODEL_NAME = os.environ.get('embedding_model_name', 'all-minilm')
embedded_rules = load_and_embed_rules(model_name=EMBEDDING_MODEL_NAME, rules_file_path=FRAUD_RULES)

class TransactionInput(BaseModel):
    # Puedes definir los campos explícitos o aceptar un dict libre
    data: Dict[str, Any]

@app.post("/process-transaction/")
async def process_single_transaction(input: TransactionInput):
    """
    Procesa una sola transacción bancaria enviada en formato JSON.
    Devuelve el resultado completo del pipeline de fraude.
    """
    try:
        result = process_transaction(transaction_data=input.data, fraud_rules=embedded_rules)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Fraud detection API. POST a transaction JSON to /process-transaction/."}