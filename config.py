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
prompt = f"""Eres un asistente experto en análisis de fraude bancario.
        Basado en los siguientes detalles de la transacción y el contexto de fraude recuperado,
        genera una explicación concisa y clara de por qué esta transacción es considerada sospechosa.

        Detalles de la transacción:
        {transaction_details}

        Contexto de fraude relevante:
        {fraud_context}

        Explicación:
        """