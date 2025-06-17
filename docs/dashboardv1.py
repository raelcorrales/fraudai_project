import streamlit as st
import pandas as pd
import json

# Título de la aplicación
st.set_page_config(layout="wide") # Opcional: para usar más ancho de pantalla
st.title("Detección de Fraude en Transacciones")

# --- DESCRIPCIÓN ACTUALIZADA ---
st.markdown("""
Esta aplicación demuestra un sistema de detección de fraude en transacciones financieras
utilizando un enfoque híbrido. **El módulo de detección de anomalías (`AnomalyDetector`)**
emplea el modelo **Llama 3.2 (con una temperatura de 0.1)** para clasificar transacciones
como fraudulentas o normales. Cuando se detecta fraude, **el módulo de explicación (`LLMExplainer`)**
utiliza el mismo modelo **Llama 3.2 (con una temperatura de 0.5)** junto con un recuperador de contexto
(RAG) para generar una explicación detallada del porqué.
""")
# --- FIN DE LA DESCRIPCIÓN ACTUALIZADA ---

st.header("Análisis de Transacciones Procesadas")

# Ruta al archivo JSON Lines generado
JSONL_FILE_PATH = "processed_transactions_results.jsonl"

# Cargar los datos desde el archivo JSONL
@st.cache_data # Cachea los datos para no recargar en cada rerun de Streamlit
def load_data(file_path: str) -> pd.DataFrame:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return pd.DataFrame(data)
    except FileNotFoundError:
        st.error(f"Error: El archivo '{file_path}' no se encontró. Asegúrate de haber ejecutado el script de procesamiento.")
        return pd.DataFrame() # Retorna DataFrame vacío en caso de error
    except json.JSONDecodeError as e:
        st.error(f"Error al decodificar JSON en la línea: {e}. Archivo: {file_path}")
        return pd.DataFrame()

df_processed = load_data(JSONL_FILE_PATH)

if not df_processed.empty:
    st.success(f"Datos de {len(df_processed)} transacciones cargados exitosamente desde '{JSONL_FILE_PATH}'")

    # Mostrar el DataFrame completo (oculto por defecto, se puede expandir)
    with st.expander("Ver todas las Transacciones Procesadas"):
        st.dataframe(df_processed)

    # Métricas clave
    total_transactions = len(df_processed)
    fraud_transactions = df_processed[df_processed['is_fraudulent']].shape[0]
    normal_transactions = total_transactions - fraud_transactions

    st.markdown("---")
    st.subheader("Estadísticas Generales")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Transacciones", total_transactions)
    with col2:
        st.metric("Transacciones Fraudulentas", fraud_transactions, delta=f"{((fraud_transactions/total_transactions)*100):.2f}% del total" if total_transactions > 0 else "0.00%", delta_color="inverse")
    with col3:
        st.metric("Transacciones Normales", normal_transactions)

    st.markdown("---")
    # Filtrar y mostrar solo las transacciones fraudulentas
    st.subheader("Transacciones Potencialmente Fraudulentas")
    fraud_df = df_processed[df_processed['is_fraudulent']].copy() # Usar .copy() para evitar SettingWithCopyWarning
    
    if not fraud_df.empty:
        # Añadir una columna de 'Riesgo' basada en el contexto para demo
        # Esto es solo un ejemplo, podrías tener un campo 'risk_level' en tus reglas JSON
        def categorize_risk(context):
            if "crítico" in context.lower() or "alto_valor" in context.lower() or "vaciar cuenta" in context.lower():
                return "Alto"
            elif "riesgo" in context.lower() or "inusual" in context.lower():
                return "Medio"
            return "Bajo"
        
        fraud_df['Risk_Category'] = fraud_df['fraud_context'].apply(categorize_risk)

        st.dataframe(fraud_df.sort_values(by='Risk_Category', ascending=False))

        # Sección para el análisis detallado de una transacción fraudulenta
        st.markdown("---")
        st.subheader("Análisis Detallado de Fraude por Transacción")
        
        # Crear una lista de opciones para el selectbox que incluya ID y monto
        fraud_options = [
            f"{row['TransactionID']} (Monto: ${row['TransactionAmount']:.2f}, Riesgo: {row['Risk_Category']})"
            for index, row in fraud_df.iterrows()
        ]
        
        selected_fraud_option = st.selectbox(
            "Selecciona una Transacción Fraudulenta para ver su explicación:",
            fraud_options
        )

        if selected_fraud_option:
            # Extraer el TransactionID de la opción seleccionada
            selected_fraud_id = selected_fraud_option.split(' ')[0]
            selected_row = fraud_df[fraud_df['TransactionID'] == selected_fraud_id].iloc[0]
            
            st.markdown(f"### Detalles para TransactionID: `{selected_row['TransactionID']}`")
            col_detail_1, col_detail_2 = st.columns(2)
            
            with col_detail_1:
                st.json(selected_row.drop(['is_fraudulent', 'fraud_context', 'explanation', 'Risk_Category']).to_dict())
            
            with col_detail_2:
                st.markdown("#### Contexto de Fraude (RAG):")
                st.info(selected_row['fraud_context'])
                st.markdown("#### Explicación de la IA (Llama 3):")
                st.success(selected_row['explanation'])

            st.markdown("---")
            st.subheader("Visualizaciones Adicionales (Ejemplos)")
            # Aquí podrías añadir gráficos si tu DataFrame es más rico para ellos
            import plotly.express as px

            # Histograma de montos de transacciones fraudulentas
            fig_amount = px.histogram(fraud_df, x="TransactionAmount",
                                      title="Distribución de Montos en Transacciones Fraudulentas",
                                      color_discrete_sequence=['red'])
            st.plotly_chart(fig_amount, use_container_width=True)

            # Let's create an intermediate DataFrame for clarity
            df_counts = fraud_df['TransactionType'].value_counts().reset_index()
            # By default, reset_index() will create columns 'index' (for the type) and 'count' (for the value)
            # Let's rename them for clarity
            df_counts.columns = ['Tipo de Transacción', 'Conteo'] # Rename columns directly

            # Gráfico de barras por tipo de transacción fraudulenta
            fig_type = px.bar(df_counts,
                  x='Tipo de Transacción', # Now 'Tipo de Transacción' is the correct column name for x-axis
                  y='Conteo',
                  title="Tipos de Transacciones Fraudulentas",
                  labels={'Tipo de Transacción': 'Tipo de Transacción', 'Conteo': 'Número de Transacciones'}, # Updated labels
                  color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_type, use_container_width=True)

    else:
        st.info("🎉 ¡Excelente! No se detectaron transacciones fraudulentas en este conjunto de datos.")

else:
    st.warning("No hay datos disponibles para mostrar. Por favor, asegúrate de que el script de procesamiento haya generado 'processed_transactions_results.jsonl'.")