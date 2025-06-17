import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import altair as alt # Para visualizaciones de embeddings más robustas
import matplotlib.pyplot as plt
import seaborn as sns

# --- Importaciones condicionales para TSNE ---
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    st.warning("La librería `scikit-learn` no está instalada. La visualización de embeddings TSNE no estará disponible. Instálala con `pip install scikit-learn`.")

# --- Configuración de la Aplicación ---
st.set_page_config(
    page_title="Análisis de Transacciones",
    page_icon="⚡",
    layout="wide"
)

# --- Función para generar datos de ejemplo (para demostración) ---
@st.cache_data
def generate_sample_data(num_records=5000): # Aumentado el número de registros para más variedad y mejores visualizaciones
    """Genera un DataFrame con datos de transacciones de ejemplo con los nombres de columna actualizados."""
    np.random.seed(42) # Para reproducibilidad

    data = {
        'TransactionID': [f'TXN{i:05d}' for i in range(num_records)],
        'AccountID': [f'ACC{np.random.randint(1, 500):03d}' for _ in range(num_records)], # Más cuentas
        'TransactionDate': pd.to_datetime(pd.date_range('2024-01-01', periods=num_records, freq='H').to_numpy() +
                                         pd.to_timedelta(np.random.randint(-24, 24, num_records), unit='H')),
        'TransactionAmount': np.round(np.random.lognormal(mean=3, sigma=1.0, size=num_records), 2) * 10,
        'TransactionType': np.random.choice(['Crédito', 'Débito'], num_records, p=[0.3, 0.7]),
        'Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte'], num_records), # Más ciudades
        'DeviceID': np.random.choice([f'DEV{i:03d}' for i in range(50)], num_records), # Más dispositivos
        'IP Address': [f'192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}' for _ in range(num_records)],
        'MerchantID': np.random.choice([f'MERCH{i:03d}' for i in range(200)], num_records), # Más comerciantes
        'AccountBalance': np.round(np.random.lognormal(mean=7, sigma=0.8, size=num_records), 2) * 100,
        'PreviousTransactionDate': pd.to_datetime(pd.date_range('2023-12-01', periods=num_records, freq='H').to_numpy() +
                                                  pd.to_timedelta(np.random.randint(-720, 0, num_records), unit='H')),
        'Channel': np.random.choice(['Online', 'ATM', 'Branch', 'Mobile'], num_records, p=[0.4, 0.25, 0.2, 0.15]),
        'CustomerAge': np.random.randint(18, 70, num_records),
        'CustomerOccupation': np.random.choice(['Engineer', 'Doctor', 'Student', 'Retired', 'Artist', 'Teacher', 'Businessman', 'Analyst', 'Consultant', 'Chef', 'Lawyer'], num_records), # Más ocupaciones
        'TransactionDuration': np.round(np.random.uniform(5, 120, num_records), 0),
        'LoginAttempts': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], num_records, p=[0.5, 0.2, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01]), # Más intentos
        'is_fraudulent': np.random.choice([True, False], num_records, p=[0.05, 0.95]), # Aumentar ligeramente el fraude para visibilidad
        'transaction_details_str_for_llm': ['detalle de transacción de ejemplo' for _ in range(num_records)],
        'transaction_embedding_shape': [np.random.rand(128).tolist() for _ in range(num_records)], # Ejemplo de embedding
        'fraud_context': [np.random.choice([None, 'Actividad inusual en la madrugada', 'Múltiples intentos fallidos de login', 'Compra de alto valor en nueva ubicación', 'Transacción con IP sospechosa', 'Patrón de gasto fuera de lo normal del usuario', 'Dispositivo no reconocido para esta cuenta', 'Velocidad de transacción anómala'], 1, p=[0.8, 0.03, 0.05, 0.02, 0.03, 0.04, 0.02, 0.01])[0] for _ in range(num_records)],
        'explanation': [np.random.choice([None, 'El sistema detectó un comportamiento anómalo en el patrón de gasto del usuario, sugiriendo un posible fraude. El monto excede los límites habituales del usuario para este tipo de transacción y canal.', 'Alerta por alta duración de la transacción y discrepancia en la dirección IP. La ubicación de la transacción es inconsistente con el historial del usuario.', 'Múltiples intentos de inicio de sesión fallidos seguidos de una transacción exitosa de gran valor. El contexto RAG indica una IP de riesgo conocida asociada a actividades maliciosas.', 'Esta transacción es parte de una cadena de transacciones sospechosas originadas desde un dispositivo no reconocido, con montos y frecuencias inusuales.', 'La transacción ocurrió desde una nueva ubicación geográfica y con un monto elevado, lo cual no es habitual para este AccountID. El LLM sugiere una potencial suplantación de identidad.', 'La combinación de un LoginAttempts alto y un TransactionDuration bajo para un monto significativo activó la alarma. El contexto RAG apunta a una IP proxy.','Comportamiento anómalo: el MerchantID no coincide con los patrones de compra históricos de la cuenta, y la TransactionDuration fue extremadamente corta.'], 1, p=[0.8, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])[0] for _ in range(num_records)]
    }

    df = pd.DataFrame(data)

    # Asignar un saldo_inicial base para cada cuenta para simular un AccountBalance más realista
    account_initial_balance = {acc_id: np.random.lognormal(mean=7, sigma=0.8) * 100 for acc_id in df['AccountID'].unique()}
    df['AccountBalance'] = df['AccountID'].map(account_initial_balance)

    # Ordenar por cuenta y fecha para simular el saldo progresivo
    df = df.sort_values(by=['AccountID', 'TransactionDate']).reset_index(drop=True)

    for index, row in df.iterrows():
        if row['TransactionType'] == 'Débito':
            df.loc[index, 'AccountBalance'] -= row['TransactionAmount']
        elif row['TransactionType'] == 'Crédito':
            df.loc[index, 'AccountBalance'] += row['TransactionAmount']
        df.loc[index, 'AccountBalance'] = max(0, df.loc[index, 'AccountBalance']) # Asegurar no negativos

    # Asegurar que fraud_context y explanation sean None si is_fraudulent es False
    df.loc[df['is_fraudulent'] == False, ['fraud_context', 'explanation']] = None

    return df

# --- Función para cargar y procesar los datos ---
@st.cache_data
def load_transactions_data(file_source):
    """
    Carga y procesa un archivo JSON Lines en un DataFrame de Pandas.
    Asegura los tipos de datos correctos y calcula columnas adicionales si no existen.
    """
    try:
        if isinstance(file_source, str): # Es una ruta de archivo
            if not os.path.exists(file_source):
                st.error(f"Error: El archivo no fue encontrado en la ruta: `{file_source}`")
                return None
            df = pd.read_json(file_source, lines=True)
        else: # Es un objeto uploaded_file de Streamlit
            df = pd.read_json(file_source, lines=True)

        # Convertir a datetime
        for col in ['TransactionDate', 'PreviousTransactionDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        df.dropna(subset=['TransactionDate'], inplace=True) # Eliminar filas con fechas no válidas

        # Convertir a numérico
        numeric_cols = ['TransactionAmount', 'AccountBalance', 'LoginAttempts', 'CustomerAge', 'TransactionDuration']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Limpiar NaNs solo de las columnas críticas para cálculos
        df.dropna(subset=['TransactionAmount', 'AccountBalance', 'LoginAttempts', 'CustomerAge'], inplace=True)

        # Asegurar que 'is_fraudulent' sea booleano
        if 'is_fraudulent' in df.columns:
            df['is_fraudulent'] = df['is_fraudulent'].astype(bool)
        else: # Si no existe, crearla y ponerla en False por defecto
            df['is_fraudulent'] = False

        # Asegurar que fraud_context y explanation sean None si is_fraudulent es False
        if 'is_fraudulent' in df.columns and 'fraud_context' in df.columns and 'explanation' in df.columns:
            df.loc[df['is_fraudulent'] == False, ['fraud_context', 'explanation']] = None
        elif 'is_fraudulent' in df.columns: # Si 'is_fraudulent' existe pero 'fraud_context'/'explanation' no
             if 'fraud_context' not in df.columns: df['fraud_context'] = None
             if 'explanation' not in df.columns: df['explanation'] = None
        else: # Si ninguna de las columnas de fraude existe
            df['is_fraudulent'] = False
            df['fraud_context'] = None
            df['explanation'] = None

        return df
    except ValueError as e:
        st.error(f"Error al leer/parsear el archivo JSON Lines: Asegúrate de que el archivo esté bien formado y las columnas esperadas existan. Detalles: `{e}`")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al cargar el archivo. Detalles: `{e}`")
        return None

# --- Contenido Principal del Dashboard ---
st.title("⚡ Análisis de Transacciones")
st.markdown("Bienvenido al dashboard de análisis de transacciones. Explora las métricas clave, la detección de fraude e insights del cliente.")

# Sección de Carga de Datos
st.markdown("---")
st.subheader("Carga de Datos")
uploaded_file = st.file_uploader(
    "Sube tu archivo JSON Lines (.jsonl)",
    type=["jsonl"],
    help="Arrastra y suelta tu archivo .jsonl aquí o haz clic para buscarlo. Asegúrate de que las columnas `is_fraudulent`, `fraud_context` y `explanation` estén presentes para el análisis de fraude."
)

df_transactions = None
if uploaded_file is not None:
    st.info("Archivo subido exitosamente. Cargando datos...")
    df_transactions = load_transactions_data(uploaded_file)
else:
    st.info("Sube un archivo para comenzar o genera datos de ejemplo si no tienes uno.")
    if st.button("Generar Datos de Ejemplo (para demostración)"):
        df_transactions = generate_sample_data()
        st.cache_data.clear() # Limpiar caché para cargar los nuevos datos
        st.rerun() # Recargar la app con los datos de ejemplo

if df_transactions is None or df_transactions.empty:
    st.warning("No hay datos cargados o los datos están vacíos. Por favor, sube un archivo o genera datos de ejemplo.")
    st.stop() # Detener la ejecución si no hay datos


st.success(f"Datos de transacciones cargados con éxito. Total de {len(df_transactions)} registros.")


# --- Opciones de Filtrado ---
st.markdown("---")
st.subheader("Opciones de Filtrado")

col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)

# Filtro de fecha
with col_filter1:
    min_date_data = df_transactions['TransactionDate'].min().date()
    max_date_data = df_transactions['TransactionDate'].max().date()
    date_range_selected = st.date_input(
        "Rango de Fechas",
        value=(min_date_data, max_date_data),
        min_value=min_date_data,
        max_value=max_date_data
    )
    if len(date_range_selected) == 2:
        df_filtered = df_transactions[(df_transactions['TransactionDate'].dt.date >= date_range_selected[0]) &
                                     (df_transactions['TransactionDate'].dt.date <= date_range_selected[1])]
    else:
        df_filtered = df_transactions.copy() # Usar el DataFrame completo si no hay rango válido

# Filtro por tipo de transacción
with col_filter2:
    if 'TransactionType' in df_filtered.columns:
        selected_types = st.multiselect(
            "Tipo de Transacción",
            options=df_filtered['TransactionType'].unique(),
            default=df_filtered['TransactionType'].unique()
        )
        df_filtered = df_filtered[df_filtered['TransactionType'].isin(selected_types)]

# Filtro por canal
with col_filter3:
    if 'Channel' in df_filtered.columns:
        selected_channels = st.multiselect(
            "Canal",
            options=df_filtered['Channel'].unique(),
            default=df_filtered['Channel'].unique()
        )
        df_filtered = df_filtered[df_filtered['Channel'].isin(selected_channels)]

# Nuevo filtro: Ver solo transacciones fraudulentas
with col_filter4:
    show_fraud_only = st.checkbox("Ver solo transacciones fraudulentas", value=False)
    if show_fraud_only:
        df_filtered = df_filtered[df_filtered['is_fraudulent'] == True]

# Mensaje si no hay datos después de aplicar filtros
if df_filtered.empty:
    st.warning("No hay datos que coincidan con los filtros seleccionados. Por favor, ajusta tus filtros.")
    st.stop() # Detener la ejecución si no hay datos filtrados


st.info(f"Mostrando {len(df_filtered)} transacciones después de aplicar filtros.")

# --- Pestañas para organizar el dashboard ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visión General", "🔒 Ciberseguridad y Fraude", "👥 Insights de Cliente", "📈 Análisis Detallado"])

with tab1:
    st.header("Visión General del Rendimiento Transaccional")
    st.markdown("""
    Esta sección ofrece una **vista rápida y de alto nivel** del rendimiento general de las transacciones.
    Ideal para **C-levels** y equipos que necesitan comprender el pulso del negocio de un vistazo.
    """)

    # Métricas Clave
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric("Total de Transacciones", f"{len(df_filtered):,}")
    with col_kpi2:
        st.metric("Volumen Total Transaccionado", f"${df_filtered['TransactionAmount'].sum():,.2f}")
    with col_kpi3:
        st.metric("Monto Promedio por Transacción", f"${df_filtered['TransactionAmount'].mean():,.2f}")

    # Tendencia de Transacciones y Volumen
    st.subheader("Tendencia Histórica de Transacciones y Volumen")
    st.write("Observa el crecimiento, la estacionalidad y los picos de actividad a lo largo del tiempo.")
    if 'TransactionDate' in df_filtered.columns and 'TransactionAmount' in df_filtered.columns:
        time_agg_general = st.radio("Agregación de Tiempo:", ('Día', 'Semana', 'Mes'), key='time_agg_general', horizontal=True)
        if time_agg_general == 'Día':
            df_trend_general = df_filtered.set_index('TransactionDate').resample('D').agg(
                Total_Transacciones=('TransactionID', 'count'),
                Volumen_Transaccionado=('TransactionAmount', 'sum')
            ).reset_index()
        elif time_agg_general == 'Semana':
            df_trend_general = df_filtered.set_index('TransactionDate').resample('W').agg(
                Total_Transacciones=('TransactionID', 'count'),
                Volumen_Transaccionado=('TransactionAmount', 'sum')
            ).reset_index()
        else: # Mes
            df_trend_general = df_filtered.set_index('TransactionDate').resample('M').agg(
                Total_Transacciones=('TransactionID', 'count'),
                Volumen_Transaccionado=('TransactionAmount', 'sum')
            ).reset_index()

        fig_general_trend = go.Figure()
        fig_general_trend.add_trace(go.Scatter(x=df_trend_general['TransactionDate'], y=df_trend_general['Total_Transacciones'], mode='lines+markers', name='Total Transacciones', line=dict(color='blue')))
        fig_general_trend.add_trace(go.Scatter(x=df_trend_general['TransactionDate'], y=df_trend_general['Volumen_Transaccionado'], mode='lines+markers', name='Volumen Transaccionado ($)', yaxis='y2', line=dict(color='green')))
        fig_general_trend.update_layout(
            title='Tendencia de Transacciones y Volumen General',
            xaxis_title='Fecha',
            yaxis_title='Total Transacciones',
            yaxis2=dict(title='Volumen Transaccionado ($)', overlaying='y', side='right'),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_general_trend, use_container_width=True)
    else:
        st.warning("Columnas 'TransactionDate' o 'TransactionAmount' no encontradas para este gráfico.")

    col_general1, col_general2 = st.columns(2)
    with col_general1:
        # Distribución de Transacciones por TransactionType
        st.subheader("Transacciones por Tipo")
        if 'TransactionType' in df_filtered.columns:
            type_counts_general = df_filtered['TransactionType'].value_counts().reset_index()
            type_counts_general.columns = ['TransactionType', 'Count']
            fig_general_type = px.pie(type_counts_general, names='TransactionType', values='Count',
                                      title='Proporción de Transacciones por Tipo',
                                      color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_general_type, use_container_width=True)
        else:
            st.warning("Columna 'TransactionType' no encontrada.")

    with col_general2:
        # Transacciones por Channel
        st.subheader("Transacciones por Canal")
        if 'Channel' in df_filtered.columns:
            channel_counts_general = df_filtered['Channel'].value_counts().reset_index()
            channel_counts_general.columns = ['Channel', 'Count']
            fig_general_channel = px.bar(channel_counts_general, x='Count', y='Channel', orientation='h',
                                         title='Número de Transacciones por Canal',
                                         labels={'Count': 'Número de Transacciones', 'Channel': 'Canal'},
                                         color='Channel',
                                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_general_channel, use_container_width=True)
        else:
            st.warning("Columna 'Channel' no encontrada.")


with tab2:
    st.header("🔒 Ciberseguridad y Fraude: Análisis Avanzado")
    st.markdown("""
    Esta sección es crítica para **expertos en ciberseguridad** y **C-Levels** para detectar, analizar y comprender profundamente
    los patrones y las razones detrás de las transacciones fraudulentas. **¡Cada dato aquí es un indicio potencial!**
    """)

    fraud_df = df_filtered[df_filtered['is_fraudulent'] == True]

    st.subheader("Métricas de Fraude y Rendimiento de Detección")
    col_fraud_kpi1, col_fraud_kpi2, col_fraud_kpi3 = st.columns(3)
    with col_fraud_kpi1:
        st.metric("Transacciones Fraudulentas Detectadas", f"{len(fraud_df):,}")
    with col_fraud_kpi2:
        if 'TransactionAmount' in fraud_df.columns:
            st.metric("Monto Total de Fraude", f"${fraud_df['TransactionAmount'].sum():,.2f}")
        else:
            st.metric("Monto Total de Fraude", "N/A")
    with col_fraud_kpi3:
        if len(df_filtered) > 0:
            st.metric("Ratio de Fraude (%)", f"{len(fraud_df) / len(df_filtered) * 100:,.2f}%")
        else:
            st.metric("Ratio de Fraude (%)", "0.00%")

    if fraud_df.empty:
        st.info("No hay transacciones marcadas como fraudulentas con los filtros actuales. Ajusta los filtros o carga más datos con fraude para ver el análisis de seguridad.")
    else:
        st.markdown("---")
        st.subheader("Detección de Anomalías Visual (Scatter Plots)")
        st.write("""
        Estos gráficos ayudan a **identificar transacciones inusuales** que se desvían de los patrones normales.
        Observa los **puntos rojos** para ver dónde se agrupa o se dispersa la actividad fraudulenta.
        """)
        col_anom1, col_anom2 = st.columns(2)
        with col_anom1:
            if 'TransactionAmount' in df_filtered.columns and 'TransactionDuration' in df_filtered.columns:
                fig_amount_duration = px.scatter(df_filtered, x='TransactionDuration', y='TransactionAmount', color='is_fraudulent',
                                                 title='Monto vs. Duración de la Transacción',
                                                 labels={'TransactionDuration': 'Duración (segundos)', 'TransactionAmount': 'Monto ($)'},
                                                 hover_data=['TransactionID', 'AccountID', 'TransactionType', 'fraud_context', 'explanation'],
                                                 color_discrete_map={True: 'red', False: 'blue'},
                                                 size='TransactionAmount', log_y=True)
                st.plotly_chart(fig_amount_duration, use_container_width=True)
            else:
                st.warning("Columnas 'TransactionAmount' o 'TransactionDuration' no encontradas para este gráfico.")
        with col_anom2:
            if 'LoginAttempts' in df_filtered.columns and 'TransactionAmount' in df_filtered.columns:
                fig_login_amount = px.scatter(df_filtered, x='LoginAttempts', y='TransactionAmount', color='is_fraudulent',
                                              title='Monto vs. Intentos de Inicio de Sesión',
                                              labels={'LoginAttempts': 'Intentos de Inicio de Sesión', 'TransactionAmount': 'Monto ($)'},
                                              hover_data=['TransactionID', 'AccountID', 'TransactionType', 'Location', 'fraud_context', 'explanation'],
                                              color_discrete_map={True: 'red', False: 'blue'},
                                              size='TransactionAmount', log_y=True)
                st.plotly_chart(fig_login_amount, use_container_width=True)
            else:
                st.warning("Columnas 'LoginAttempts' o 'TransactionAmount' no encontradas.")

        st.markdown("---")
        st.subheader("Tendencias y Distribuciones de Fraude")
        col_fraud_dist1, col_fraud_dist2 = st.columns(2)
        with col_fraud_dist1:
            st.subheader("Fraude por Canal y Tipo")
            if 'Channel' in fraud_df.columns and 'TransactionType' in fraud_df.columns:
                fig_fraud_channel = px.pie(fraud_df, names='Channel', title='Fraude por Canal', hole=0.3, color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig_fraud_channel, use_container_width=True)
                fig_fraud_type = px.pie(fraud_df, names='TransactionType', title='Fraude por Tipo de Transacción', hole=0.3, color_discrete_sequence=px.colors.qualitative.Vivid)
                st.plotly_chart(fig_fraud_type, use_container_width=True)
            else:
                st.warning("Columnas 'Channel' o 'TransactionType' no encontradas para estos gráficos.")
        with col_fraud_dist2:
            st.subheader("Tendencia de Transacciones Fraudulentas")
            if 'TransactionDate' in fraud_df.columns:
                fraud_trend_daily = fraud_df.set_index('TransactionDate').resample('D').agg(Fraud_Count=('TransactionID', 'count')).reset_index()
                fig_fraud_trend = px.line(fraud_trend_daily, x='TransactionDate', y='Fraud_Count',
                                          title='Transacciones Fraudulentas por Día',
                                          labels={'TransactionDate': 'Fecha', 'Fraud_Count': 'Número de Fraudes'},
                                          markers=True, line_shape='spline', color_discrete_sequence=['red'])
                st.plotly_chart(fig_fraud_trend, use_container_width=True)
            else:
                st.warning("Columna 'TransactionDate' no encontrada para este gráfico.")


        st.markdown("---")
        st.subheader("Hotspots de Fraude: Ubicación, IP y Dispositivo")
        col_hotspot1, col_hotspot2 = st.columns(2)
        with col_hotspot1:
            if 'Location' in fraud_df.columns:
                location_fraud_counts = fraud_df['Location'].value_counts().nlargest(10).reset_index()
                location_fraud_counts.columns = ['Location', 'FraudulentTransactions']
                fig_fraud_loc = px.bar(location_fraud_counts, x='Location', y='FraudulentTransactions',
                                       title='Top 10 Ubicaciones con Fraude',
                                       labels={'Location': 'Ubicación', 'FraudulentTransactions': 'Número de Fraudes'},
                                       color='Location', color_discrete_sequence=px.colors.qualitative.Dark2)
                st.plotly_chart(fig_fraud_loc, use_container_width=True)
            else:
                st.warning("Columna 'Location' no encontrada para este gráfico.")
        with col_hotspot2:
            if 'DeviceID' in fraud_df.columns:
                device_fraud_counts = fraud_df['DeviceID'].value_counts().nlargest(10).reset_index()
                device_fraud_counts.columns = ['DeviceID', 'FraudulentTransactions']
                fig_fraud_dev = px.bar(device_fraud_counts, x='DeviceID', y='FraudulentTransactions',
                                       title='Top 10 Dispositivos con Fraude',
                                       labels={'DeviceID': 'ID Dispositivo', 'FraudulentTransactions': 'Número de Fraudes'},
                                       color='DeviceID', color_discrete_sequence=px.colors.qualitative.G10)
                st.plotly_chart(fig_fraud_dev, use_container_width=True)
            else:
                st.warning("Columna 'DeviceID' no encontrada para este gráfico.")

        if 'IP Address' in fraud_df.columns:
            st.subheader("Principales Direcciones IP Involucradas en Fraude")
            ip_fraud_counts = fraud_df['IP Address'].value_counts().nlargest(10).reset_index()
            ip_fraud_counts.columns = ['IP Address', 'FraudulentTransactions']
            fig_fraud_ip = px.bar(ip_fraud_counts, x='IP Address', y='FraudulentTransactions',
                                  title='Top 10 Direcciones IP con Fraude',
                                  labels={'IP Address': 'Dirección IP', 'FraudulentTransactions': 'Número de Fraudes'},
                                  color='IP Address', color_discrete_sequence=px.colors.qualitative.Alphabet)
            st.plotly_chart(fig_fraud_ip, use_container_width=True)
        else:
            st.warning("Columna 'IP Address' no encontrada para este gráfico.")


        st.markdown("---")
        st.subheader("Visualización Avanzada de Embeddings para Detección de Fraude")
        st.write("""
        Los **embeddings de transacciones** representan las características de cada transacción en un espacio multidimensional.
        Al reducirlos a 2D con **t-SNE**, podemos visualizar **clusters de transacciones similares**.
        Observa cómo los **puntos rojos (fraudulentos)** tienden a agruparse, indicando patrones detectados por el modelo.
        Pasa el cursor sobre los puntos para ver el **`fraud_context` y la `explanation`**.
        """)
        if TSNE_AVAILABLE and 'transaction_embedding_shape' in df_filtered.columns and len(df_filtered) >= 50: # TSNE necesita al menos 2*perplexity + 1 data points
            try:
                embeddings = np.array(df_filtered['transaction_embedding_shape'].tolist())
                if embeddings.shape[1] > 2: # Solo aplicar TSNE si es > 2D
                    tsne_perplexity = st.slider("Perplexity para t-SNE (ajusta la granularidad de los clusters):", 5, min(50, int(len(embeddings)/2) -1), 30)
                    with st.spinner(f"Calculando t-SNE con perplexity {tsne_perplexity}... Esto puede tomar un momento."):
                        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, learning_rate='auto', init='random', n_iter=500)
                        tsne_results = tsne.fit_transform(embeddings)
                    df_filtered['tsne_x'] = tsne_results[:,0]
                    df_filtered['tsne_y'] = tsne_results[:,1]

                    chart = alt.Chart(df_filtered).mark_point(opacity=0.7).encode(
                        x=alt.X('tsne_x', axis=alt.Axis(title='Componente t-SNE 1')),
                        y=alt.Y('tsne_y', axis=alt.Axis(title='Componente t-SNE 2')),
                        color=alt.Color('is_fraudulent:N', title='Es Fraudulenta', scale=alt.Scale(domain=[True, False], range=['red', 'blue'])),
                        tooltip=[
                            'TransactionID', 'TransactionAmount', 'TransactionDate', 'AccountID',
                            alt.Tooltip('is_fraudulent:N', title='Es Fraudulenta'),
                            alt.Tooltip('fraud_context:N', title='Contexto Fraude'),
                            alt.Tooltip('explanation:N', title='Explicación')
                        ]
                    ).properties(
                        title='Embeddings de Transacciones Reducidas con t-SNE'
                    ).interactive() # Permite zoom y paneo
                    st.altair_chart(chart, use_container_width=True)
                else: # Si ya es 2D o 1D, no aplicar t-SNE
                    st.info("Los embeddings ya tienen 2 o menos dimensiones. Mostrando directamente.")
                    df_filtered['emb_x'] = embeddings[:,0]
                    df_filtered['emb_y'] = embeddings[:,1] if embeddings.shape[1] > 1 else 0 # Handle 1D case

                    chart = alt.Chart(df_filtered).mark_point(opacity=0.7).encode(
                        x=alt.X('emb_x', axis=alt.Axis(title='Componente Embedding 1')),
                        y=alt.Y('emb_y', axis=alt.Axis(title='Componente Embedding 2')),
                        color=alt.Color('is_fraudulent:N', title='Es Fraudulenta', scale=alt.Scale(domain=[True, False], range=['red', 'blue'])),
                        tooltip=[
                            'TransactionID', 'TransactionAmount', 'TransactionDate', 'AccountID',
                            alt.Tooltip('is_fraudulent:N', title='Es Fraudulenta'),
                            alt.Tooltip('fraud_context:N', title='Contexto Fraude'),
                            alt.Tooltip('explanation:N', title='Explicación')
                        ]
                    ).properties(
                        title='Embeddings de Transacciones (Originales 2D o menos)'
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Error al generar el gráfico de embeddings con t-SNE. Asegúrate de que los embeddings sean válidos: {e}")
        else:
            st.info("La columna 'transaction_embedding_shape' no fue encontrada, no hay suficientes datos (>50 registros), o scikit-learn no está instalado para generar el gráfico de embeddings.")


        st.markdown("---")
        st.subheader("Insights Profundos de Fraude (LLM & RAG)")
        st.write("""
        Esta es la información más valiosa: el **porqué** detrás de la detección de fraude.
        El **contexto RAG** te muestra los datos relevantes que el modelo usó, y la **explicación del LLM**
        traduce esa información en un lenguaje comprensible.
        """)

        if not fraud_df.empty:
            fraud_display_cols = [
                'TransactionID', 'TransactionDate', 'TransactionAmount',
                'AccountID', 'Location', 'LoginAttempts', 'Channel',
                'fraud_context', 'explanation', 'IP Address', 'MerchantID', 'DeviceID'
            ]
            fraud_display_df = fraud_df[[col for col in fraud_display_cols if col in fraud_df.columns]].copy()

            # Ordenar por fecha para ver los fraudes más recientes primero
            fraud_display_df = fraud_display_df.sort_values(by='TransactionDate', ascending=False)

            st.dataframe(
                fraud_display_df,
                use_container_width=True,
                height=300,
                column_config={
                    "TransactionID": st.column_config.Column("Transaction ID", help="Identificador único de la transacción"),
                    "TransactionDate": st.column_config.DatetimeColumn("Fecha", format="YYYY-MM-DD HH:mm"),
                    "TransactionAmount": st.column_config.NumberColumn("Monto ($)", format="$%.2f"),
                    "AccountID": st.column_config.Column("Account ID", help="Identificador de la cuenta"),
                    "Location": st.column_config.Column("Ubicación"),
                    "LoginAttempts": st.column_config.NumberColumn("Intentos Login"),
                    "Channel": st.column_config.Column("Canal"),
                    "IP Address": st.column_config.Column("Dirección IP"),
                    "MerchantID": st.column_config.Column("Merchant ID"),
                    "DeviceID": st.column_config.Column("Device ID"),
                    "fraud_context": st.column_config.Column("Contexto RAG", width="medium"),
                    "explanation": st.column_config.Column("Explicación LLM", width="large")
                }
            )

            selected_fraud_txn_id = st.selectbox(
                "Selecciona un **TransactionID fraudulento** de la tabla para ver detalles expandidos:",
                options=[''] + fraud_display_df['TransactionID'].unique().tolist(), # Añadir opción vacía
                key='select_fraud_txn_llm'
            )

            if selected_fraud_txn_id:
                selected_fraud_row = fraud_df[fraud_df['TransactionID'] == selected_fraud_txn_id].iloc[0]
                st.markdown(f"### **Detalles Expandidos para TransactionID: `{selected_fraud_row['TransactionID']}`**")

                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.markdown(f"**Fecha:** `{selected_fraud_row['TransactionDate'].strftime('%Y-%m-%d %H:%M:%S')}`")
                    st.markdown(f"**Monto:** `${selected_fraud_row['TransactionAmount']:,.2f}`")
                    st.markdown(f"**Cuenta ID:** `{selected_fraud_row['AccountID']}`")
                    st.markdown(f"**Tipo de Transacción:** `{selected_fraud_row['TransactionType']}`")
                    st.markdown(f"**Ubicación:** `{selected_fraud_row['Location']}`")
                with col_det2:
                    st.markdown(f"**Intentos de Login:** `{selected_fraud_row['LoginAttempts']}`")
                    st.markdown(f"**Canal:** `{selected_fraud_row['Channel']}`")
                    st.markdown(f"**Dirección IP:** `{selected_fraud_row['IP Address']}`")
                    st.markdown(f"**Merchant ID:** `{selected_fraud_row['MerchantID']}`")
                    st.markdown(f"**Device ID:** `{selected_fraud_row['DeviceID']}`")

                st.markdown("#### **Contexto de Fraude (RAG)**")
                if pd.notna(selected_fraud_row['fraud_context']) and selected_fraud_row['fraud_context']:
                    st.info(f"👉 {selected_fraud_row['fraud_context']}")
                else:
                    st.warning("No hay contexto RAG disponible para esta transacción.")

                st.markdown("#### **Explicación (LLM)**")
                if pd.notna(selected_fraud_row['explanation']) and selected_fraud_row['explanation']:
                    st.success(f"💡 {selected_fraud_row['explanation']}")
                else:
                    st.warning("No hay explicación del LLM disponible para esta transacción.")

                st.markdown("---")
                st.write("Ver todos los datos brutos de la transacción en formato JSON:")
                st.json(fraud_df[fraud_df['TransactionID'] == selected_fraud_txn_id].iloc[0].to_dict())

        else:
            st.info("No hay transacciones fraudulentas para mostrar los detalles del LLM/RAG con los filtros actuales.")


with tab3:
    st.header("👥 Insights del Cliente: Conoce a tus Usuarios")
    st.markdown("""
    Esta sección profundiza en el **comportamiento y las características demográficas de tus clientes**.
    Información crucial para **equipos de ventas y marketing**, así como para la **planificación estratégica de C-levels**.
    """)

    col_cust_overview1, col_cust_overview2 = st.columns(2)
    with col_cust_overview1:
        st.subheader("Distribución por Edad del Cliente")
        if 'CustomerAge' in df_filtered.columns:
            fig_cust_age = px.histogram(df_filtered, x='CustomerAge', nbins=20,
                                        title='Distribución de Edades de Clientes',
                                        labels={'CustomerAge': 'Edad del Cliente', 'count': 'Número de Clientes'},
                                        marginal="box", color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig_cust_age, use_container_width=True)
        else:
            st.warning("Columna 'CustomerAge' no encontrada.")
    with col_cust_overview2:
        st.subheader("Distribución por Ocupación del Cliente")
        if 'CustomerOccupation' in df_filtered.columns:
            occupation_counts = df_filtered['CustomerOccupation'].value_counts().reset_index()
            occupation_counts.columns = ['Occupation', 'Count']
            fig_cust_occupation = px.bar(occupation_counts, x='Count', y='Occupation', orientation='h',
                                         title='Número de Transacciones por Ocupación',
                                         labels={'Occupation': 'Ocupación del Cliente', 'Count': 'Número de Transacciones'},
                                         color='Occupation', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_cust_occupation, use_container_width=True)
        else:
            st.warning("Columna 'CustomerOccupation' no encontrada.")

    st.markdown("---")
    st.subheader("Patrones de Gasto por Demografía")
    col_cust_spend1, col_cust_spend2 = st.columns(2)
    with col_cust_spend1:
        if 'TransactionAmount' in df_filtered.columns and 'CustomerOccupation' in df_filtered.columns:
            avg_amount_by_occupation = df_filtered.groupby('CustomerOccupation')['TransactionAmount'].mean().reset_index()
            fig_avg_occ = px.bar(avg_amount_by_occupation, x='CustomerOccupation', y='TransactionAmount',
                                 title='Monto Promedio por Ocupación',
                                 labels={'CustomerOccupation': 'Ocupación', 'TransactionAmount': 'Monto Promedio ($)'},
                                 color='CustomerOccupation', color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig_avg_occ, use_container_width=True)
        else:
            st.warning("Columnas 'TransactionAmount' o 'CustomerOccupation' no encontradas.")
    with col_cust_spend2:
        if 'TransactionAmount' in df_filtered.columns and 'CustomerAge' in df_filtered.columns:
            fig_amount_age = px.scatter(df_filtered, x='CustomerAge', y='TransactionAmount',
                                        title='Monto de Transacción vs. Edad del Cliente',
                                        labels={'CustomerAge': 'Edad del Cliente', 'TransactionAmount': 'Importe ($)'},
                                        hover_data=['TransactionType', 'AccountBalance', 'CustomerOccupation'],
                                        color='CustomerOccupation', size='TransactionAmount', opacity=0.7)
            st.plotly_chart(fig_amount_age, use_container_width=True)
        else:
            st.warning("Columnas 'TransactionAmount' o 'CustomerAge' no encontradas.")

    st.markdown("---")
    st.subheader("Actividad y Saldo de Cuentas")
    col_cust_acc1, col_cust_acc2 = st.columns(2)
    with col_cust_acc1:
        st.subheader("Frecuencia de Transacciones por Cuenta")
        if 'PreviousTransactionDate' in df_filtered.columns and 'TransactionDate' in df_filtered.columns:
            df_filtered['TransactionFrequencyDays'] = (df_filtered['TransactionDate'] - df_filtered['PreviousTransactionDate']).dt.days
            df_filtered['TransactionFrequencyDays'] = df_filtered['TransactionFrequencyDays'].fillna(
                df_filtered['TransactionFrequencyDays'].median() # Rellenar para primeras transacciones
            )
            fig_freq_hist = px.histogram(df_filtered, x='TransactionFrequencyDays', nbins=50,
                                         title='Frecuencia de Transacciones (Días entre TXNs)',
                                         labels={'TransactionFrequencyDays': 'Días desde Transacción Previa', 'count': 'Número de Transacciones'})
            st.plotly_chart(fig_freq_hist, use_container_width=True)
        else:
            st.warning("Columnas de fecha de transacción (TransactionDate/PreviousTransactionDate) no encontradas.")
    with col_cust_acc2:
        st.subheader("Distribución de Saldo por Ocupación")
        if 'AccountBalance' in df_filtered.columns and 'CustomerOccupation' in df_filtered.columns:
            fig_balance_occupation = px.box(df_filtered, x='CustomerOccupation', y='AccountBalance',
                                            title='Distribución de Saldo de Cuenta por Ocupación',
                                            labels={'CustomerOccupation': 'Ocupación', 'AccountBalance': 'Saldo de Cuenta ($)'},
                                            color='CustomerOccupation', color_discrete_sequence=px.colors.qualitative.Dark2)
            st.plotly_chart(fig_balance_occupation, use_container_width=True)
        else:
            st.warning("Columnas 'AccountBalance' o 'CustomerOccupation' no encontradas.")

    st.markdown("---")
    st.subheader("Principales Clientes por Volumen Transaccionado")
    st.write("Identifica a tus clientes más activos y valiosos para estrategias de retención o beneficios.")
    if 'AccountID' in df_filtered.columns and 'TransactionAmount' in df_filtered.columns:
        top_n_accounts = st.slider("Mostrar Top N Cuentas:", 5, 100, 15, key='top_n_accounts_tab3')
        account_summary = df_filtered.groupby('AccountID').agg(
            Total_Transacciones=('TransactionID', 'count'),
            Volumen_Transaccionado=('TransactionAmount', 'sum'),
            Saldo_Actual=('AccountBalance', 'last') # Tomar el último saldo registrado
        ).nlargest(top_n_accounts, 'Volumen_Transaccionado').reset_index()

        fig_top_accounts = px.bar(account_summary, x='AccountID', y='Volumen_Transaccionado',
                                  title=f'Top {top_n_accounts} Cuentas por Volumen Transaccionado',
                                  labels={'AccountID': 'ID de Cuenta', 'Volumen_Transaccionado': 'Volumen Transaccionado ($)'},
                                  hover_data=['Total_Transacciones', 'Saldo_Actual'],
                                  color='Volumen_Transaccionado', color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_top_accounts, use_container_width=True)
    else:
        st.warning("Columnas 'AccountID' o 'TransactionAmount' no encontradas.")

with tab4:
    st.header("📈 Análisis Detallado de Transacciones")
    st.markdown("""
    Esta tabla interactiva te permite explorar los detalles de cada transacción individual, con capacidades de búsqueda,
    filtrado y ordenación. Es ideal para una inspección exhaustiva de cualquier transacción.
    """)

    st.subheader("Tabla de Transacciones Filtradas")
    st.dataframe(df_filtered, use_container_width=True, height=600)

    st.subheader("Exportar Datos Filtrados")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos filtrados como CSV",
        data=csv,
        file_name='transacciones_filtradas.csv',
        mime='text/csv',
    )


# Agrega un botón para limpiar la caché y forzar una recarga (útil en desarrollo)
st.markdown("---")
if st.button("Limpiar Caché de Datos y Reiniciar Dashboard"):
    st.cache_data.clear()
    st.rerun() # Reinicia la aplicación para reflejar la caché limpia