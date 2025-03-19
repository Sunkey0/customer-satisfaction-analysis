import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import re
from nltk.corpus import stopwords
import nltk
from textblob import TextBlob

# Configuración inicial
nltk.download('stopwords')
st.set_page_config(page_title="Análisis de Satisfacción", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("customer_satisfaction_large.csv")

df = load_data()

# Función para análisis de sentimiento
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.2:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

# Crear columna de sentimiento si no existe
if 'sentiment' not in df.columns:
    df['sentiment'] = df['feedback_text'].apply(analyze_sentiment)

# Preprocesamiento de texto
stop_words = set(stopwords.words('english'))
spanish_stopwords = set(stopwords.words('spanish'))
stop_words.update(spanish_stopwords)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return ' '.join([word for word in text.split() if word not in stop_words])

df['feedback_clean'] = df['feedback_text'].apply(clean_text)

# Sidebar con filtros
st.sidebar.header("Filtros")
selected_region = st.sidebar.multiselect('Región', df['region'].unique())
selected_category = st.sidebar.multiselect('Categoría de Producto', df['product_category'].unique())
selected_sentiment = st.sidebar.multiselect('Sentimiento', df['sentiment'].unique())

# Aplicar filtros
filtered_df = df.copy()
if selected_region:
    filtered_df = filtered_df[filtered_df['region'].isin(selected_region)]
if selected_category:
    filtered_df = filtered_df[filtered_df['product_category'].isin(selected_category)]
if selected_sentiment:
    filtered_df = filtered_df[filtered_df['sentiment'].isin(selected_sentiment)]

# Main content
st.title("Análisis de Satisfacción del Cliente")

# Pestañas
tab1, tab2 = st.tabs(["Análisis y Visualización", "Detalles Técnicos"])

with tab1:
    # Métricas clave
    col1, col2, col3 = st.columns(3)
    col1.metric("Satisfacción Promedio", f"{filtered_df['satisfaction_score'].mean():.1f}/10")
    col2.metric("Tasa de Retención", f"{(filtered_df['retention_status'] == 'Retained').mean()*100:.1f}%")
    col3.metric("Tickets Soporte Promedio", f"{filtered_df['support_tickets'].mean():.1f}")

    # Gráficos principales
    st.subheader("Distribución de Satisfacción")
    fig1 = px.histogram(filtered_df, x='satisfaction_score', nbins=10,
                       title='Distribución de Satisfacción')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Relación Satisfacción vs Frecuencia de Compra")
    fig2 = px.scatter(filtered_df, x='satisfaction_score', y='purchase_frequency',
                     color='retention_status',
                     title='Relación Satisfacción vs Frecuencia de Compra')
    st.plotly_chart(fig2, use_container_width=True)

    # Word Clouds
    st.subheader("Análisis de Texto - Word Clouds")
    sentiment_option = st.selectbox("Selecciona un sentimiento para el Word Cloud", 
                                    ['Positive', 'Neutral', 'Negative'])

    if sentiment_option:
        text = ' '.join(filtered_df[filtered_df['sentiment'] == sentiment_option]['feedback_clean'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Palabras Clave - Sentimiento {sentiment_option}')
        st.pyplot(plt)

    # Sistema de Alertas Tempranas
    st.subheader("Sistema de Alertas Tempranas")
    risk_threshold = st.slider('Umbral de Riesgo', 0.0, 10.0, 5.0)

    def flag_at_risk_customers(df):
        conditions = (
            (df['satisfaction_score'] < 6) |
            (df['support_tickets'] >= 3) |
            (df['purchase_frequency'] < 2)
        )
        
        risk_customers = df[conditions].sort_values('satisfaction_score')[
            ['customer_id', 'satisfaction_score', 'support_tickets', 'purchase_frequency']
        ]
        
        risk_customers['risk_score'] = (
            0.5 * (10 - df['satisfaction_score']) +
            0.3 * df['support_tickets'] +
            0.2 * (10 - df['purchase_frequency'])
        )
        
        return risk_customers

    risk_df = flag_at_risk_customers(filtered_df)
    risk_df = risk_df[risk_df['risk_score'] > risk_threshold]

    st.dataframe(risk_df.sort_values('risk_score', ascending=False))

    # Modelo Predictivo de Abandono
    st.subheader("Predicción de Abandono (Churn)")

    # Entrenar modelo (solo una vez)
    if 'model' not in st.session_state:
        df['churn'] = df['retention_status'].apply(lambda x: 1 if x == 'Churned' else 0)
        features = ['satisfaction_score', 'purchase_frequency', 'support_tickets', 'age']
        X = df[features]
        y = df['churn']
        
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

    # Mostrar métricas del modelo
    y_pred = st.session_state.model.predict(st.session_state.X_test)
    st.write("Precisión del Modelo (AUC-ROC):", roc_auc_score(st.session_state.y_test, y_pred))

    # Predicción en tiempo real
    st.subheader("Simulador de Predicción de Abandono")
    satisfaction = st.slider("Puntuación de Satisfacción", 1, 10, 5)
    purchase_freq = st.slider("Frecuencia de Compra (mensual)", 1, 10, 3)
    support_tickets = st.slider("Tickets de Soporte", 0, 5, 1)
    age = st.slider("Edad del Cliente", 18, 70, 30)

    if st.button("Predecir Riesgo de Abandono"):
        prediction = st.session_state.model.predict([[satisfaction, purchase_freq, support_tickets, age]])[0]
        st.write(f"**Riesgo de Abandono:** {'Alto' if prediction == 1 else 'Bajo'}")

with tab2:
    st.header("Detalles Técnicos del Dashboard")

    st.subheader("1. Análisis de Sentimientos")
    st.markdown("""
    **Técnica utilizada**: Análisis de sentimientos con TextBlob.
    - **Polaridad**: Mide la positividad o negatividad de un texto en un rango de [-1, 1].
    - **Clasificación**:
      - **Positivo**: Polaridad > 0.2
      - **Neutral**: Polaridad entre -0.2 y 0.2
      - **Negativo**: Polaridad < -0.2
    """)
    st.latex(r"""
    \text{Polaridad} = \frac{\sum (\text{Polaridad de cada palabra})}{\text{Número de palabras}}
    """)

    st.subheader("2. Modelo Predictivo de Abandono (Churn)")
    st.markdown("""
    **Técnica utilizada**: Random Forest Classifier.
    - **Características**:
      - `satisfaction_score`: Puntuación de satisfacción del cliente.
      - `purchase_frequency`: Frecuencia de compra mensual.
      - `support_tickets`: Número de tickets de soporte abiertos.
      - `age`: Edad del cliente.
    - **Métrica de Evaluación**: AUC-ROC (Área bajo la curva ROC).
    - **Balanceo de Datos**: SMOTE (Synthetic Minority Over-sampling Technique).
    """)
    st.latex(r"""
    \text{Churn} = f(\text{satisfaction_score}, \text{purchase_frequency}, \text{support_tickets}, \text{age})
    """)

    st.subheader("3. Sistema de Alertas Tempranas")
    st.markdown("""
    **Técnica utilizada**: Puntaje de riesgo basado en reglas.
    - **Factores**:
      - Puntuación de satisfacción (< 6).
      - Tickets de soporte (≥ 3).
      - Frecuencia de compra (< 2).
    """)
    st.latex(r"""
    \text{Risk Score} = 0.5 \times (10 - \text{satisfaction_score}) + 0.3 \times \text{support_tickets} + 0.2 \times (10 - \text{purchase_frequency})
    """)

    st.subheader("4. Word Clouds")
    st.markdown("""
    **Técnica utilizada**: Nube de palabras.
    - **Proceso**:
      1. Limpieza de texto (eliminación de stopwords y puntuación).
      2. Frecuencia de palabras.
      3. Visualización proporcional al peso de cada palabra.
    """)
