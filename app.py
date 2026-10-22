import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración de página
st.set_page_config(
    page_title="Customer Segmentation | K-Means",
    page_icon="🌸",
    layout="wide"
)

# Estilos CSS personalizados en tonos rosados / pastel / fucsia moderno
st.markdown("""
    <style>
    /* Fondo principal y tipografías */
    .main {
        background-color: #fffafc;
    }
    
    /* Títulos principales */
    h1 {
        color: #b83280 !important;
        font-weight: 800;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    h2, h3, h4 {
        color: #97266d !important;
        font-weight: 700;
    }

    /* Tarjeta superior con autoras */
    .header-box {
        background: linear-gradient(135deg, #fbb6ce 0%, #fed7e2 50%, #fff5f7 100%);
        border: 1px solid #f687b3;
        border-radius: 14px;
        padding: 20px 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 14px rgba(246, 135, 179, 0.25);
    }
    
    .autoras-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        color: #702459;
        margin-bottom: 4px;
    }
    
    .autoras-names {
        font-size: 1.25rem;
        font-weight: 800;
        color: #4a154b;
    }

    /* Pestañas personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #fce7f3;
        border-radius: 8px 8px 0px 0px;
        color: #9d174d;
        padding: 10px 18px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #db2777 !important;
        color: white !important;
    }

    /* Cuadro de resultado */
    .result-card {
        background: linear-gradient(135deg, #fff1f2 0%, #fdf2f8 100%);
        border: 2px solid #f472b6;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(244, 114, 182, 0.2);
    }

    /* Conclusiones */
    .conclusion-box {
        background-color: #ffffff;
        border-left: 6px solid #ec4899;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado visual con autoras
st.markdown("""
    <div class="header-box">
        <div class="autoras-title">Proyecto Final de Minería de Datos</div>
        <div class="autoras-names">✨ LEIDY VALENTINA AGUIRRE & ANGIE FERNANDA BENAVIDES</div>
        <p style="margin: 5px 0 0 0; color: #831843; font-size: 0.95rem;">
            Segmentación estratégica de clientes y proyección en tiempo real utilizando <b>K-Means Clustering</b>
        </p>
    </div>
""", unsafe_allow_html=True)

# 1. Carga de datos directa (con fallback automático)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
    try:
        return pd.read_csv(url)
    except Exception:
        np.random.seed(42)
        ingresos = np.concatenate([
            np.random.normal(25, 5, 40), np.random.normal(25, 5, 40),
            np.random.normal(55, 8, 40), np.random.normal(88, 10, 40),
            np.random.normal(88, 10, 40)
        ])
        gastos = np.concatenate([
            np.random.normal(20, 6, 40), np.random.normal(80, 7, 40),
            np.random.normal(50, 6, 40), np.random.normal(18, 6, 40),
            np.random.normal(82, 7, 40)
        ])
        return pd.DataFrame({
            'CustomerID': range(1, 201),
            'Age': np.random.randint(18, 70, 200),
            'Annual Income (k$)': np.clip(ingresos, 15, 140).astype(int),
            'Spending Score (1-100)': np.clip(gastos, 1, 99).astype(int)
        })

df = load_data()
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# 2. Escalado y modelo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_optimo = 5
kmeans = KMeans(n_clusters=k_optimo, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Mapeo descriptivo por cuadrantes
def etiquetar_perfil(ingreso, score):
    if ingreso > 70 and score > 60:
        return "Clientes de Alto Valor (VIP)", "Trato preferencial, membresías de lujo y preventas exclusivas.", "#be185d"
    elif ingreso > 70 and score < 40:
        return "Clientes con Potencial (Ahorradores)", "Incentivos en productos de alta gama y campañas enfocadas en valor/calidad.", "#0284c7"
    elif ingreso < 40 and score > 60:
        return "Clientes Derrochadores / Descuidados", "Promociones flash, financiamiento en cuotas y tendencias de temporada.", "#d97706"
    elif ingreso < 40 and score < 40:
        return "Clientes de Bajo Valor", "Ofertas tipo outlet y descuentos básicos con baja inversión en publicidad directa.", "#6b7280"
    else:
        return "Clientes Promedio", "Cupones recurrentes, tarjetas de puntos y ofertas periódicas.", "#8b5cf6"

# Paleta rosa armónica para los 5 clusters
pink_palette = ['#f43f5e', '#ec4899', '#d946ef', '#a855f7', '#fb7185']

# Pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "🌸 Clusters Generados", 
    "📐 Método del Codo", 
    "🎯 Clasificar Nuevo Cliente",
    "📝 Conclusiones y Negocio"
])

# ==========================================
# PESTAÑA 1: CLUSTERS GENERADOS & RESUMEN
# ==========================================
with tab1:
    st.markdown("### 📊 Segmentación de Clientes y Resumen Ejecutivo")
    
    col_chart, col_summary = st.columns([1.7, 1.1])
    
    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#fff8fa')

        for i in range(k_optimo):
            subset = df[df['Cluster'] == i]
            ax.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'], 
                       label=f"Cluster {i}", color=pink_palette[i], alpha=0.85, s=70, edgecolors='none')
            
        # Centroides destacados
        ax.scatter(centers[:, 0], centers[:, 1], s=260, c='#1e1b4b', marker='X', 
                   label='Centroides', edgecolors='white', linewidth=2, zorder=6)
        
        ax.set_title("Mapa de Agrupamiento K-Means (K=5)", fontsize=13, fontweight='bold', color='#831843', pad=12)
        ax.set_xlabel('Ingreso Anual (k$)', fontsize=10, fontweight='bold', color='#831843')
        ax.set_ylabel('Puntuación de Gasto (1-100)', fontsize=10, fontweight='bold', color='#831843')
        ax.grid(True, linestyle=':', alpha=0.5, color='#f472b6')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='#ffffff')
        plt.tight_layout()
        st.pyplot(fig)

    with col_summary:
        st.markdown("#### 📋 Resumen por Cluster")
        resumen = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)
        resumen['N° Clientes'] = df['Cluster'].value_counts()
        resumen.columns = ['Ingreso Prom. (k$)', 'Gasto Prom. (1-100)', 'Total Clientes']
        
        st.dataframe(resumen.style.background_gradient(cmap='PuRd'), use_container_width=True)
        
        st.markdown("""
        **Observaciones clave:**
        * Los clusters representan patrones claros de gasto vs. capacidad adquisitiva.
        * El centroide negro marca el promedio matemático exacto de cada grupo.
        """)

# ==========================================
# PESTAÑA 2: MÉTODO DEL CODO
# ==========================================
with tab2:
    st.markdown("### 📐 Método del Codo (Elbow Method)")
    st.write("Identificación matemática del número ideal de clusters evaluando la inercia (WCSS).")
    
    wcss = []
    rango_k = range(1, 11)
    for k in rango_k:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        
    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
    fig_elbow.patch.set_facecolor('#ffffff')
    ax_elbow.set_facecolor('#fff8fa')

    ax_elbow.plot(rango_k, wcss, marker='o', color='#db2777', linewidth=2.5, markersize=8, label='Inercia (WCSS)')
    
    # Marcador especial para K=5
    ax_elbow.axvline(x=5, color='#be185d', linestyle='--', linewidth=1.8, label='Codo Seleccionado (K=5)')
    ax_elbow.scatter([5], [wcss[4]], color='#9d174d', s=160, zorder=6, edgecolors='white', linewidth=2)
    
    ax_elbow.set_title("Curva de Decaimiento WCSS", fontsize=12, fontweight='bold', color='#831843')
    ax_elbow.set_xlabel('Número de Clusters (K)', fontweight='bold', color='#831843')
    ax_elbow.set_ylabel('Inercia (WCSS)', fontweight='bold', color='#831843')
    ax_elbow.set_xticks(list(rango_k))
    ax_elbow.grid(True, linestyle=':', alpha=0.5, color='#f472b6')
    ax_elbow.legend()
    plt.tight_layout()
    st.pyplot(fig_elbow)
    
    st.markdown("""
    > **Justificación Técnica:** En **K = 5** se evidencia el codo de desaceleración. Agregar más agrupaciones más allá de 5 solo genera divisiones artificiales con reducciones mínimas de varianza.
    """)

# ==========================================
# PESTAÑA 3: CLASIFICAR NUEVO CLIENTE
# ==========================================
with tab3:
    st.markdown("### 🎯 Clasificador en Tiempo Real de Nuevos Clientes")
    st.write("Ingresa los datos para categorizar instantáneamente al cliente según el modelo:")

    col_form, col_plot = st.columns([1, 1.2])

    with col_form:
        nuevo_ingreso = st.slider("💰 Ingreso Anual estimado (k$):", min_value=10, max_value=150, value=75, step=1)
        nuevo_score = st.slider("🛍️ Puntuación de Gasto (1-100):", min_value=1, max_value=100, value=78, step=1)

        # Predicción automática
        punto_scaled = scaler.transform([[nuevo_ingreso, nuevo_score]])
        cluster_pred = kmeans.predict(punto_scaled)[0]
        perfil, estrategia, color_perfil = etiquetar_perfil(nuevo_ingreso, nuevo_score)

        st.markdown(f"""
            <div class="result-card">
                <span style="font-size:0.85rem; font-weight:700; color:#9d174d; text-transform:uppercase;">Resultado de Segmentación</span>
                <h2 style="margin:5px 0; color:#831843;">Asignado a: Cluster {cluster_pred}</h2>
                <div style="font-size:1.15rem; font-weight:800; color:{color_perfil}; margin-bottom:8px;">{perfil}</div>
                <div style="font-size:0.95rem; color:#4a154b;"><b>Estrategia de Marketing:</b><br>{estrategia}</div>
            </div>
        """, unsafe_allow_html=True)

    with col_plot:
        fig_realtime, ax_rt = plt.subplots(figsize=(7, 5))
        fig_realtime.patch.set_facecolor('#ffffff')
        ax_rt.set_facecolor('#fff8fa')

        # Clientes existentes como fondo suave
        ax_rt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c='#f9a8d4', alpha=0.4, s=35, label='Población Actual')
        ax_rt.scatter(centers[:, 0], centers[:, 1], s=140, c='#831843', marker='X', label='Centroides')

        # Estrella del nuevo cliente
        ax_rt.scatter([nuevo_ingreso], [nuevo_score], s=350, c='#f59e0b', marker='*', 
                     edgecolors='#78350f', linewidth=2, label='Nuevo Cliente', zorder=10)
        
        ax_rt.set_title("Ubicación en el Espacio Multivariado", fontsize=11, fontweight='bold', color='#831843')
        ax_rt.set_xlabel('Ingreso Anual (k$)', color='#831843')
        ax_rt.set_ylabel('Puntuación de Gasto (1-100)', color='#831843')
        ax_rt.grid(True, linestyle=':', alpha=0.5, color='#f472b6')
        ax_rt.legend(loc='upper right')
        plt.tight_layout()
        st.pyplot(fig_realtime)

# ==========================================
# PESTAÑA 4: CONCLUSIONES Y NEGOCIO
# ==========================================
with tab4:
    st.markdown("### 📝 Conclusiones y Recomendaciones Estratégicas")
    
    st.markdown("""
    <div class="conclusion-box">
        <h4 style="margin-top:0;">1. Hallazgos del Algoritmo K-Means</h4>
        <p>El comportamiento de los clientes del centro comercial no es lineal ni depende únicamente del poder adquisitivo. K-Means identificó <b>5 arquetipos claros</b> que permiten abandonar las estrategias de marketing masivas y adoptar campañas hiperpersonalizadas.</p>
    </div>
    
    <div class="conclusion-box">
        <h4 style="margin-top:0;">2. Oportunidad de Crecimiento Prioritaria</h4>
        <p>El segmento de <b>Clientes con Potencial</b> (ingresos altos, pero bajo nivel de gasto) representa el mayor retorno de inversión posible. Poseen la liquidez financiera, por lo que la meta del mall es elevar su ticket promedio mediante ofertas aspiracionales, marcas exclusivas y beneficios de fidelidad.</p>
    </div>

    <div class="conclusion-box">
        <h4 style="margin-top:0;">3. Viabilidad del Despliegue en Tiempo Real</h4>
        <p>Gracias a la integración con Streamlit, el modelo permite a los gerentes comerciales o analistas ingresar nuevos prospectos y clasificarlos al instante, automatizando la toma de decisiones publicitarias de forma inmediata.</p>
    </div>
    """, unsafe_allow_html=True)
