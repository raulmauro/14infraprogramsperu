import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from scipy.stats import variation

# Funciones de desigualdad
def gini(array):
    """Calcula el índice de Gini de un array de ingresos"""
    array = np.array(array)
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # hacer no-negativos
    array += 0.0000001  # evitar división por cero
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def theil(array):
    """Calcula el índice de Theil (T) para una distribución de ingresos"""
    array = np.array(array)
    array = array[array > 0]
    mean = np.mean(array)
    theil_index = np.sum((array / mean) * np.log(array / mean)) / len(array)
    return theil_index

# Cargar el archivo
@st.cache_data
def load_data():
    return pd.read_excel("estadisticas_programas_con_total.xlsx", sheet_name=None)

# Cargar hojas
data = load_data()
programas = list(data.keys())

st.title("Análisis de Remuneraciones en Programas Estatales del Perú")
st.markdown("Datos procesados a partir del portal de transparencia del Estado. Incluye análisis por régimen laboral, categoría y medidas de desigualdad.")

# Selector de programa
selected_program = st.selectbox("Selecciona un programa estatal", programas)

df = data[selected_program]

st.subheader(f"Resumen para: {selected_program}")

# Sección: Estadísticas por régimen
st.markdown("### 📋 Estadísticas por Régimen Laboral")
st.dataframe(df[['Regimen', 'n', 'promedio', 'mediana', 'min', 'max', 'coef_variacion']])

# Sección: Estadísticas por categoría laboral
st.markdown("### 🧾 Estadísticas por Categoría Laboral")
st.dataframe(df[['Categoria_laboral', 'n_cat', 'promedio_cat', 'mediana_cat', 'min_cat', 'max_cat', 'coef_var_cat']])

# Sección: Distribución de salarios
st.markdown("### 📊 Distribución de Remuneraciones")
fig = px.histogram(df, x="Remuneracion", nbins=50, title="Distribución de salarios mensuales")
st.plotly_chart(fig)

# Sección: Índices de desigualdad
st.markdown("### ⚖️ Índices de Desigualdad")

if 'Gini' in df.columns:
    gini_val = df['Gini'].dropna().values[0]
    st.metric("Índice de Gini", f"{gini_val:.3f}")

if 'Theil_total' in df.columns:
    theil_total = df['Theil_total'].dropna().values[0]
    theil_btw_sex = df['Theil_entre_sexos'].dropna().values[0]
    theil_intra_sex = df['Theil_intra_sexos'].dropna().values[0]
    st.markdown(f"**Theil Total:** {theil_total:.3f}  \n"
                f"**Entre sexos:** {theil_btw_sex:.3f}  \n"
                f"**Intra sexos:** {theil_intra_sex:.3f}")

if 'Theil_entre_categoria' in df.columns:
    theil_btw_cat = df['Theil_entre_categoria'].dropna().values[0]
    theil_intra_cat = df['Theil_intra_categoria'].dropna().values[0]
    st.markdown(f"**Entre categorías:** {theil_btw_cat:.3f}  \n"
                f"**Intra categorías:** {theil_intra_cat:.3f}")

st.markdown("---")
st.caption("Desarrollado por Raul con apoyo de análisis automatizado y datos abiertos del Estado peruano.")
