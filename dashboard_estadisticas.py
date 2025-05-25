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
        array -= np.amin(array)
    array += 0.0000001
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

# Cargar archivo Excel
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

# Estadísticas por Régimen Laboral
st.markdown("### 📋 Estadísticas por Régimen Laboral")
cols_regimen = ['Regimen', 'n', 'promedio', 'mediana', 'min', 'max', 'coef_variacion']
if all(col in df.columns for col in cols_regimen):
    df_reg = df[cols_regimen].copy()
    df_reg['promedio'] = df_reg['promedio'].apply(lambda x: f"{x:,.1f}")
    df_reg['mediana'] = df_reg['mediana'].apply(lambda x: f"{x:,.1f}")
    df_reg['min'] = df_reg['min'].apply(lambda x: f"{x:,.1f}")
    df_reg['max'] = df_reg['max'].apply(lambda x: f"{x:,.1f}")
    df_reg['coef_variacion'] = df_reg['coef_variacion'].apply(lambda x: f"{x * 100:.1f}%")
    st.dataframe(df_reg)
else:
    st.warning("⚠️ Las columnas para 'Estadísticas por Régimen Laboral' no están disponibles.")

# Estadísticas por Categoría Laboral
st.markdown("### 🧾 Estadísticas por Categoría Laboral")
cols_categoria = ['Categoria_laboral', 'n_cat', 'promedio_cat', 'mediana_cat', 'min_cat', 'max_cat', 'coef_var_cat']
if all(col in df.columns for col in cols_categoria):
    df_cat = df[cols_categoria].copy()
    df_cat['promedio_cat'] = df_cat['promedio_cat'].apply(lambda x: f"{x:,.1f}")
    df_cat['mediana_cat'] = df_cat['mediana_cat'].apply(lambda x: f"{x:,.1f}")
    df_cat['min_cat'] = df_cat['min_cat'].apply(lambda x: f"{x:,.1f}")
    df_cat['max_cat'] = df_cat['max_cat'].apply(lambda x: f"{x:,.1f}")
    df_cat['coef_var_cat'] = df_cat['coef_var_cat'].apply(lambda x: f"{x * 100:.1f}%")
    st.dataframe(df_cat)
else:
    st.warning("⚠️ Las columnas para 'Estadísticas por Categoría Laboral' no están disponibles.")

# Distribución de salarios
st.markdown("### 📊 Distribución de Remuneraciones")
if 'Remuneracion' in df.columns:
    fig = px.histogram(df, x="Remuneracion", nbins=50, title="Distribución de salarios mensuales")
    st.plotly_chart(fig)
else:
    st.warning("⚠️ No se encuentra la columna 'Remuneracion' para graficar.")

# Índices de Desigualdad
st.markdown("### ⚖️ Índices de Desigualdad")
if 'Gini' in df.columns:
    gini_val = df['Gini'].dropna().values[0]
    st.metric("Índice de Gini", f"{gini_val * 100:.1f}%")

if 'Theil_total' in df.columns:
    theil_total = df['Theil_total'].dropna().values[0]
    theil_btw_sex = df['Theil_entre_sexos'].dropna().values[0]
    theil_intra_sex = df['Theil_intra_sexos'].dropna().values[0]
    st.markdown(f"**Theil Total:** {theil_total * 100:.1f}%  \n"
                f"**Entre sexos:** {theil_btw_sex * 100:.1f}%  \n"
                f"**Intra sexos:** {theil_intra_sex * 100:.1f}%")

if 'Theil_entre_categoria' in df.columns:
    theil_btw_cat = df['Theil_entre_categoria'].dropna().values[0]
    theil_intra_cat = df['Theil_intra_categoria'].dropna().values[0]
    st.markdown(f"**Entre categorías:** {theil_btw_cat * 100:.1f}%  \n"
                f"**Intra categorías:** {theil_intra_cat * 100:.1f}%")

st.markdown("---")
st.caption("Desarrollado por Raul con apoyo de análisis automatizado y datos abiertos del Estado peruano.")
