import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import variation

# -----------------------------
# Funciones de desigualdad
# -----------------------------

def gini(array):
    """Índice de Gini"""
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
    """Índice de Theil (T)"""
    array = np.array(array)
    array = array[array > 0]
    mean = np.mean(array)
    theil_index = np.sum((array / mean) * np.log(array / mean)) / len(array)
    return theil_index

# -----------------------------
# Cargar datos
# -----------------------------
st.set_page_config(layout="wide", page_title="Dashboard Estadísticas 14 Programas")
st.title("📊 Dashboard de Estadísticas de los 14 Programas Estatales")

uploaded_file = st.file_uploader("🔼 Cargar archivo Excel procesado", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.success("Archivo cargado correctamente.")
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    programas = ["TODOS"] + sorted(df["PROGRAMA"].dropna().unique().tolist())
    programa_seleccionado = st.selectbox("Selecciona un programa", programas)

    if programa_seleccionado != "TODOS":
        df_prog = df[df["PROGRAMA"] == programa_seleccionado]
    else:
        df_prog = df.copy()

    st.subheader("📈 Estadísticas por Régimen Laboral")
    stats_regimen = df_prog.groupby("Regimen")["Remuneracion"].agg([
        ("Cantidad", "count"),
        ("Promedio", "mean"),
        ("Mediana", "median"),
        ("Máximo", "max"),
        ("Mínimo", "min"),
        ("Coeficiente de variación", variation)
    ]).round(2)
    st.dataframe(stats_regimen)

    st.subheader("📊 Estadísticas por Categoría Laboral")
    stats_categoria = df_prog.groupby("CATEGORIA_LABORAL")["Remuneracion"].agg([
        ("Cantidad", "count"),
        ("Promedio", "mean"),
        ("Mediana", "median"),
        ("Máximo", "max"),
        ("Mínimo", "min"),
        ("Coeficiente de variación", variation)
    ]).round(2)
    st.dataframe(stats_categoria)

    st.subheader("📉 Indicadores de Desigualdad Global")
    gini_val = gini(df_prog["Remuneracion"])
    theil_val = theil(df_prog["Remuneracion"])

    col1, col2 = st.columns(2)
    col1.metric("Índice de Gini", round(gini_val, 4))
    col2.metric("Índice de Theil", round(theil_val, 4))

    st.subheader("📊 Desigualdad: Theil entre/intra por Sexo")
    theil_total = theil(df_prog["Remuneracion"])
    grupos_sexo = df_prog.groupby("Sexo")
    theil_intra = grupos_sexo.apply(lambda g: len(g)/len(df_prog) * theil(g["Remuneracion"])).sum()
    theil_between = theil_total - theil_intra

    st.write(f"**Theil Total:** {theil_total:.4f}")
    st.write(f"**Intra grupos (sexo):** {theil_intra:.4f}")
    st.write(f"**Entre grupos (sexo):** {theil_between:.4f}")

    st.subheader("📊 Desigualdad: Theil entre/intra por Categoría Laboral")
    grupos_categoria = df_prog.groupby("CATEGORIA_LABORAL")
    theil_intra_cat = grupos_categoria.apply(lambda g: len(g)/len(df_prog) * theil(g["Remuneracion"])).sum()
    theil_between_cat = theil_total - theil_intra_cat

    st.write(f"**Intra grupos (categoría):** {theil_intra_cat:.4f}")
    st.write(f"**Entre grupos (categoría):** {theil_between_cat:.4f}")

    st.subheader("📉 Histograma de Remuneraciones")
    fig = px.histogram(df_prog, x="Remuneracion", nbins=30, title="Distribución de Remuneraciones")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Por favor, sube un archivo Excel con la base de datos procesada.")
