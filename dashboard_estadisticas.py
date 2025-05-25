import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import variation

# Configuración inicial de la página
st.set_page_config(layout="wide")
st.title("Análisis de Remuneraciones en Programas Estatales del Perú")
st.markdown("Datos procesados a partir del portal de transparencia del Estado")

# Funciones de desigualdad
def gini(array):
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
    array = np.array(array)
    array = array[array > 0]
    mean = np.mean(array)
    theil_index = np.sum((array / mean) * np.log(array / mean)) / len(array)
    return theil_index

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_excel("estadisticas_programas_con_total.xlsx", sheet_name=None)

data = load_data()

# Obtener lista de programas disponibles (excluyendo 'TOTAL' si es necesario)
programas = [p for p in data['Resumen por Regimen']['programa'].unique() if p != 'TOTAL']

# Selector de programa
selected_program = st.selectbox("Selecciona un programa estatal", programas)

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📋 Por Régimen", "🧾 Por Categoría", "⚖️ Desigualdad"])

with tab1:
    # Datos por régimen
    df_regimen = data['Resumen por Regimen'][data['Resumen por Regimen']['programa'] == selected_program]
    df_regimen = df_regimen[['regimen', 'n', 'media', 'mediana', 'min', 'max', 'coef_var']]
    
    # Formatear números
    for col in ['media', 'mediana', 'min', 'max']:
        df_regimen[col] = df_regimen[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) else "")
    
    st.dataframe(df_regimen, hide_index=True)
    
    # Gráfico de distribución por régimen
    fig_reg = px.bar(df_regimen, x='regimen', y='media', 
                     title=f"Media de remuneraciones por régimen - {selected_program}")
    st.plotly_chart(fig_reg, use_container_width=True)

with tab2:
    # Datos por categoría
    df_categoria = data['Resumen por Categoria'][data['Resumen por Categoria']['programa'] == selected_program]
    df_categoria = df_categoria[['categoria_laboral', 'n', 'media', 'mediana', 'min', 'max', 'coef_var']]
    
    # Formatear números
    for col in ['media', 'mediana', 'min', 'max']:
        df_categoria[col] = df_categoria[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) else "")
    
    st.dataframe(df_categoria, hide_index=True)
    
    # Gráfico de distribución por categoría
    fig_cat = px.bar(df_categoria, x='categoria_laboral', y='media',
                     title=f"Media de remuneraciones por categoría - {selected_program}")
    st.plotly_chart(fig_cat, use_container_width=True)

with tab3:
    # Índices de desigualdad
    col1, col2 = st.columns(2)
    
    with col1:
        # Índice de Gini
        gini_val = data['Indice Gini'][data['Indice Gini']['programa'] == selected_program]['indice_gini'].values[0]
        st.metric("Índice de Gini", f"{gini_val:.3f}")
        
        # Theil por sexo
        theil_sexo = data['Theil por Sexo'][data['Theil por Sexo']['programa'] == selected_program]
        st.markdown("**Theil por Sexo**")
        st.markdown(f"- Total: {theil_sexo['theil_total_sexo'].values[0]:.3f}")
        st.markdown(f"- Entre sexos: {theil_sexo['theil_between_sexo'].values[0]:.3f}")
        st.markdown(f"- Intra sexos: {theil_sexo['theil_within_sexo'].values[0]:.3f}")
    
    with col2:
        # Theil por categoría
        theil_cat = data['Theil por Categoria'][data['Theil por Categoria']['programa'] == selected_program]
        st.markdown("**Theil por Categoría**")
        st.markdown(f"- Total: {theil_cat['theil_total_categoria'].values[0]:.3f}")
        st.markdown(f"- Entre categorías: {theil_cat['theil_between_categoria'].values[0]:.3f}")
        st.markdown(f"- Intra categorías: {theil_cat['theil_within_categoria'].values[0]:.3f}")

# Nota al pie
st.markdown("---")
st.caption("Desarrollado por Raúl Mauro con datos abiertos del Estado peruano | Actualizado: 2023")
