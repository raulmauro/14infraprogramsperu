import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import variation

# Configuración inicial de la página
st.set_page_config(layout="wide")
st.title("📊 Análisis de Remuneraciones en Programas Estatales del Perú")
st.markdown("**Datos procesados a partir del portal de transparencia del Estado**")

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
selected_program = st.selectbox("Selecciona un programa estatal", programas, index=0)

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📋 Por Régimen", "🧾 Por Categoría", "⚖️ Desigualdad"])

with tab1:
    st.subheader(f"Análisis por Régimen Laboral - {selected_program}")
    
    # Datos por régimen
    df_regimen = data['Resumen por Regimen'][data['Resumen por Regimen']['programa'] == selected_program]
    df_regimen = df_regimen[['regimen', 'n', 'media', 'mediana', 'min', 'max', 'coef_var']]
    
    # Formatear números
    for col in ['media', 'mediana', 'min', 'max']:
        df_regimen[col] = df_regimen[col].apply(lambda x: f"S/ {x:,.1f}" if pd.notnull(x) else "")
    
    # Formatear coeficiente de variación como porcentaje
    df_regimen['coef_var'] = df_regimen['coef_var'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")
    
    st.dataframe(df_regimen.style.format({'n': '{:,.0f}'}), 
                height=(len(df_regimen) * 35 + 38),
                use_container_width=True)
    
    # Gráfico de distribución por régimen con min, media y max
    if not df_regimen.empty:
        fig_reg = px.bar(df_regimen, x='regimen', y='media', 
                         title=f"Distribución de Remuneraciones por Régimen - {selected_program}",
                         labels={'media': 'Remuneración (S/)', 'regimen': 'Régimen Laboral'},
                         text='media')
        
        # Añadir líneas para mínimo y máximo
        fig_reg.add_scatter(x=df_regimen['regimen'], y=df_regimen['min'], 
                           mode='markers+text', name='Mínimo',
                           marker=dict(color='red', size=10),
                           text=df_regimen['min'], textposition='top center')
        
        fig_reg.add_scatter(x=df_regimen['regimen'], y=df_regimen['max'], 
                           mode='markers+text', name='Máximo',
                           marker=dict(color='green', size=10),
                           text=df_regimen['max'], textposition='top center')
        
        fig_reg.update_layout(showlegend=True)
        st.plotly_chart(fig_reg, use_container_width=True)

with tab2:
    st.subheader(f"Análisis por Categoría Laboral - {selected_program}")
    
    # Datos por categoría
    df_categoria = data['Resumen por Categoria'][data['Resumen por Categoria']['programa'] == selected_program]
    df_categoria = df_categoria[['categoria_laboral', 'n', 'media', 'mediana', 'min', 'max', 'coef_var']]
    
    # Formatear números
    for col in ['media', 'mediana', 'min', 'max']:
        df_categoria[col] = df_categoria[col].apply(lambda x: f"S/ {x:,.1f}" if pd.notnull(x) else "")
    
    # Formatear coeficiente de variación como porcentaje
    df_categoria['coef_var'] = df_categoria['coef_var'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")
    
    st.dataframe(df_categoria.style.format({'n': '{:,.0f}'}), 
                height=(len(df_categoria) * 35 + 38),
                use_container_width=True)
    
    # Gráfico de distribución por categoría con min, media y max
    if not df_categoria.empty:
        fig_cat = px.bar(df_categoria, x='categoria_laboral', y='media',
                         title=f"Distribución de Remuneraciones por Categoría - {selected_program}",
                         labels={'media': 'Remuneración (S/)', 'categoria_laboral': 'Categoría Laboral'},
                         text='media')
        
        fig_cat.add_scatter(x=df_categoria['categoria_laboral'], y=df_categoria['min'], 
                           mode='markers+text', name='Mínimo',
                           marker=dict(color='red', size=10),
                           text=df_categoria['min'], textposition='top center')
        
        fig_cat.add_scatter(x=df_categoria['categoria_laboral'], y=df_categoria['max'], 
                           mode='markers+text', name='Máximo',
                           marker=dict(color='green', size=10),
                           text=df_categoria['max'], textposition='top center')
        
        fig_cat.update_layout(showlegend=True)
        st.plotly_chart(fig_cat, use_container_width=True)

with tab3:
    st.subheader(f"Índices de Desigualdad - {selected_program}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Índice de Gini")
        gini_val = data['Indice Gini'][data['Indice Gini']['programa'] == selected_program]['indice_gini'].values[0]
        st.metric("", f"{gini_val*100:.1f}%",
                 help="Mide la desigualdad en la distribución de ingresos (0 = perfecta igualdad, 100 = máxima desigualdad)")
        
        st.markdown("### 👥 Theil por Sexo")
        theil_sexo = data['Theil por Sexo'][data['Theil por Sexo']['programa'] == selected_program]
        st.markdown(f"- **Total:** {theil_sexo['theil_total_sexo'].values[0]:.3f}")
        st.markdown(f"- **Entre sexos:** {theil_sexo['theil_between_sexo'].values[0]:.3f}")
        st.markdown(f"- **Intra sexos:** {theil_sexo['theil_within_sexo'].values[0]:.3f}")
    
    with col2:
        st.markdown("### 🏷️ Theil por Categoría")
        theil_cat = data['Theil por Categoria'][data['Theil por Categoria']['programa'] == selected_program]
        st.markdown(f"- **Total:** {theil_cat['theil_total_categoria'].values[0]:.3f}")
        st.markdown(f"- **Entre categorías:** {theil_cat['theil_between_categoria'].values[0]:.3f}")
        st.markdown(f"- **Intra categorías:** {theil_cat['theil_within_categoria'].values[0]:.3f}")
        
        # Explicación de los índices
        st.markdown("---")
        st.markdown("**Nota:** Los índices de Theil miden la desigualdad, donde:")
        st.markdown("- **Entre grupos:** Desigualdad entre diferentes categorías/sexos")
        st.markdown("- **Intra grupos:** Desigualdad dentro de la misma categoría/sexo")

# Nota al pie
st.markdown("---")
st.caption("© 2023 - Desarrollado por Raúl Mauro con datos de Transparencia del Estado peruano | Versión 1.1")
