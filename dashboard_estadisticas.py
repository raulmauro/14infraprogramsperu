import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import variation

# Configuración inicial de la página
st.set_page_config(layout="wide")
st.title("📊 Análisis de Remuneraciones en Programas Estatales del Perú")
st.markdown("**Datos procesados a partir del portal de transparencia del Estado**")

# Funciones de desigualdad (se mantienen igual)
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

# Cargar datos (se mantiene igual)
@st.cache_data
def load_data():
    return pd.read_excel("estadisticas_programas_con_total.xlsx", sheet_name=None)

data = load_data()

# Obtener lista de programas (se mantiene igual)
programas = [p for p in data['Resumen por Regimen']['programa'].unique() if p != 'TOTAL']
selected_program = st.selectbox("Selecciona un programa estatal", programas, index=0)

# Crear pestañas (se mantiene igual)
tab1, tab2, tab3 = st.tabs(["📋 Por Régimen", "🧾 Por Categoría", "⚖️ Desigualdad"])

# Función para crear gráfico mejorado
def crear_grafico_mejorado(df, x_col, y_col, title):
    fig = go.Figure()
    
    for regimen in df[x_col].unique():
        subset = df[df[x_col] == regimen]
        
        # Solo agregar datos si hay valores válidos
        if not subset.empty and pd.notnull(subset['min'].iloc[0]) and pd.notnull(subset['max'].iloc[0]):
            fig.add_trace(go.Box(
                x=[regimen]*3,
                y=[subset['min'].iloc[0], subset['media'].iloc[0], subset['max'].iloc[0]],
                name=regimen,
                boxpoints='all',
                jitter=0.3,
                pointpos=0,
                marker=dict(
                    color='rgb(7,40,89)',
                    size=8,
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5
                    )
                ),
                line=dict(color='rgb(8,48,107)'),
                whiskerwidth=0.2,
                fillcolor='rgba(255,255,255,0)',
                hoveron='points',
                hoverinfo='y'
            ))
    
    # Personalizar diseño
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Remuneración (S/)",
        showlegend=False,
        height=600,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        boxmode='group'
    )
    
    # Añadir anotaciones para los valores
    for regimen in df[x_col].unique():
        subset = df[df[x_col] == regimen]
        if not subset.empty:
            # Media
            fig.add_annotation(
                x=regimen,
                y=subset['media'].iloc[0],
                text=f"Media: S/ {subset['media'].iloc[0]:,.1f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
            # Mínimo
            fig.add_annotation(
                x=regimen,
                y=subset['min'].iloc[0],
                text=f"Mín: S/ {subset['min'].iloc[0]:,.1f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40
            )
            # Máximo
            fig.add_annotation(
                x=regimen,
                y=subset['max'].iloc[0],
                text=f"Máx: S/ {subset['max'].iloc[0]:,.1f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
    
    return fig

with tab1:
    st.subheader(f"Análisis por Régimen Laboral - {selected_program}")
    
    # Datos por régimen (se mantiene igual)
    df_regimen = data['Resumen por Regimen'][data['Resumen por Regimen']['programa'] == selected_program]
    df_regimen = df_regimen[['regimen', 'n', 'media', 'mediana', 'min', 'max', 'coef_var']]
    
    # Formatear números (se mantiene igual)
    for col in ['media', 'mediana', 'min', 'max']:
        df_regimen[col] = df_regimen[col].apply(lambda x: f"S/ {x:,.1f}" if pd.notnull(x) else "")
    df_regimen['coef_var'] = df_regimen['coef_var'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")
    
    st.dataframe(df_regimen.style.format({'n': '{:,.0f}'}), 
                height=(len(df_regimen) * 35 + 38),
                use_container_width=True)
    
    # Nuevo gráfico mejorado
    if not df_regimen.empty:
        fig_reg = crear_grafico_mejorado(
            df=data['Resumen por Regimen'][data['Resumen por Regimen']['programa'] == selected_program],
            x_col='regimen',
            y_col='media',
            title=f"Distribución de Remuneraciones por Régimen - {selected_program}"
        )
        st.plotly_chart(fig_reg, use_container_width=True)

with tab2:
    st.subheader(f"Análisis por Categoría Laboral - {selected_program}")
    
    # Datos por categoría (se mantiene igual)
    df_categoria = data['Resumen por Categoria'][data['Resumen por Categoria']['programa'] == selected_program]
    df_categoria = df_categoria[['categoria_laboral', 'n', 'media', 'mediana', 'min', 'max', 'coef_var']]
    
    # Formatear números (se mantiene igual)
    for col in ['media', 'mediana', 'min', 'max']:
        df_categoria[col] = df_categoria[col].apply(lambda x: f"S/ {x:,.1f}" if pd.notnull(x) else "")
    df_categoria['coef_var'] = df_categoria['coef_var'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")
    
    st.dataframe(df_categoria.style.format({'n': '{:,.0f}'}), 
                height=(len(df_categoria) * 35 + 38),
                use_container_width=True)
    
    # Nuevo gráfico mejorado
    if not df_categoria.empty:
        fig_cat = crear_grafico_mejorado(
            df=data['Resumen por Categoria'][data['Resumen por Categoria']['programa'] == selected_program],
            x_col='categoria_laboral',
            y_col='media',
            title=f"Distribución de Remuneraciones por Categoría - {selected_program}"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

with tab3:
    # (Se mantiene igual que en la versión anterior)
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
        
        st.markdown("---")
        st.markdown("**Nota:** Los índices de Theil miden la desigualdad, donde:")
        st.markdown("- **Entre grupos:** Desigualdad entre diferentes categorías/sexos")
        st.markdown("- **Intra grupos:** Desigualdad dentro de la misma categoría/sexo")

# Nota al pie (se mantiene igual)
st.markdown("---")
st.caption("© 2023 - Desarrollado por Raúl Mauro con datos de Transparencia del Estado peruano | Versión 1.2")
