import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import variation

# Configuración inicial de la página
st.set_page_config(layout="wide")
st.title("📊 Comparativa de Remuneraciones: Programas vs ANIN")
st.markdown("**Análisis comparativo de remuneraciones de trabajadores de programas en extinción - Marzo 2025**")

# Color distintivo para ANIN
COLOR_ANIN = '#E63946'  # Rojo institucional
COLOR_OTROS = '#457B9D'  # Azul para contrastar
COLOR_TOTAL = '#A8DADC'  # Color neutral para total

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

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_excel("estadisticas_programas_con_total.xlsx", sheet_name=None)

data = load_data()

# Obtener lista de programas (excluyendo TOTAL y ANIN)
programas = [p for p in data['Resumen por Regimen']['programa'].unique() if p not in ['TOTAL', 'ANIN']]
programas.insert(0, 'ANIN')  # ANIN siempre como primera opción

selected_program = st.selectbox("Selecciona un programa para comparar con ANIN", programas, index=0)

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📋 Comparativa por Régimen", "🧾 Comparativa por Categoría", "⚖️ Análisis de Desigualdad"])

# Función para crear gráfico comparativo
def crear_grafico_comparativo(df, x_col, y_col, title, programa_seleccionado):
    # Filtrar datos para el programa seleccionado y ANIN
    df_selected = df[df['programa'] == programa_seleccionado]
    df_anin = df[df['programa'] == 'ANIN']
    
    # Combinar los datos
    df_combined = pd.concat([df_selected, df_anin])
    
    fig = go.Figure()
    
    # Convertir la columna x a categórica para mejor posicionamiento
    categories = df_combined[x_col].unique()
    
    for i, (regimen, group) in enumerate(df_combined.groupby(x_col)):
        programa = group['programa'].iloc[0]
        
        color = COLOR_ANIN if programa == 'ANIN' else (COLOR_OTROS if programa == selected_program else COLOR_TOTAL)
        
        if not group.empty and pd.notnull(group['min'].iloc[0]) and pd.notnull(group['max'].iloc[0]):
            fig.add_trace(go.Box(
                x=[i]*3,
                y=[group['min'].iloc[0], group['media'].iloc[0], group['max'].iloc[0]],
                name=f"{regimen} ({programa})",
                boxpoints='all',
                jitter=0,
                pointpos=0,
                marker=dict(color=color, size=10, line=dict(width=2)),
                line=dict(color=color, width=2),
                whiskerwidth=0.3,
                fillcolor=f"rgba{(*hex_to_rgb(color), 0.1)}",
                hoverinfo='y',
                width=0.6
            ))
    
    # Personalizar diseño
    fig.update_layout(
        title=dict(text=f"{title} - Comparativa con ANIN", x=0.5, xanchor='center'),
        xaxis=dict(
            title=x_col,
            tickvals=list(range(len(categories))),
            ticktext=categories,
            type='category'
        ),
        yaxis_title="Remuneración (S/)",
        showlegend=True,
        height=600,
        margin=dict(l=80, r=80, b=120, t=80, pad=10),
        boxgap=0.5,
        boxgroupgap=0.3,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Añadir anotaciones
    for i, (regimen, group) in enumerate(df_combined.groupby(x_col)):
        programa = group['programa'].iloc[0]
        color = COLOR_ANIN if programa == 'ANIN' else (COLOR_OTROS if programa == selected_program else COLOR_TOTAL)
        
        if not group.empty:
            # Media
            fig.add_annotation(
                x=i,
                y=group['media'].iloc[0],
                text=f"<b>Media:</b> S/ {group['media'].iloc[0]:,.1f}",
                showarrow=True,
                arrowhead=2,
                font=dict(size=12, color=color),
                bgcolor="white",
                bordercolor=color,
                borderwidth=1
            )
    
    return fig

# Función auxiliar para convertir color HEX a RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

with tab1:
    st.subheader(f"Comparativa por Régimen Laboral: {selected_program} vs ANIN")
    
    # Datos por régimen
    df_regimen = data['Resumen por Regimen']
    df_regimen_selected = df_regimen[df_regimen['programa'] == selected_program]
    df_regimen_anin = df_regimen[df_regimen['programa'] == 'ANIN']
    df_regimen_combined = pd.concat([df_regimen_selected, df_regimen_anin])
    
    # Formatear números
    for col in ['media', 'mediana', 'min', 'max']:
        df_regimen_combined[col] = df_regimen_combined[col].apply(lambda x: f"S/ {x:,.1f}" if pd.notnull(x) else "")
    df_regimen_combined['coef_var'] = df_regimen_combined['coef_var'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")
    
    # Resaltar filas de ANIN
    def highlight_anin(row):
        color = 'background-color: #FFE5E7' if row['programa'] == 'ANIN' else ''
        return [color] * len(row)
    
    st.dataframe(df_regimen_combined[['programa', 'regimen', 'n', 'media', 'mediana', 'min', 'max', 'coef_var']]
                 .style.apply(highlight_anin, axis=1)
                 .format({'n': '{:,.0f}'}), 
                 height=(len(df_regimen_combined) * 35 + 38),
                 use_container_width=True)
    
    # Gráfico comparativo
    if not df_regimen_combined.empty:
        fig_reg = crear_grafico_comparativo(
            df=data['Resumen por Regimen'],
            x_col='regimen',
            y_col='media',
            title="Distribución de Remuneraciones por Régimen",
            programa_seleccionado=selected_program
        )
        st.plotly_chart(fig_reg, use_container_width=True)

with tab2:
    st.subheader(f"Comparativa por Categoría Laboral: {selected_program} vs ANIN")
    
    # Datos por categoría
    df_categoria = data['Resumen por Categoria']
    df_categoria_selected = df_categoria[df_categoria['programa'] == selected_program]
    df_categoria_anin = df_categoria[df_categoria['programa'] == 'ANIN']
    df_categoria_combined = pd.concat([df_categoria_selected, df_categoria_anin])
    
    # Formatear números
    for col in ['media', 'mediana', 'min', 'max']:
        df_categoria_combined[col] = df_categoria_combined[col].apply(lambda x: f"S/ {x:,.1f}" if pd.notnull(x) else "")
    df_categoria_combined['coef_var'] = df_categoria_combined['coef_var'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")
    
    # Mostrar tabla comparativa
    st.dataframe(df_categoria_combined[['programa', 'categoria_laboral', 'n', 'media', 'mediana', 'min', 'max', 'coef_var']]
                 .style.apply(highlight_anin, axis=1)
                 .format({'n': '{:,.0f}'}), 
                 height=(len(df_categoria_combined) * 35 + 38),
                 use_container_width=True)
    
    # Gráfico comparativo
    if not df_categoria_combined.empty:
        fig_cat = crear_grafico_comparativo(
            df=data['Resumen por Categoria'],
            x_col='categoria_laboral',
            y_col='media',
            title="Distribución de Remuneraciones por Categoría",
            programa_seleccionado=selected_program
        )
        st.plotly_chart(fig_cat, use_container_width=True)

with tab3:
    st.subheader(f"Comparativa de Índices de Desigualdad: {selected_program} vs ANIN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparativa de Gini
        st.markdown("### 📈 Índice de Gini Comparativo")
        gini_selected = data['Indice Gini'][data['Indice Gini']['programa'] == selected_program]['indice_gini'].values[0]
        gini_anin = data['Indice Gini'][data['Indice Gini']['programa'] == 'ANIN']['indice_gini'].values[0]
        
        delta_gini = (gini_anin - gini_selected) * 100
        st.metric(
            label=f"ANIN: {gini_anin*100:.1f}%",
            value=f"{selected_program}: {gini_selected*100:.1f}%",
            delta=f"{delta_gini:.1f}%",
            delta_color="inverse",
            help="0% = perfecta igualdad, 100% = máxima desigualdad"
        )
        
        # Comparativa de Theil por Sexo
        st.markdown("### 👥 Theil por Sexo Comparativo")
        theil_sexo_selected = data['Theil por Sexo'][data['Theil por Sexo']['programa'] == selected_program]
        theil_sexo_anin = data['Theil por Sexo'][data['Theil por Sexo']['programa'] == 'ANIN']
        
        st.markdown("**Total**")
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("ANIN", f"{theil_sexo_anin['theil_total_sexo'].values[0]:.3f}")
        with col1b:
            delta = theil_sexo_anin['theil_total_sexo'].values[0] - theil_sexo_selected['theil_total_sexo'].values[0]
            st.metric(selected_program, f"{theil_sexo_selected['theil_total_sexo'].values[0]:.3f}",
                     delta=f"{delta:.3f}", delta_color="inverse")
    
    with col2:
        # Comparativa de Theil por Categoría
        st.markdown("### 🏷️ Theil por Categoría Comparativo")
        theil_cat_selected = data['Theil por Categoria'][data['Theil por Categoria']['programa'] == selected_program]
        theil_cat_anin = data['Theil por Categoria'][data['Theil por Categoria']['programa'] == 'ANIN']
        
        st.markdown("**Total**")
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("ANIN", f"{theil_cat_anin['theil_total_categoria'].values[0]:.3f}")
        with col2b:
            delta = theil_cat_anin['theil_total_categoria'].values[0] - theil_cat_selected['theil_total_categoria'].values[0]
            st.metric(selected_program, f"{theil_cat_selected['theil_total_categoria'].values[0]:.3f}",
                     delta=f"{delta:.3f}", delta_color="inverse")
        
        st.markdown("---")
        st.markdown("**Interpretación:**")
        st.markdown("- 🔴 **ANIN** (color rojo) muestra los indicadores para la Autoridad Nacional de Infraestructura")
        st.markdown(f"- 🔵 **{selected_program}** (color azul) representa el programa seleccionado")
        st.markdown("- 📉 Valores más bajos indican menor desigualdad salarial")

# Nota al pie
st.markdown("---")
st.caption("© 2025 - Análisis de Remuneraciones de Programas y ANIN desarrollado por Raúl Mauro | Datos abiertos del Estado peruano | Versión 2.0")
