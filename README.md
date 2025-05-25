# Dashboard: Análisis de Remuneraciones en Programas Estatales del Perú

Este proyecto permite explorar de manera interactiva los salarios, regímenes laborales y desigualdades en catorce programas estatales que fueron objeto de una propuesta de extinción en mayo de 2025. Los datos han sido extraídos y procesados a partir del portal de transparencia del Estado peruano.

## 🎯 Objetivo

Brindar a ciudadanos, investigadores, periodistas y trabajadores públicos una herramienta que permita:

- Consultar información salarial desagregada por programa estatal.
- Comparar remuneraciones según régimen laboral y categoría laboral.
- Analizar la distribución salarial y los niveles de desigualdad (índice de Gini y de Theil).
- Promover la transparencia y el análisis riguroso de las políticas públicas laborales.

## 🧰 Herramientas utilizadas

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- Índices de desigualdad: Gini y Theil

## 📊 ¿Qué contiene el dashboard?

- Tablas de resumen por **régimen laboral** y **categoría laboral**.
- Histograma de distribución de remuneraciones.
- Cálculo de **índice de Gini** por programa.
- Cálculo de **índice de Theil**, incluyendo separación entre grupos (sexo y categoría) e intragrupos.
- Comparación de todos los programas de manera agregada.

## 🚀 Cómo usar el dashboard

Una vez desplegado en Streamlit Cloud, simplemente:

1. Accede al enlace público (por ejemplo: `https://raulmauro-14infraprogramsperu.streamlit.app`).
2. Selecciona un programa estatal en el menú desplegable.
3. Explora la información disponible para ese programa.
4. Consulta los indicadores de desigualdad al final del análisis.

## 📁 Estructura del repositorio

14infraprogramsperu/
├── dashboard_estadisticas.py # Código principal del dashboard
├── estadisticas_programas_con_total.xlsx # Base de datos procesada
└── README.md # Este archivo


## 📬 Contacto

Desarrollado por **Raul Mauro** con fines de transparencia y análisis público.  
Puedes encontrarme en: [https://tiktok.com/@raulmauro](https://tiktok.com/@raulmauro)

---

**Nota:** Este proyecto no representa posiciones oficiales del gobierno y fue desarrollado con fines académicos y ciudadanos.
