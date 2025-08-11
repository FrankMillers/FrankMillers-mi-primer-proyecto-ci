import streamlit as st
import numpy as np
from PIL import Image

st.title('🏥 Sistema IA COVID-19 - Grupo 02')

# Navegación a EDA
if st.sidebar.button('📊 Ver Análisis EDA'):
    st.markdown('## 📊 Análisis Exploratorio de Datos')
    st.info('Dataset: COVID-19 Radiography Database de Kaggle')
    st.write('**Clases:** COVID, Normal, Pneumonia, Viral Pneumonia')
    st.write('**Total:** ~21,165 imágenes')

# Navegación a Comparación de Modelos
if st.sidebar.button('🧠 Comparar Modelos'):
    from src.modelos.comparador_modelos import mostrar_interfaz_comparacion
    mostrar_interfaz_comparacion()

# Navegación a Validación Estadística
if st.sidebar.button('📈 Validación Estadística'):
    from src.modelos.validacion_estadistica import mostrar_validacion_estadistica
    mostrar_validacion_estadistica()
