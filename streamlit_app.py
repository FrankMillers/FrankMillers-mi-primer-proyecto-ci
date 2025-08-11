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
