"""
Descargador automático del dataset COVID-19 Radiography Database
Dataset público de Kaggle para análisis académico

GRUPO 02:
- ALIPIO ESQUIVEL FRANK MILLER
- CASTAÑEDA COBEÑAS JORGE LUIS  
- VASQUEZ MORAN LIZARDO VIDAL
"""

import kagglehub
import pandas as pd
import os
from pathlib import Path
import streamlit as st

class DescargadorKaggle:
    """Clase para descargar y gestionar dataset de Kaggle"""
    
    def __init__(self, ruta_datos="datos/"):
        self.ruta_datos = Path(ruta_datos)
        self.nombre_dataset = "tawsifurrahman/covid19-radiography-database"
        self.ruta_descarga = self.ruta_datos / "kaggle" / "covid19_radiography"
    
    def descargar_dataset(self):
        """Descarga el dataset COVID-19 Radiography Database"""
        try:
            st.info("📥 Descargando dataset de Kaggle...")
            
            # Descargar usando kagglehub
            ruta_dataset = kagglehub.dataset_download(self.nombre_dataset)
            
            st.success(f"✅ Dataset descargado en: {ruta_dataset}")
            return ruta_dataset
            
        except Exception as e:
            st.error(f"❌ Error descargando dataset: {e}")
            return None
    
    def obtener_informacion_dataset(self):
        """Retorna información del dataset"""
        info_dataset = {
            "nombre": "COVID-19 Radiography Database",
            "fuente": "Kaggle - tawsifurrahman",
            "url": "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database",
            "descripcion": "Dataset público con radiografías de tórax para COVID-19, Neumonía y casos Normales",
            "clases": ["COVID", "Normal", "Pneumonia", "Viral Pneumonia"],
            "total_imagenes": "21,165 imágenes aproximadamente",
            "formato": "PNG, JPG",
            "uso_academico": "Permitido para investigación y educación"
        }
        return info_dataset

def mostrar_informacion_dataset():
    """Muestra información del dataset en Streamlit"""
    descargador = DescargadorKaggle()
    info = descargador.obtener_informacion_dataset()
    
    st.markdown("## 📊 Dataset: COVID-19 Radiography Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Información General")
        st.write(f"**Nombre:** {info['nombre']}")
        st.write(f"**Fuente:** {info['fuente']}")
        st.write(f"**Total:** {info['total_imagenes']}")
        st.write(f"**Formato:** {info['formato']}")
    
    with col2:
        st.markdown("### 🏷️ Clases del Dataset")
        for clase in info['clases']:
            st.write(f"• {clase}")
        
        st.markdown("### 🔗 Enlaces")
        st.markdown(f"[Ver en Kaggle]({info['url']})")
    
    return descargador
