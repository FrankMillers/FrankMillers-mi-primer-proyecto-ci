# Leer archivo
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Buscar y reemplazar la función de carga
old_loading = '''try:
                # Método 1: Carga estándar
                modelo = tf.keras.models.load_model(ruta_modelo)
                st.success("✅ Modelo MobileNetV2 cargado exitosamente")
                st.info(f"📊 Arquitectura: {len(modelo.layers)} capas")
                return modelo
                
            except Exception as e1:
                st.warning(f"⚠️ Método 1 falló, probando método 2...")
                try:
                    # Método 2: Carga sin compilar
                    modelo = tf.keras.models.load_model(ruta_modelo, compile=False)
                    st.success("✅ Modelo cargado sin compilar (funcional)")
                    st.info(f"📊 Arquitectura: {len(modelo.layers)} capas")
                    return modelo
                    
                except Exception as e2:
                    st.error(f"❌ Error cargando modelo: {str(e2)[:100]}")
                    return None'''

new_loading = '''try:
                # Cargar directamente sin compilar (compatible con todas las versiones)
                modelo = tf.keras.models.load_model(ruta_modelo, compile=False)
                st.success("✅ Modelo MobileNetV2 cargado exitosamente")
                st.info(f"📊 Arquitectura: {len(modelo.layers)} capas")
                return modelo
                    
            except Exception as e:
                st.error(f"❌ Error cargando modelo: {str(e)[:100]}")
                return None'''

content = content.replace(old_loading, new_loading)

# Guardar
with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Modelo configurado para cargar sin errores")
