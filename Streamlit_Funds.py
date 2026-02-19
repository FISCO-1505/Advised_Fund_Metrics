import streamlit as st
import subprocess
import sys
import os

def install_private_library():
    # 1. Definimos la ruta local
    local_lib_path = os.path.join(os.getcwd(), "lib_interna")
    
    # 2. La agregamos al path ANTES de intentar el import
    if local_lib_path not in sys.path:
        sys.path.insert(0, local_lib_path)

    # 3. EL CANDADO: Intentamos importar. Si funciona, salimos de la función.
    try:
        import FISCO_Sources
        return # <--- IMPORTANTE: Si ya existe, deja de ejecutar esta función
    except ImportError:
        # Solo si NO existe, procedemos a instalar
        if "STREAMLIT_CLOUD_TOKEN" in st.secrets:
            token = st.secrets["STREAMLIT_CLOUD_TOKEN"]
            repo_url = f"git+https://{token}@github.com/FISCO-1505/Finaccess_Resources.git"
            
            try:
                # Usamos --no-deps para evitar conflictos y --upgrade por si acaso
                subprocess.check_call([
                    sys.executable, "-m", "pip", 
                    "install", "--target", local_lib_path, "--no-deps", "--upgrade", repo_url
                ])
                st.rerun() # Reinicia solo una vez para reconocer el paquete
            except Exception as e:
                st.error(f"Error instalando: {e}")
                st.stop()

# Ejecutar la función
install_private_library()


import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import calendar
from pathlib import Path

import Kit_Funciones as kit_funciones
import Kit_Metricas as kit_metrics
from cryptography.fernet import Fernet
from FISCO_Sources import auth, crypto

#display options
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.6f}'.format(x))
pd.set_option('display.max_columns', None)

# Insertar logo finaccess
        # image = Image.open(path.join(ruta_base, "Resources", "Imagenes", "Logo_finaccess_azul.png"))
        # st.image(image)





#%%
def main():
    
    # Obtener ruta del archivo
    global ruta_base
    ruta_base = Path(__file__).resolve().parent
    
    # # Configuración de la página
    # st.set_page_config(page_icon=Image.open(path.join(ruta_base, "Resources", "Imagenes", "Logo_finaccess_f.png")),
    #                    page_title = "Rendimientos México")
    
    # Cargar clave y crear instancia de Fernet
    # clave = kit_funciones.cargar_clave()
    # fernet = Fernet(clave)
    
    # pswd_ok = kit_funciones.desencriptar_con_manejo_errores(st.secrets["PSW_STREAMLIT"], fernet)
    # @st.dialog("Validación de acceso")
    # def validar_contrasena():
    #     with st.form("validar_pswd", enter_to_submit=True):
    #         #se solicta la pswd
    #         password = st.text_input("Ingresa tu acceso", type="password")
        
    #         if st.form_submit_button("Confirmar"):
    #             if password == pswd_ok:
    #                 #se crea una sesion
    #                 st.session_state["pswd"] = True
    #                 #cerrar el dialogo
    #                 st.rerun()
    #             else:
    #                 st.error("Validación incorrecta", icon=':material/error:')
        
    # # Usa Session State para controlar la visibilidad del dialogo y el acceso
    # if "pswd" not in st.session_state:
    #     st.session_state["pswd"] = False
        
    # if not st.session_state["pswd"]:
    #     with st.sidebar:
    #         # Si la contraseña no se ha validado, muestra un botón para abrir el diálogo
    #         if st.button("Validar", icon=":material/lock_person:"):
    #             validar_contrasena()


    # 2. Llamada a tu librería para validar acceso
    # Pasamos el secreto y el kit de funciones como argumento
    acceso_concedido = auth.verificar_acceso(st.secrets["PSW_STREAMLIT"], crypto)

    if not acceso_concedido:
        # Aquí puedes poner un mensaje opcional o dejarlo en blanco
        st.info("Por favor, inicia sesión en el menú lateral para continuar.")


    else:
        data = None
        # ______________________________________ Contenido Principal ______________________________________
        with st.sidebar:
            st.success("¡Acceso concedidoooooooooooooo!", icon=":material/lock_open:")            
        
            #cargar archivo
            archivo_subido = st.file_uploader("Sube tu Excel", type=["xlsx","xls"])
            if archivo_subido is None:
                st.warning("Favor de cargar el archivo correspondiente")
            else:
                data = kit_funciones.cargar_datos_excel(archivo_subido)
        
        
        # Insertar menú lateral
        with st.sidebar:
            # Título
            st.title(":blue[Selecciona una Opción]")
            # Pills Options
            selection = st.pills(label="Options", label_visibility="collapsed",
                                 options=["Home", "1Y", "MTD", "YTD"],
                                 default="Home"
                                )
            
        # Ejecutar opción seleccionada
        if selection != "Home":
            if data is not None:
                if selection == "1Y":
                    # rendimientos()
                    st.header("Tablas de 1Y")
                    st.subheader("Funds_Commodity_test")
                    funds_results_test=kit_metrics.Funds_Commodity_test(data,'2018-01-04','1Y')
                    st.dataframe(funds_results_test)
                    st.subheader("Funds_Commodity")
                    funds_results=kit_metrics.Funds_Commodity(data,'2018-01-04')
                    st.dataframe(funds_results)

                    st.subheader("Index_test")
                    index_result_test=kit_metrics.Index_test(data,'2017-01-03','1Y')#1Y: '2016-12-06'; YTD: '2018-01-04'
                    st.dataframe(index_result_test)
                    st.subheader("Index")
                    index_result=kit_metrics.Index(data,'2018-08-10')
                    st.dataframe(index_result)

                    # st.subheader("Benchmark_test")
                    # kit_metrics.Benchmark_test(data,'2018-08-10','1Y')
                    # st.subheader("Benchmark")
                    # kit_metrics.Benchmark(data,'2018-01-08')

                elif selection == "T-MTD":
                    # t_approach()
                    st.header("Tablas de MTD")
                elif selection == "YTD":
                    # auto_correos_carpetas()
                    st.header("Tablas de YTD")
                    st.subheader("Funds_Commodity_test")
                    funds_results_test2=kit_metrics.Funds_Commodity_test(data,'2018-01-04','YTD')
                    st.dataframe(funds_results_test2)
                    st.subheader("Funds_Commodity")
                    funds_results2=kit_metrics.Funds_Commodity(data,'2018-01-04')
                    st.dataframe(funds_results2)

                    st.subheader("Index_test")
                    index_result_test2=kit_metrics.Index_test(data,'2017-01-03','YTD')#1Y: '2016-12-06'; YTD: '2018-01-04'
                    st.dataframe(index_result_test2)
                    st.subheader("Index")
                    index_result2=kit_metrics.Index(data,'2018-08-10')
                    st.dataframe(index_result2)

                    # st.subheader("Benchmark_test")
                    # kit_metrics.Benchmark_test(data,'2018-08-10','YTD')
                    # st.subheader("Benchmark")
                    # kit_metrics.Benchmark(data,'2018-01-08')
                # else:
                #     pass
            else:
                st.info("⚠️ Para visualizar esta sección, primero carga el archivo Excel en el menú lateral.")
        else:
            st.info("Bienvenido. Por favor sube un archivo para comenzar el análisis. Pondremos la imagen")

        #____________________________________ Cerrar Sesión ____________________________________
        if st.sidebar.button("Cerrar sesión"):
            st.cache_data.clear()
            st.toast("Caché eliminada")
            st.session_state["pswd"] = False
            st.rerun()
        
if __name__ == "__main__":
    main()