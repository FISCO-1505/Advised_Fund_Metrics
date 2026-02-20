import streamlit as st
import subprocess
import sys
import os

def ensure_private_lib():
    # DETECCIÓN AUTOMÁTICA DE ENTORNO
    # Streamlit Cloud siempre define 'STREAMLIT_RUNTIME_IS_CO_PILOT_RESIDENT' o 'HOME' como /home/adminuser
    is_cloud = os.environ.get("STREAMLIT_RUNTIME_IS_CO_PILOT_RESIDENT") or os.path.exists("/home/adminuser")

    if is_cloud:
        # Ruta en la Nube (Segura y fuera del repo)
        local_lib_path = "/tmp/fisco_vendor"
    else:
        # Ruta Local (Para que lo veas en tu carpeta vendor)
        local_lib_path = os.path.join(os.getcwd(), "vendor")

    # Aseguramos que la ruta esté en el path de Python antes de intentar el import
    if local_lib_path not in sys.path:
        sys.path.insert(0, local_lib_path)

    try:
        # Intentamos importar la librería
        import FISCO_Sources
    except ImportError:
        if "GITHUB_TOKEN" in st.secrets:
            token = st.secrets["GITHUB_TOKEN"]
            repo_url = f"git+https://{token}@github.com/FISCO-1505/Finaccess_Resources.git"
            
            if not os.path.exists(local_lib_path):
                os.makedirs(local_lib_path)
            
            try:
                # Instalación silenciosa
                subprocess.check_call([
                    sys.executable, "-m", "pip", 
                    "install", "--target", local_lib_path, 
                    repo_url
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Forzamos refresco de módulos instalados
                st.rerun() 
            except Exception:
                st.error("Error crítico: No se pudo configurar el entorno de seguridad.")
                st.stop()
        else:
            st.error("Credenciales GITHUB_TOKEN no encontradas en Secrets.")
            st.stop()

# Ejecutar la función
ensure_private_lib()


# import streamlit as st
# import subprocess
# import sys
# import os

# def ensure_private_lib():
#     # Definimos una ruta local con permisos de escritura
#     local_lib_path = os.path.join(os.getcwd(), "vendor")
    
#     try:
#         # Intentamos importar
#         from FISCO_Sources import auth
#     except ImportError:
#         if "GITHUB_TOKEN" in st.secrets:
#             token = st.secrets["GITHUB_TOKEN"]
#             repo_url = f"git+https://{token}@github.com/FISCO-1505/Finaccess_Resources.git"
            
#             # Aseguramos que la carpeta exista
#             if not os.path.exists(local_lib_path):
#                 os.makedirs(local_lib_path)
#             try:
#                 subprocess.check_call([
#                     sys.executable, "-m", "pip", 
#                     "install", "--target", local_lib_path, 
#                     repo_url
#                 ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#                 if local_lib_path not in sys.path:
#                     sys.path.insert(0, local_lib_path)
                
#                 st.success("Configuración de seguridad completada.")
#             except Exception:
#                 st.error("Error de configuración: No se pudo acceder a los recursos privados.")
#                 st.stop()
#         else:
#             st.error("Credenciales no encontradas.")
#             st.stop()

# ensure_private_lib()


import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import calendar
from pathlib import Path

import Kit_Funciones as kit_funciones
import Kit_Metricas as kit_metrics
from cryptography.fernet import Fernet
from FISCO_Sources import auth, crypto, images

#display options
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.6f}'.format(x))
pd.set_option('display.max_columns', None)


images.imagen_f("Advised Funds Metrics")

#%%
def main():
    
    # Obtener ruta del archivo
    global ruta_base
    ruta_base = Path(__file__).resolve().parent
    

    


    # 2. Llamada a tu librería para validar acceso
    # Pasamos el secreto y el kit de funciones como argumento
    acceso_concedido = auth.verificar_acceso(st.secrets["PSW_STREAMLIT"], crypto)

    if not acceso_concedido:
        # Aquí puedes poner un mensaje opcional o dejarlo en blanco
        st.info("Por favor, inicia sesión en el menú lateral para continuar.")


    else:
        data = None
        # ______________________________________ Contenido Principal ______________________________________
        
        # Insertar logo finaccess
        # image = Image.open(path.join(ruta_base, "Resources", "Imagenes", "Logo_finaccess_azul.png"))
        # st.image(image)
        
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

                images.imagen_f_azul()

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

            st.title("Página de Inicio", text_alignment="center")

            #llamado de la imagen de inicio
            images.imagen_home()

            st.info("Bienvenido. Por favor sube un archivo para comenzar el análisis.")


        #____________________________________ Cerrar Sesión ____________________________________
        if st.sidebar.button("Cerrar sesión"):
            st.cache_data.clear()
            st.toast("Caché eliminada")
            st.session_state["pswd"] = False
            st.rerun()
        
if __name__ == "__main__":
    main()