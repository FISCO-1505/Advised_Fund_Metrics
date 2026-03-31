import streamlit as st
import subprocess
import sys
import os

def ensure_private_lib():
    # Si existe /home/adminuser, estamos en Streamlit Cloud
    is_cloud = os.path.exists("/home/adminuser")
    
    if is_cloud:
        # En la nube usamos /tmp para evitar problemas de permisos de escritura
        local_lib_path = "/tmp/fisco_vendor"
    else:
        # En local usamos la carpeta vendor en el directorio actual
        local_lib_path = os.path.join(os.getcwd(), "vendor")

    # Asgurar que la ruta exista
    if not os.path.exists(local_lib_path):
        os.makedirs(local_lib_path)
    
    if local_lib_path not in sys.path:
        # Insertamos al inicio para dar prioridad a nuestra librería
        sys.path.insert(0, local_lib_path)

    try:
        import FISCO_Sources
    except ImportError:
        if "GITHUB_TOKEN" in st.secrets:
            token = st.secrets["GITHUB_TOKEN"]
            repo_url = f"git+https://{token}@github.com/FISCO-1505/Finaccess_Resources.git"
            
            with st.spinner("Uploading sources..."):
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", 
                        "install", "--target", local_lib_path, 
                        repo_url
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    st.success("Sources loaded correctly.")
                    import FISCO_Sources
                except Exception as e:
                    st.error(f"Error: Library did not load correctly.")
                    st.stop()
        else:
            st.error("No se encontró GITHUB_TOKEN en los Secrets de Streamlit.")
            st.stop()

# Ejecutar la función
ensure_private_lib()

import pandas as pd
# import numpy as np
import warnings
# from datetime import datetime
# import calendar
from pathlib import Path

import Kit_Funciones_Secundarias as kit_f_secundarias
import Kit_Metricas as kit_metrics
import Kit_Funciones_Principales as kit_f_principales
# from cryptography.fernet import Fernet
from FISCO_Sources import auth, crypto, images

import gc

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

    # Llamada a tu librería para validar acceso
    acceso_concedido = auth.verificar_acceso(st.secrets["PSW_STREAMLIT"], crypto,"EN")

    if not acceso_concedido:
        # Aquí puedes poner un mensaje opcional o dejarlo en blanco
        st.info("Please, login on the side bar menu.")


    else:
        data = None
        # ______________________________________ Contenido Principal ______________________________________
        
        with st.sidebar:
            st.success("¡Access granted!", icon=":material/lock_open:")            
        
            #cargar archivo
            archivo_subido = st.file_uploader("Load the Excel file", type=["xlsx","xls"])
            
            if archivo_subido is None:
                st.warning("Please upload the corresponding file.")
            else:
                data = kit_f_secundarias.cargar_datos_excel(archivo_subido)
        
            st.title(":blue[Select an option]")

            topic = st.selectbox("Choose one:",["Funds", "Portfolio","Comparative"],
                                  label_visibility="collapsed"
                                  )

            selection = st.pills(label="Options", label_visibility="collapsed",
                                 options=["Home", "MTD", "YTD","1Y",
                                          "Since Inception","Custom Date"],
                                 default="Home"
                                )

        #imagen del logo de la institución    
        images.imagen_home("Advisors")

        # Ejecutar opción seleccionada
        if selection != "Home":
            if data is not None:
                if topic=="Funds" and selection is not None:

                    st.markdown("<h1 style='text-align: center; color: #1D59A9;'>Funds Analysis</h1>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color: #1D59A9;'>Select the assets</h3>", unsafe_allow_html=True)
                    all_assets,assets_selected=kit_f_secundarias.assets_filter(topic,data["Info"])
                    assets=st.multiselect("Select all the assets that you want:",
                            all_assets,
                            label_visibility="collapsed",default=assets_selected)

                    # st.subheader(":blue[Select the stats]")
                    st.markdown("<h3 style='color: #1D59A9;'>Select the stats</h3>", unsafe_allow_html=True)
                    all_stats,stats_selected=kit_f_secundarias.stats_filter(topic)
                    stats=st.multiselect("Select all the stats that you want:",
                            all_stats,
                            label_visibility="collapsed",default=stats_selected)
                    
                    
                    if selection in ["MTD","YTD","1Y","Since Inception"]:
                        selected_date=kit_f_secundarias.calendar(data["Prices"]["Date"], mode="single")
                        
                    
                    elif selection == "Custom Date":
                        start_date, end_date=kit_f_secundarias.calendar(data["Prices"]["Date"], mode="range")
                        pass
                       


                    #####________ Se seleccionan la periodicidad ________#####
                    
                    if selection == "Since Inception":
                        toggle_button=st.toggle("Comparative")
                        if toggle_button:
                            stats_aux=stats
                            assets=assets
                        else:
                            stats_aux=None
                            assets_aux=None

                        button_load=st.button("Load metrics")
                        if button_load:

                            st.header(f"{selection} preview table")
                            st.subheader("Funds_Commodity_test")
                            st.write(selected_date)
                            funds_results_test=kit_metrics.Funds_Commodity_test(data,selected_date,selection,stats_aux,assets_aux)
                            st.dataframe(funds_results_test[stats].loc[assets])
                            

                    elif selection == "MTD":
                        
                        button_load=st.button("Load metrics")
                        if button_load:
                            st.header(f"{selection} preview table")

                    elif selection == "1Y":

                        toggle_button=st.toggle("Comparative")

                        if toggle_button:
                            stats_aux=stats
                            assets_aux=assets
                        else:
                            stats_aux=None
                            assets_aux=None
                        
                        button_load=st.button("Load metrics")
                        if button_load:

                            # st.header(f"{selection} preview table")
                            # st.write("___________________________________")
                            st.subheader("Test de la nueva estructura")
                            funds_results_test=kit_f_principales.Funds_Commodity_test(data,selected_date,selection,stats_aux,assets_aux)
                            st.dataframe(funds_results_test[stats+['Real Date']].loc[assets].style.format("{:.8f}"))

                            # kit_f_principales.Benchmark_test(data,selected_date)
                            # st.dataframe(bmrk_results_test.style.format("{:.8f}"))


                    elif selection == "YTD":

                        toggle_button=st.toggle("Comparative")
                        if toggle_button:
                            stats_aux=stats
                            assets_aux=assets
                        else:
                            stats_aux=None
                            assets_aux=None

                        button_load=st.button("Load metrics")
                        if button_load:

                            st.subheader("Test de la nueva estructura")
                            funds_results_test=kit_f_principales.Funds_Commodity_test(data,selected_date,selection,stats_aux,assets_aux)
                            st.dataframe(funds_results_test[stats].loc[assets].style.format("{:.8f}"))

                    
                    elif selection == "Custom Date":

                        toggle_button=st.toggle("Comparative")
                        if toggle_button:
                            stats_aux=stats
                            assets_aux=assets
                        else:
                            stats_aux=None
                            assets_aux=None

                        button_load=st.button("Load metrics")
                        if button_load:
                            
                            st.subheader("Test de la nueva estructura")
                            funds_results_test=kit_f_principales.Funds_Commodity_test(data,end_date,selection, stats_aux,assets_aux,start_date)
                            st.dataframe(funds_results_test[stats].loc[assets].style.format("{:.8f}"))
                


                elif topic=="Portfolio" and selection is not None:
                    st.header("Inicia apartado de los portafolios")



                    pass

                
                elif topic == "Comparative" and selection is not None:
                    
                    st.markdown("<h1 style='text-align: center; color: #1D59A9;'>Comparative Analysis</h1>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color: #1D59A9;'>Select the assets</h3>", unsafe_allow_html=True)

                    

            else:
                st.info("⚠️ First, upload the Excel file on the side bar to view this section.")
        else:
            pass
            # st.title("Página de Inicio", text_alignment="center")

            #llamado de la imagen de inicio
            # images.imagen_home("Advisors")

            # st.info("Bienvenido. Por favor sube un archivo para comenzar el análisis.")


        #____________________________________ Cerrar Sesión ____________________________________
        if st.sidebar.button("Log out"):
            #se limpian las funcines en donde se haya buscado optimzar el tiempo en RAm
            st.cache_data.clear()
            st.cache_resource.clear()

            #borra y hace una liempieza total del cache y de RAM
            st.session_state.clear()

            #forza la liberación de memoria física
            gc.collect()

            st.toast("Caché eliminada")

            st.rerun()
        
if __name__ == "__main__":
    main()