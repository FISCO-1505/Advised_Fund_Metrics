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
import numpy as np
import warnings
from datetime import datetime
import calendar
from pathlib import Path

import Kit_Funciones as kit_funciones
import Kit_Metricas as kit_metrics
import Test_BMRK as test #solo es para hacer pruebas
from cryptography.fernet import Fernet
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
                data = kit_funciones.cargar_datos_excel(archivo_subido)
        
        # with st.sidebar:
        #     # Título
            st.title(":blue[Select an option]")
            #
            topic = st.selectbox("Choose one:",["Funds", "Portfolio","Comparative"],
                                  label_visibility="collapsed"
                                  )

            # Pills Options
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
                if topic is not None and selection is not None:
                    
                    st.title(":blue[Select the assets]")
                    assets=st.multiselect("Select all the assets that you want:",
                            kit_funciones.topic_filter(topic,data["Info"]),
                            label_visibility="collapsed")

                    st.title(":blue[Select the stats]")
                    stats=st.multiselect("Select all the stats that you want:",
                            kit_funciones.stats_filter(topic),
                            label_visibility="collapsed")
                    
                    # Seleccionar las fechas                    
                    # select_date=kit_funciones.select_date(data["Prices"]["Date"],selection)

                    if selection in ["MTD","YTD","1Y","Since Inception"]:
                        selected_date=kit_funciones.calendar(data["Prices"]["Date"], mode="single")
                        # return selected_date
                    
                    elif selection == "Custom Date":
                        start_date, end_date=kit_funciones.calendar(data["Prices"]["Date"], mode="range")
                        pass
                        # return start_date, end_date


                    #####________ Se seleccionan la periodicidad ________#####
                    
                    if selection == "Since Inception":

                        button_load=st.button("Load metrics")
                        if button_load:

                            st.header(f"{selection} preview table")
                            st.subheader("Funds_Commodity_test")
                            st.write(selected_date)
                            funds_results_test=kit_metrics.Funds_Commodity_test(data,selected_date,selection)
                            st.dataframe(funds_results_test[stats].loc[assets])

                    elif selection == "MTD":
                        
                        button_load=st.button("Load metrics")
                        if button_load:
                            st.header(f"{selection} preview table")

                    elif selection == "1Y":
                        
                        button_load=st.button("Load metrics")
                        if button_load:

                            st.header(f"{selection} preview table")
                            st.subheader("Funds_Commodity_test")
                            st.write(selected_date)
                            #esto si pero haremos pruebas

                            # funds_results_test=kit_metrics.Funds_Commodity_test(data,selected_date,selection)
                            # st.dataframe(funds_results_test[stats].loc[assets].style.format("{:.8f}"))
                            # st.write("___________________________________")
                            st.subheader("Test de la nueva estructura")
                            funds_results_test=test.Funds_Commodity_test(data,selected_date,selection)
                            st.dataframe(funds_results_test[stats].loc[assets].style.format("{:.8f}"))



                            # funds_results=kit_metrics.Funds_Commodity_test(data,selected_date)
                            # st.dataframe(funds_results[stats].loc[assets])

                            # funds_resul=kit_metrics.Funds_Commodity(data,selected_date)
                            # st.dataframe(funds_resul[stats].loc[assets])



                            # st.subheader("Funds_Commodity")
                            # funds_results=kit_metrics.Funds_Commodity(data,'2018-01-04')
                            # st.dataframe(funds_results)

                        # st.subheader("Index_test")
                        # index_result_test=kit_metrics.Index_test(data,'2017-01-03','1Y')#1Y: '2016-12-06'; YTD: '2018-01-04'
                        # st.dataframe(index_result_test)
                        # st.subheader("Index")
                        # index_result=kit_metrics.Index(data,'2018-08-10')
                        # st.dataframe(index_result)

                        # st.subheader("Benchmark_test")
                        # kit_metrics.Benchmark_test(data,'2018-08-10','1Y')
                        # st.subheader("Benchmark")
                        # kit_metrics.Benchmark(data,'2018-01-08')

                    elif selection == "YTD":

                        button_load=st.button("Load metrics")
                        if button_load:

                            st.header(f"{selection} preview table")
                            st.subheader("Funds_Commodity_test")
                            # funds_results_test=kit_metrics.Funds_Commodity_test(data,selected_date,selection)
                            # st.dataframe(funds_results_test[stats])
                            st.subheader("Test de la nueva estructura")
                            funds_results_test=test.Funds_Commodity_test(data,selected_date,selection)
                            st.dataframe(funds_results_test[stats].loc[assets].style.format("{:.8f}"))

                            # # auto_correos_carpetas()
                            # st.header("Tablas de YTD")
                            # st.subheader("Funds_Commodity_test")
                            # funds_results_test2=kit_metrics.Funds_Commodity_test(data,'2018-01-04','YTD')
                            # st.dataframe(funds_results_test2)
                            # st.subheader("Funds_Commodity")
                            # funds_results2=kit_metrics.Funds_Commodity(data,'2018-01-04')
                            # st.dataframe(funds_results2)

                            # st.subheader("Index_test")
                            # index_result_test2=kit_metrics.Index_test(data,'2017-01-03','YTD')#1Y: '2016-12-06'; YTD: '2018-01-04'
                            # st.dataframe(index_result_test2)
                            # st.subheader("Index")
                            # index_result2=kit_metrics.Index(data,'2018-08-10')
                            # st.dataframe(index_result2)

                            # st.subheader("Benchmark_test")
                            # kit_metrics.Benchmark_test(data,'2018-08-10','YTD')
                            # st.subheader("Benchmark")
                            # kit_metrics.Benchmark(data,'2018-01-08')
                    
                    elif selection == "Custom Date":
                        button_load=st.button("Load metrics")
                        if button_load:
                            
                            st.subheader("Test de la nueva estructura")
                            funds_results_test=test.Funds_Commodity_test(data,end_date,selection,start_date)
                            st.dataframe(funds_results_test[stats].loc[assets].style.format("{:.8f}"))

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