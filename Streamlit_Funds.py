import bootstrap #Carga la libreria de FISCO_Sources, siempre ponerlo
import streamlit as st
from pathlib import Path
import gc

import Kit_Funciones_Secundarias as kit_f_secundarias
import Kit_Funciones_Principales as kit_f_principales
from FISCO_Sources import auth, images

def contenido_principal():
    """
    Función en donde se lleva a cabo el contenido principal de toda 
    la interfaz de streamlit con el llamado respectivo de sus funciones
    para poder llevar a cabo el correcto uso, cálculo y manipulación 
    de la herramienta.
    """

    #imagen del logo de la institución    
    images.imagen_home("Advisors")

    data = None

#_____________
    with st.sidebar:
        st.success("¡Access granted!", icon=":material/lock_open:")            
    
        #cargar archivo
        archivo_subido = st.file_uploader("Load the Excel file", type=["xlsx","xls"])
        
        if archivo_subido is None:
            st.warning("Please upload the corresponding file.")
        else:

            # --- PROCESO DE VALIDACIÓN ---
            es_valido, mensaje = kit_f_secundarias.validar_estructura_excel(archivo_subido)
            
            if not es_valido:
                st.error("⚠️ Validation Error in Excel Structure")
                st.warning(mensaje)
                data = None
            
            else:
                # --- SI PASA LA VALIDACIÓN, CONTINUAR ---
                st.success("File validated successfully!")
                data = kit_f_secundarias.cargar_datos_excel(archivo_subido)
    
        # seleccionar el tipo de proyecto
        st.title(":blue[Select a Process]")
        process = st.selectbox("Choose one:",["Benchmarks Tables", "Advised Funds Metrics"],
                                label_visibility="collapsed"
                                )
        
    if process == "Advised Funds Metrics" and data is not None:

        with st.sidebar:
        
            st.title(":blue[Select an option]")

            topic = st.selectbox("Choose one:",["Funds", "Portfolio","Returns Table"],
                                    label_visibility="collapsed"
                                    )

            if topic == "Returns Table":
                selection = st.pills(label="Options", label_visibility="collapsed",
                                        options=["Home", "YTD"],
                                        default="YTD"
                                )
            else:
                selection = st.pills(label="Options", label_visibility="collapsed",
                                        options=["Home", "MTD", "YTD","1Y",
                                                "Since Inception","Custom Date"],
                                        default="Home"
                                )


        if selection != "Home":
            if data is not None and selection is not None:
                # Títulos dinámicos
                titulo = "Funds Analysis" if topic == "Funds" else ("Portfolio Analysis" if topic == "Portfolio" else "Returns Tables")
                st.markdown(f"<h1 style='text-align: center; color: #1D59A9;'>{titulo}</h1>", unsafe_allow_html=True)
                
                #fucion que decide si es funds o portafolios para poder procesar la info
                if topic in ["Funds", "Portfolio"]:
                    # Filtros de Assets y Stats
                    st.markdown("<h3 style='color: #1D59A9;'>Select assets</h3>", unsafe_allow_html=True)
                    all_assets, assets_selected, ticker_map = kit_f_secundarias.assets_filter(topic, data["Info"])
                    assets = st.multiselect("Assets:", all_assets, default=assets_selected, label_visibility="collapsed")
                
                    #recuperar los assets con el nombre original (ticker)
                    assets_tickers = [ticker_map[n] for n in assets]
                    
                    st.markdown("<h3 style='color: #1D59A9;'>Select stats</h3>", unsafe_allow_html=True)
                    all_stats, stats_selected = kit_f_secundarias.stats_filter(topic)
                    stats = st.multiselect("Stats:", all_stats, default=stats_selected, label_visibility="collapsed")

                    kit_f_principales.procesar_analisis(topic, data, selection, stats, assets_tickers,ticker_map)

                elif topic == "Returns Table":

                    st.markdown("<h3 style='color: #1D59A9;'>Select Portfolios</h3>", unsafe_allow_html=True)
                    all_assets, assets_selected, ticker_map = kit_f_secundarias.assets_filter(topic, data["Info"])
                    assets = st.multiselect("Assets:", all_assets, default=assets_selected, label_visibility="collapsed")
                    
                    #recuperar los assets con el nombre original (ticker)
                    assets_tickers = [ticker_map[n] for n in assets]
                    
                    selected_date = kit_f_secundarias.calendar(data["Prices"]["Date"], mode="single")

                    # Si la fecha o los assets cambian, esta cadena cambiará.
                    current_params = f"{selected_date}_{sorted(assets_tickers)}"

                    #inicializamos estados si no existen
                    if "last_params" not in st.session_state:
                        st.session_state.last_params = current_params
                        st.session_state.cargar_tabla = False

                    # Si los parámetros cambian, apagamos la tabla
                    if st.session_state.last_params != current_params:
                        st.session_state.cargar_tabla = False
                        st.session_state.last_params = current_params
                        
                    if st.button("Load Process", key="btn_tables"):
                        st.session_state.cargar_tabla = True

                    if st.session_state.cargar_tabla:
                        # --- validación ---
                        if not assets_tickers:
                            st.warning("⚠️ Please select at least one asset/portfolio to continue.")
                            return
                        
                        kit_f_principales.tabla_rendimientos(data, selected_date, assets_tickers)
                        st.success("You can download the Reports!")


            elif data is None and selection != "Home":
                st.info("⚠️ First, upload the Excel file on the side bar to view this section.")
        else:
            pass

    elif process == "Benchmarks Tables" and data is not None:
        # Write title
        titulo = "Benchmark Returns"
        st.markdown(f"<h1 style='text-align: center; color: #1D59A9;'>{titulo}</h1>", unsafe_allow_html=True)

        max_date, min_date = kit_f_secundarias.date_bmrk_process(data)

        # Input end date
        st.markdown("<h3 style='color: #1D59A9;'>Select the end date</h3>", unsafe_allow_html=True)
        end_date = st.date_input("Select end date", min_value=min_date, max_value=max_date,label_visibility="collapsed")

        submitted = st.button("Create Excel")
        if submitted:
            # Run main function
            kit_f_principales.bmrk_process(data,end_date)      
            
            # Insert msg 'Done'
            st.success("Done!", icon="✅")

    
    else:
        st.warning("Upload the corresponding file before select a Process")
        

#%%
def main():
    """
    Función main en donde se hace llamado de la gestión segura de
    la librería privada (FISCO_Sources) y así obtener un acceso
    controlado.
    """
    # Obtener ruta del archivo
    global ruta_base
    ruta_base = Path(__file__).resolve().parent

    images.imagen_f("Advised Funds Metrics")

    auth.gestionar_sesion_segura(
        contenido_principal_func = contenido_principal,
        password_secreta = st.secrets["PSW_STREAMLIT"],
        idioma = "EN",
        timeout_segundos = 3600 #1 hora
    )

        
if __name__ == "__main__":
    main()