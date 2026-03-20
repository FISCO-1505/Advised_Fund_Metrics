import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def start_dt(end_date, period, custom_start=None, min_allowed_date='2015-12-03'):
    end_dt = pd.to_datetime(end_date)
    min_dt = pd.to_datetime(min_allowed_date)
 
    if period == "Since Inception":
        return None 
    
    elif period == "Custom Date":
        if custom_start is None:
            # Si no hay fecha seleccionada, devolvemos un error específico
            return "MISSING CUSTOM DATE"
        start_date = pd.to_datetime(custom_start)
        
    elif period == "MTD":
        start_date = (end_dt.replace(day=1) - pd.Timedelta(days=1))
        
    elif period == "YTD":
        start_date = pd.Timestamp(year=end_dt.year - 1, month=12, day=31)
        
    elif period == "1Y":
        start_date = end_dt - pd.DateOffset(years=1)
        
    
    # 4. VALIDACIÓN DE LÍMITE (Para todos excepto Inception)
    if start_date < min_dt:
        return "INSUFFICIENT_DATA"

    return start_date


@st.cache_data(show_spinner=False)
def obtener_nombres_hojas(file):
    excel_file = pd.ExcelFile(file)
    return excel_file.sheet_names

@st.cache_resource(show_spinner=False)
def cargar_datos_excel(_file):
    dict_dfs = pd.read_excel(_file, sheet_name=None)
    
    for sheet in dict_dfs:
        dict_dfs[sheet] = dict_dfs[sheet].applymap(
            lambda x: str(x) if isinstance(x, list) else x
        )
    return dict_dfs

@st.cache_data(show_spinner=False)
def topic_filter(topic,_data):
    '''
    topic: nombre del apartado principal para obtener el los fondos/portafolios
    funds_list: lista de los fondos/portafolios filtrados
    '''

    funds_list=[]

    if topic == "Funds":
        
        funds_list=_data[(_data["Type"]=="Fund") | (_data['Type']=='Commodity')]["Ticker"].tolist()
        
    elif topic== "Portfolio":
        pass
    elif topic == "Benchmark":
        pass
        # funds_list=_data[_data["Type"]=="Index"]["Ticker"].tolist()
    elif topic == "Index":
        pass
        # funds_list=_data[_data["Type"]=="Index"]["Ticker"].tolist()

    return funds_list


@st.cache_data(show_spinner=False)
def stats_filter(topic):
    '''
    topic: nombre del apartado principal para obtener las métricas correspondientes
    stats_list: list de las estadisticas de acuerdo al topic
    '''

    stats_list=[]

    if topic == "Funds":
        stats_list=["PX_LAST","% Change","Cumulative","RF","Vol",
                              "Sharpe Ratio","VaR",'Negative Returns',
                              'Downside Deviation', 'Sortino Ratio',
                              'Beta','Treynor Ratio','Daily Active Return',
                              'Tracking Error','Info. Ratio','Alpha','J Alpha',
                              'Correlation','R^2','Drawdown','Max. Drawdown']
    
    elif topic== "Portfolio":
        stats_list=[]    
    
    elif topic == "Benchmark":
        stats_list=["PX_LAST","% Change","Cumulative","RF","Vol",
                              "Sharpe Ratio","VaR",'Negative Returns',
                              'Downside Deviation', 'Sortino Ratio']

    elif topic == "Index":
        stats_list=["PX_LAST","% Change","Cumulative","RF","Vol",
                              "Sharpe Ratio"]

    return stats_list


def calendar(df, mode):
    """
    df: DataFrame con los datos (considerando solo la columna de las fechas).
    mode: "range" (inicio y fin de fechas) o "single" (solo fecha final).
    """
    # Obtener las fechas disponibles
    available_dates = pd.to_datetime(df).dt.date.unique()
    min_date = min(available_dates)
    max_date = max(available_dates)

    if mode == "range":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(":blue[Select the start date]")
            start_date = st.date_input(
                "Start date:",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="start_date_input",
                label_visibility="collapsed"
            )
        
        with col2:
            st.subheader(":blue[Select the end date]")
            end_date = st.date_input(
                "End date:",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="end_date_input",
                label_visibility="collapsed"
            )
        
        # Validaciones
        if start_date not in available_dates or end_date not in available_dates:
            st.warning("⚠️ Warning: One of the selected dates is not available.")
            
        if start_date >= end_date :
            st.error("❌ Error: Start date cannot be after or equal end date.")
            return None, None

        return str(start_date), str(end_date)
    
    else:
        # Fecha Única (solo al fecha final)
        selected_date = st.date_input(
            "Select the end date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="single_date_input",
            # value=None
        )
        
        if selected_date not in available_dates:
            st.error("❌ Error: The selected date is not available: Please select another one.")
            return None
        
        return str(selected_date)


# def select_date(df,selection):
#     """
#     df: DataFrame con los datos (considerando solo la columna de las fechas).
#     selection: depende de la periodicidad que se requiera
#     """
#     if selection in ["MTD","YTD","1Y","Since Inception"]:
#         selected_date=calendar(df, mode="single")
#         return selected_date
    
#     elif selection == "Custom Date":
#         start_date, end_date=calendar(df, mode="range")
#         return start_date, end_date

