import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go




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

def assets_filter(topic,_data):
    '''
    topic: nombre del apartado principal para obtener el los fondos/portafolios
    funds_list: lista de los fondos/portafolios filtrados
    '''

    assets=[]
    assets_selected=[]

    pills=st.pills(label="Options for funds", label_visibility="collapsed",
                                 options=["Custom Assets","All Assets", "Manager Assets"],
                                 default="Custom Assets")


    if topic == "Funds":
        assets=_data[(_data["Type"]=="Fund") | (_data['Type']=='Commodity')]["Ticker"].tolist()
        
        if pills=="Custom Assets":
            assets_selected=None
            
        elif pills=="All Assets":
            assets_selected=assets

        elif pills== "Manager Assets":

            assets_selected=assets[:4]
        
        # else:
        #     st.warning("Choose an option")


    elif topic== "Portfolio":
        pass
    elif topic == "Benchmark":
        pass
        # assets=_data[_data["Type"]=="Index"]["Ticker"].tolist()
    elif topic == "Index":
        pass
        # assets=_data[_data["Type"]=="Index"]["Ticker"].tolist()
    
    return assets,assets_selected

def stats_filter(topic):
    '''
    topic: nombre del apartado principal para obtener las métricas correspondientes
    stats_list: list de las estadisticas de acuerdo al topic
    '''

    stats=[]
    stats_selected=[]

    pills=st.pills(label="Options for funds", label_visibility="collapsed",
                                 options=["Custom Stats","All Stats", "Manager Stats"],
                                 default="Custom Stats",key="pills_stats")

    if topic == "Funds":

        stats=["PX_LAST","% Change","Cumulative","RF","Vol",
                                "Sharpe Ratio","VaR",'Negative Returns',
                                'Downside Deviation', 'Sortino Ratio',
                                'Beta','Treynor Ratio','Daily Active Return',
                                'Tracking Error','Info. Ratio','Alpha','J Alpha',
                                'Correlation','R^2','Drawdown','Max. Drawdown']

        if pills=="Custom Stats":
            stats_selected=None
            
        elif pills=="All Stats":
            stats_selected=stats
            

        elif pills== "Manager Stats":
            stats_selected=["PX_LAST","% Change","Cumulative","RF","Vol",]
                                #   "Sharpe Ratio","VaR",'Negative Returns',
                                #   'Downside Deviation', 'Sortino Ratio',
                                #   'Beta','Treynor Ratio','Daily Active Return',
                                #   'Tracking Error','Info. Ratio','Alpha','J Alpha',
                                #   'Correlation','R^2','Drawdown','Max. Drawdown']
        # else:
        #     st.warning("Choose an option")

    
    elif topic== "Portfolio":
        stats=[]    
    
    # elif topic == "Benchmark":
    #     stats_list=["PX_LAST","% Change","Cumulative","RF","Vol",
    #                           "Sharpe Ratio","VaR",'Negative Returns',
    #                           'Downside Deviation', 'Sortino Ratio']

    # elif topic == "Index":
    #     stats_list=["PX_LAST","% Change","Cumulative","RF","Vol",
    #                           "Sharpe Ratio"]
    

    return stats,stats_selected

def calendar(df, mode):
    """
    df: DataFrame con los datos (considerando solo la columna de las fechas).
    mode: "range" (inicio y fin de fechas) o "single" (solo fecha final).
    """

    available_dates = pd.to_datetime(df).dt.date.unique()
    min_date = min(available_dates)
    max_date = max(available_dates)

    if mode == "range":
        col1, col2 = st.columns(2)
        
        with col1:
            # st.subheader(":blue[Select the start date]")
            st.markdown("<h3 style='color: #1D59A9;'>Select the start date</h3>", unsafe_allow_html=True)
            start_date = st.date_input(
                "Start date:",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="start_date_input",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("<h3 style='color: #1D59A9;'>Select the end date</h3>", unsafe_allow_html=True)
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
        st.markdown("<h3 style='color: #1D59A9;'>Select the end date</h3>", unsafe_allow_html=True)
        selected_date = st.date_input(
            "Select the end date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="single_date_input",
            label_visibility="collapsed"
            # value=None
        )
        
        if selected_date not in available_dates:
            st.error("❌ Error: The selected date is not available: Please select another one.")
            return None
        
        return str(selected_date)

def graficos_interactivos(df_metrics, df_prices, stats_to_plot,periodicity):
    """
    Función auxiliar para generar gráficos de barras para métricas
    y gráfico de líneas para precios.
    """
    st.divider()
    
    st.write("### Historical Price Evolution")
    # Normalizamos a base 100 para que la comparación visual sea justa
    prices_norm = (df_prices / df_prices.iloc[0]) * 100

    fig_line = px.line(prices_norm, x=prices_norm.index, y=prices_norm.columns,
                      labels={'value': 'Normalized Price', 'Date': 'Date'},
                      title=f"Price of {periodicity}")
    fig_line.update_traces(connectgaps=True)
    fig_line.update_layout(hovermode="x unified", legend_title="Assets")
    st.plotly_chart(fig_line, width="stretch")

    #Gráfico de Barras para Métricas Seleccionadas
    if stats_to_plot:
        st.write("### Key Metrics Comparison")
        #métricas que el usuario seleccionó y que existen en el df
        available_stats = [s for s in stats_to_plot if s in df_metrics.columns]
        
        if available_stats:
            for stat in available_stats:
                
                fig_bar = px.bar(df_metrics, x=df_metrics.index, y=stat,
                                 text_auto='.2f',
                                 title=f"Metric: {stat}",
                                 color=df_metrics.index,
                                 color_discrete_sequence=px.colors.qualitative.Prism)

                fig_bar.update_layout(
                    xaxis_title="Assets",
                    yaxis_title="Value",
                    legend_title="Assets",
                    bargap=.25,      
                    bargroupgap=.1, 
                    uniformtext_minsize=8, 
                    uniformtext_mode='hide')
                
                st.plotly_chart(fig_bar, width="stretch")

@st.cache_data
def calculus_bmrk(_data):
    df_prices = _data['Prices'].set_index('Date')
    df_weights = _data['Weights'].copy()
    
    if df_weights['Weights'].max() > 1:
        df_weights['Weights'] = df_weights['Weights'] / 100

    dict_bmks_returns = {}

    for bmk_name in df_weights['Benchmark Name'].unique():
        w_sub = df_weights[df_weights['Benchmark Name'] == bmk_name]
        w_pivot = w_sub.pivot_table(
                    index='Start Date', 
                    columns='Component Ticker', 
                    values='Weights', 
                    aggfunc='sum'
                    )   
        
        # Pesos alineados
        w_aligned = w_pivot.reindex(df_prices.index).ffill().shift(1)
        
        # Retornos de componentes sin fillna (mantener NaNs originales)
        comp_tickers = w_pivot.columns
        comp_returns = df_prices[comp_tickers].pct_change() 
        
        # Cálculo del Benchmark:
        dict_bmks_returns[bmk_name] = (comp_returns * w_aligned).sum(axis=1, min_count=1)

    # Retornos con NaNs (Para Volatilidad, Sharpe, Beta, Tracking Error)
    returns_bmrk = pd.DataFrame(dict_bmks_returns)
    
    # Retornos con Zeros (Para Precios e Índices acumulados)
    returns_bmrk_z = returns_bmrk.fillna(0)
    
    #Precios Acumulados (Basados en los retornos con ceros)
    px_last = 100000 * (1 + returns_bmrk_z).cumprod()
    
    return returns_bmrk, px_last, returns_bmrk_z

@st.cache_data
def bmrk_aux_funds(df_funds_bmrk, returns_bmrk, fecha_fin):
    """
    Crea una matriz de retornos de benchmark que 'sigue' a cada fondo 
    según su asignación histórica.
    """
    all_bmrk_mapped = pd.DataFrame(index=returns_bmrk.index)
    
    for ticker in df_funds_bmrk['Ticker'].unique():
        # Historial de benchmarks para este fondo
        hist = df_funds_bmrk[df_funds_bmrk['Ticker'] == ticker].sort_values('Start Date')
        serie_fondo = pd.Series(index=returns_bmrk.index, dtype=float)
        
        for _, row in hist.iterrows():
            bmk = row['Associate Benchmark']
            start = row['Start Date']
            if bmk in returns_bmrk.columns:
                serie_fondo.loc[start:] = returns_bmrk[bmk].loc[start:]
        
        all_bmrk_mapped[ticker] = serie_fondo
        
    return all_bmrk_mapped.loc[:fecha_fin]


#  col1, col2 = st.columns([.4, .6], vertical_alignment="center")
#         # Títulos en columna 1
#         with col1:
#             st.subheader("Año")
#             st.subheader("Mes")
            
#         # Elegir Año y Mes
#         with col2:
#             # Elegir Año
#             año = st.selectbox("Año", años, años.index(max(años)), label_visibility="collapsed")
#             # Filtrar y elegir Meses
#             meses = mesaño.query("Año == @año")["Mes"].unique().tolist()
#             meses.sort()
#             mes = st.selectbox("Mes", meses, meses.index(max(meses)), label_visibility="collapsed")