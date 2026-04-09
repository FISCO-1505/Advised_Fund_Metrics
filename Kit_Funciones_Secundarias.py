import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import io

# librerias para el gráfico
import matplotlib.dates as mdates
import numpy as np
import matplotlib.ticker as mtick
from scipy.interpolate import make_interp_spline

import warnings
warnings.filterwarnings('ignore')

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

@st.cache_resource(show_spinner=False)
def cargar_datos_excel(_file):
    dict_dfs = pd.read_excel(_file, sheet_name=None)
    
    for sheet in dict_dfs:
        dict_dfs[sheet] = dict_dfs[sheet].map(
            lambda x: str(x) if isinstance(x, list) else x
        )
    return dict_dfs

def assets_filter(topic, _data):

    if topic == "Returns Table":
        name = "Returns"
        # Para Returns Table solo mostramos Custom y All
        pills_options = [f"Custom {name}", f"All {name}"]
    else:
        name = "Assets" if topic == "Funds" else "Portfolio"
        # Para los demás mostramos las 3 opciones
        pills_options = [f"Custom {name}", f"All {name}", f"Manager {name}"]

    # Pills
    pills = st.pills(
        label="Options", 
        label_visibility="collapsed",
        options=pills_options,
        default=pills_options[0],
        key=f"pills_{topic}" # Añadimos key única por seguridad
    )

    #Filtrado del DataFrame según el tipo de activo
    if topic == "Funds":
        df_filtered = _data[(_data["Type"] == "Fund") | (_data['Type'] == 'Commodity')]
    elif topic == "Portfolio":
        df_filtered = _data[_data["Type"] == "Portfolio"]
    elif topic == "Returns Table":
        df_filtered = _data[(_data["Type"] == "Portfolio") & (_data['Ticker'].isin(["Port 6 (Conservative)","Port 7", "Port 8 (BALANCED GROWTH)"]))]
        
    
    # Mapeo y lista de nombres
    mapping = dict(zip(df_filtered["Short Name"], df_filtered["Ticker"]))
    assets_names = df_filtered["Short Name"].tolist()
    
    #lista para los fondos y creación de los reportes
    list_mngr_fnds = [
        "Nabucco", "Turandot", "Rothschild Wealth Strategy", "BBVA Strategic Equity",
        "BBVA Absolute Global Trends", "MS Risk Control", "MS Growth"
    ]
    
    assets_selected = []

    if pills == f"Custom {name}":
        assets_selected = []
    
    elif pills == f"All {name}":
        assets_selected = assets_names
        
    elif pills == f"Manager {name}":
        #esta condicional no considera a Returns Table
        if topic == "Funds":
            assets_selected = list_mngr_fnds
        else:
            assets_selected = assets_names[:2]

    return assets_names, assets_selected, mapping

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
            stats_selected=["Cumulative","Vol", "Sharpe Ratio","VaR",
                                  'Treynor Ratio','Sortino Ratio','Info. Ratio',
                                  'Tracking Error', 'Beta','Correlation',
                                  'R^2','Max. Drawdown']
       
    
    elif topic== "Portfolio":
        stats=["PX_LAST","% Change","Cumulative","RF","Vol",
                                "Sharpe Ratio","VaR",'Negative Returns',
                                'Downside Deviation', 'Sortino Ratio',
                                'Drawdown','Max. Drawdown']

        if pills=="Custom Stats":
            stats_selected=None
            
        elif pills=="All Stats":
            stats_selected=stats

        elif pills== "Manager Stats":
            stats_selected=["PX_LAST","% Change","Cumulative","RF","Vol",]
    
    

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

@st.cache_data(show_spinner=False)
def graficos_interactivos(df_metrics, df_prices, stats_to_plot, periodicity, key_suffix=""):
    """
     Función auxiliar para generar gráficos de barras para métricas
     y gráfico de líneas para precios.
    """
    st.divider()
    
    st.write("### Historical Price Evolution")
    prices_norm = (df_prices / df_prices.iloc[0]) * 100
    fig_line = px.line(prices_norm, x=prices_norm.index, y=prices_norm.columns,
                       labels={'value': 'Normalized Price', 'Date': 'Date'},
                       title=f"Price of {periodicity}")
    fig_line.update_traces(connectgaps=True)
    fig_line.update_layout(hovermode="x unified", legend_title="Assets")
    
    st.plotly_chart(fig_line, width="stretch", key=f"line_{key_suffix}")

    if stats_to_plot:
        st.write("### Key Metrics Comparison")
        available_stats = [s for s in stats_to_plot if s in df_metrics.columns]
        
        if available_stats:
            # Tu bucle interno para las barras
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
                    bargap=.25)
                
                st.plotly_chart(fig_bar,width="stretch", key=f"bar_{stat}_{key_suffix}")

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
        comp_returns = df_prices[comp_tickers].ffill().pct_change() 
        
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

# -- Funciones de tablas de rendimientos --
@st.cache_data
def portfolio_Prices(df_prices_funds,df_fixed_portfolios,df_nominals):
    """
    Genera los precios de los portafolios combinando:
    1. Precios fijos históricos (Hoja 'Portfolio Prices').
    2. Cálculo dinámico (Suma Producto) con nominales variables (Hoja 'Nominals').
    """
    dict_portfolios_px = {}

    #es importante ejecutar primero la jerarquia Standar para posteriormente generar
    #los portafolios de portafolios (Combo)
    execution_order = ['Standard', 'Combo']

    for p_type in execution_order:
        #filtramos los portafolios según el tipo de este nivel
        ports_in_level = df_nominals[df_nominals['Type'] == p_type]['Portfolio'].unique()

        for port_name in ports_in_level:
            
            #portafolio actual en el bucle
            df_port_config = df_nominals[df_nominals['Portfolio'] == port_name]
            
            nom_pivot = df_port_config.pivot_table(
                index='Start Date', 
                columns='Ticker', 
                values='Nominal', 
                aggfunc='sum'
            )
            
            # Si es Standard usa los precios de los fondos, si es Combo, usa los ya calculados en dict_portfolios_px
            if p_type == 'Standard':
                source_prices = df_prices_funds
            else:
                # Combinamos precios de fondos y portafolios calculados para que el 
                # en caso de ser portafolio de portafolios (Combo) encuentre sus Tickers de los portafolios
                source_prices = pd.concat([df_prices_funds, pd.DataFrame(dict_portfolios_px)], axis=1)

            # Identificar los tickers que pertenecen al portafolio (ordenados según el pivot)
            comp_tickers = nom_pivot.columns.tolist()
            
            # Filtramos y ordenamos la fuente de precios para que coincida EXACTAMENTE
            #hacemos ffill a los precios de los fondos, para obtener info en el 
            #precio del portafolio
            source_prices_ordered = source_prices.loc[:,comp_tickers].ffill()
            
            # Alineamos los nominales al índice de tiempo de los precios
            nom_aligned = nom_pivot.reindex(source_prices_ordered.index).ffill()
            
            # Multiplicamos la matriz de precios (de fondos o portafolios) por la matriz de nominales
            #considerando que la matriz de precios y de nominales están ordenados por las mismas columnas
            px_dynamic = (source_prices_ordered[comp_tickers] * nom_aligned).sum(axis=1, min_count=1)

            #Integración del nuevo cálculo con el Histórico Fijo
            if port_name in df_fixed_portfolios.columns:
                serie_fija = df_fixed_portfolios[port_name].dropna()
                
                #primera fecha de nominales
                start_calc_dt = nom_pivot.index.min()
                
                #Recortamos la serie fija para que termine un día antes del inicio de los nominales
                serie_fija_trimmed = serie_fija[serie_fija.index < start_calc_dt]
                
                # Histórico + Dinámico (cálculo de suma producto)
                full_series = pd.concat([serie_fija_trimmed, px_dynamic[px_dynamic.index >= start_calc_dt]])
            else:
                # Si no existe en la hoja de fijos, el precio es 100% el cálculo dinámico
                full_series = px_dynamic
                
            dict_portfolios_px[port_name] = full_series

    df_final_px = pd.DataFrame(dict_portfolios_px)
    
    return df_final_px

def generar_grafico_linea_suave(df_port, df_bmrk, nombre_port):
    plt.switch_backend('Agg')

    if nombre_port[5:6] == "6":
        titulo = "Portafolio Conservador (Port-6)"
        subtitulo = "2026 YTD"#cambiar a tomar la fecha año
        legend_indice = "Portafolio Conservador (Port-6)"
        legend_bmrk = "Indice de Referencia (20% Renta Variable/80% Renta Fija)"
    elif nombre_port[5:6] == "7":
        titulo = "Portafolio Balanceado (Port-7)"
        subtitulo = "2026 YTD"#cambiar a tomar la fecha año
        legend_indice = "Portafolio Conservador (Port-7)"
        legend_bmrk = "Indice de Referencia (40% Renta Variable/60% Renta Fija)"

    elif nombre_port[5:6] == "8":
        titulo = "Portafolio Crecimiento Balanceado (Port-8)"
        subtitulo = "2026 YTD"#cambiar a tomar la fecha año
        legend_indice = "Portafolio Crecimiento Balanceado (Port-8)"
        legend_bmrk = "Indice de Referencia (60% Renta Variable/40% Renta Fija)"

    fig, ax = plt.subplots(figsize=(12, 4))

    def suavizar_datos(df):
        x = mdates.date2num(df.index)
        y = df.values*100
        x_new = np.linspace(x.min(), x.max(), 300) 
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_new)
        return x_new, y_smooth

    # 1. Graficar datos suavizados
    x_p, y_p = suavizar_datos(df_port)
    ax.plot(x_p, y_p, color='#203764', label=legend_indice, linewidth=2.2, antialiased=True, zorder=3)
    
    if df_bmrk is not None:
        x_b, y_b = suavizar_datos(df_bmrk)
        ax.plot(x_b, y_b, color='#D3D3D3', label=legend_bmrk, linewidth=1.5, alpha=0.8, zorder=2)

    # 2. Resaltar el eje 0 (Línea de base)
    ax.axhline(0, color='#808080', linewidth=1.2, zorder=1) # Gris medio

    # 3. Formato del Eje Y (Porcentaje con 1 decimal)
    # Si tus datos ya están en formato 10.5 (para 10.5%), usa esto:
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
    # Si tus datos están en 0.105 para 10.5%, usa: mtick.PercentFormatter(1.0, decimals=1)

    # 4. Formato del Eje X (Fechas con inclinación y más frecuencia)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    # Forzamos a que aparezcan más etiquetas (ej. cada 7 días o ajuste automático más denso)
    # ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12)) 

    # Configuramos el localizador para que sea cada 2 días
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    
    # Inclinación ligera de las fechas
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 5. Colores de los ejes y etiquetas en Gris
    color_gris = '#666666'
    ax.tick_params(axis='both', colors=color_gris, labelsize=9) # Números en gris
    ax.spines['bottom'].set_color(color_gris) # Línea del eje X en gris
    ax.spines['left'].set_color(color_gris)   # Línea del eje Y en gris

    # Estética final
    ax.set_title(f"{titulo} <br> {subtitulo}", fontsize=11, fontweight='bold', color='#0073C6', pad=15)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_ti
    
    # Quitar marcos innecesarios
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color=color_gris)

    plt.tight_layout()

    # Guardar en buffer
    imgdata = io.BytesIO()
    fig.savefig(imgdata, format='png', dpi=130, bbox_inches='tight') # bbox_inches evita que se corten las fechas inclinadas
    plt.close(fig)
    imgdata.seek(0)
    
    return imgdata

@st.cache_data
def crear_excel(cols_name,total_portafolio,start_prices, final_prices,
                allocation,port,prices,img_buffer,
                df_ocw,df_dfaf,start_dt_ocw,start_dt_dfaf):
    
    output = io.BytesIO()
    
    if port[5:6] == "6":
        titulo = "PORTAFOLIO CONSERVADOR"
        subtitulo = "(Port-6)"
        tipo_sub = "Conservador"
    elif port[5:6] == "7":
        titulo = "PORTAFOLIO BALANCEADO"
        subtitulo = "(Port-7)"
        tipo_sub = "Balanceado"
    elif port[5:6] == "8":
        titulo = "PORTAFOLIO CRECIMIENTO"
        subtitulo = "BALANCEADO (Port-8)"
        tipo_sub = "Balanceado"

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Portfolio")

        # --- TUS FORMATOS ---
        money = {"num_format": "_-$* #,##0.00_-", "font_name":"Lato Light", "align":"center", "valign":"vcenter"}

        pct = {"num_format": '0.00%', "font_name":"Lato Light","align":"center", "valign":"vcenter"}
        
        header = {"align":"left", "valign":"vcenter", "font_name":"Lato Light",
                "font_size":11,"bg_color":"#FFFFFF", "font_color":"#000000", "bold":True,
                                            "text_wrap":True}
        
        funds = {"align":"left", "valign":"vcenter", "font_name":"Lato Light","font_size":11}
        
        format_text = workbook.add_format({"align":"left", "valign":"vcenter", "font_name":"Lato Light",
                                           "font_size":10, "italic":True})

        format_money = workbook.add_format(money)

        format_money_2 = workbook.add_format({**money, "top": 1, "bottom": 1})

        format_pct = workbook.add_format(pct)

        format_pct_2 = workbook.add_format({**pct, "top": 1, "bottom": 1})

        format_header = workbook.add_format(header)

        format_header_2 = workbook.add_format({**header, "bottom": 1})

        format_funds = workbook.add_format(funds)

        format_funds_2 = workbook.add_format({**funds,"top": 1, "bottom": 1})

        format_title = workbook.add_format({"align":"center", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11, "font_color":"#FFFFFF", "bg_color":"#203764",
                                            "bold":True,"text_wrap": True})

        format_background = workbook.add_format({"align":"right", "valign":"vcenter", "font_name":"Lato Light",
                                                "font_size":11, "font_color":"#000000","bg_color":"#D9D9D9", "bold":True})
        
        format_date =workbook.add_format({"num_format": "dd/mm/yyyy","align": "center","valign": "vcenter","font_name": "Lato Light",
                                        "font_size":11, "font_color":"#FFFFFF", "bg_color":"#203764"})
        
        format_total_pct = workbook.add_format({"num_format": '0.00%', "font_name":"Lato Light",
                                                "align":"center","valign":"vcenter","font_color":"#000000","bg_color":"#D9D9D9", "bold":True})


        # --- DISEÑO ---
        worksheet.hide_gridlines(2)

        for i,fund in enumerate(cols_name):
            # -- Fondos --
            worksheet.write(7+i,3,fund,format_funds)

            # -- Fecha inicio --
            # worksheet.write(6,4,start_prices.index,format_date)
            worksheet.write(7+i,4,start_prices[fund].iloc[0],format_money)

            # -- Fecha fin --
            # worksheet.write(6,5,final_prices.index,format_date)
            worksheet.write(7+i,5,final_prices[fund].iloc[0],format_money)

            # -- Rendimiento --
            # worksheet.write(6,6,final_prices.index,format_date)
            worksheet.write(7+i,6,(final_prices[fund].iloc[0]/start_prices[fund].iloc[0])-1,format_pct)

            # -- Allocation --
            worksheet.write(7+i,8,allocation[fund].iloc[0],format_pct)

        # -- Fecha inicio --
        worksheet.write(6,4,start_prices.index[0],format_date)
        # -- Fecha fin --
        worksheet.write(6,5,final_prices.index[0],format_date)
        # -- Rendimiento --
        worksheet.write(6,6,final_prices.index[0],format_date)
    
        # -- Allocation --
        worksheet.merge_range("I6:I7","% \nAsignación",format_title)

        worksheet.write(5,3, titulo, format_header)
        worksheet.write(6,3, subtitulo, format_header_2)
        
        worksheet.merge_range("E6:F6", "NAV", format_title)
        worksheet.write(5,6, "Rendimiento YTD", format_title)
        
        worksheet.merge_range(f"D{7+len(cols_name)+1}:F{7+len(cols_name)+1}",f"Rendimiento YTD - Perfil {tipo_sub}", format_background)
        worksheet.write(7+len(cols_name),6,total_portafolio,format_total_pct)
        worksheet.write(7+len(cols_name),8,allocation.iloc[0].sum(),format_total_pct)
        worksheet.write(7+len(cols_name)+1,3,"Estos fondos tienen NAVs diarios.",format_text)

        #Fondos alternativos
        worksheet.write(7+len(cols_name)+3,3,"FONDOS ALTERNATIVOS",format_header)
        
        fondos_fijos = [
            {"ticker": "OCWHAUA LX Equity", "nombre": "Wilshire - Hedged Opportunities - USD"},
            {"ticker": "WELSTGD SW EQUITY", "nombre": "Wealth Strategy Gold Fund- USD"},
            {"ticker": "FDAF", "nombre": "Finaccess Digital Assets Fund (FDAF) "}
        ]

        fila_inicio_bloque = 7 + len(cols_name) + 4

        for i, fondo in enumerate(fondos_fijos):
            ticker = fondo["ticker"]
            nombre = fondo["nombre"]
            
            row_fecha = fila_inicio_bloque + (i * 4)
            row_valor = fila_inicio_bloque + (i * 4) + 1
            
            if ticker == "OCWHAUA LX Equity":
                p_min = df_ocw[ticker].loc[start_dt_ocw]
                p_max = df_ocw[ticker].iloc[-1]
                f_min = start_dt_ocw 
                f_max = df_ocw.index[-1]
                
            elif ticker == "FDAF":
                p_min = df_dfaf[ticker].loc[start_dt_dfaf]
                p_max = df_dfaf[ticker].iloc[-1]
                f_min = start_dt_dfaf
                f_max = df_dfaf.index[-1]

            else:
                p_min = prices[ticker].loc[prices.index.min()]
                p_max = prices[ticker].loc[prices.index.max()]
                f_min = prices[ticker].index.min()
                f_max = prices[ticker].index.max()
            
            worksheet.write(row_valor, 3, nombre, format_funds_2)

            worksheet.write(row_fecha, 4, f_min, format_date)
            worksheet.write(row_valor, 4, p_min, format_money_2)

            worksheet.write(row_fecha, 5, f_max, format_date)
            worksheet.write(row_valor, 5, p_max, format_money_2)

            worksheet.write(row_fecha, 6, f_max, format_date)
            worksheet.write(row_valor, 6, (p_max / p_min) - 1, format_pct_2)

        worksheet.write(7+len(cols_name)+6,3,"Este fondo tiene NAV oficial trimestral proporcionado dentro de los 60 días posateriores al fin del trimestre.",format_text)
        worksheet.write(7+len(cols_name)+10,3,"Este fondo tiene NAV diario.",format_text)
        worksheet.write(7+len(cols_name)+14,3,"Este fondo tiene NAV oficial mensual proporcionado dentro de los 15 días posteriores al fin de cada mes.",format_text)

        # # --- INSERTAR EL GRÁFICO ---
        worksheet.insert_image(2, 11, f"graph_{port}.png", {'image_data': img_buffer})

        # Ajustar ancho de columnas
        worksheet.set_column("D:D", 38)
        worksheet.set_column("E:F", 14)
        worksheet.set_column("G:G", 19)
        worksheet.set_column("H:H", 3)
        worksheet.set_column("I:I", 12)
        
    #Extraer los datos del buffer
    data = output.getvalue()
    return data

# -- Funciones de crear excel de las estadísticas --
@st.cache_data
def formato_santander(funds_cmmdty):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Santander")

        # --- TUS FORMATOS ---
        num = workbook.add_format({"num_format": "_-#,##0.00_-", "font_name":"Lato Light", 
                                            "align":"center", "valign":"vcenter"})

        pct = workbook.add_format({"num_format": '0.00%', "font_name":"Lato Light",
                                        "align":"center", "valign":"vcenter"})
        
        header = {"align":"left", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11,"bg_color":"#FFFFFF", "font_color":"#000000", "bold":True,
                                            }
        format_header = workbook.add_format(header)

        format_title = workbook.add_format({"align":"center", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11, "font_color":"#000000", "bg_color":"#DDEBF7",
                                            "bold":True,"text_wrap": True,"bottom": 5,"bottom_color": "#BFBFBF","bold":True})

        format_trama = workbook.add_format({"pattern": 14})

            # --- DISEÑO ---
        
        worksheet.hide_gridlines(2)

        for i in range(15):
             # -- linea de trama --
            worksheet.write(7,i,"",format_trama)

        worksheet.write(0,0,"SAN - Nabucco USD",format_header)
        worksheet.write(2,1,"Statistics",format_header)
        worksheet.write(10,0,"SAN - Turandot USD",format_header)
        worksheet.write(12,1,"Statistics",format_header)

        # funds_cmmdty = funds_cmmdty.loc[["BENIDUI Equity", "BELICUS Equity"]]
        
        for i,stat in enumerate(funds_cmmdty.columns.drop("VaR")):
            # -- Nabucco --
            worksheet.write(1,2+i,stat,format_title)
            if stat in ["Cumulative","Vol","Tracking Error","Max. Drawdown"]:
                worksheet.write(2,2+i,funds_cmmdty[stat].iloc[0],pct)
            else:
                worksheet.write(2,2+i,funds_cmmdty[stat].iloc[0],num)

            # -- Turandot --
            worksheet.write(11,2+i,stat,format_title)
            if stat in ["Cumulative","Vol","Tracking Error","Max. Drawdown"]:
                worksheet.write(12,2+i,funds_cmmdty[stat].iloc[-1],pct)
            else:
                worksheet.write(12,2+i,funds_cmmdty[stat].iloc[-1],num)
            

        # Ajustar ancho de columnas
        worksheet.set_column("A:A", )
        worksheet.set_column("B:B", 10)
        worksheet.set_column("C:C", 15)
        worksheet.set_column("D:N", 13)

    #Extraer los datos del buffer
    data = output.getvalue()
    return data

@st.cache_data
def formato_bbva(funds_cmmdty):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        # -- Hoja BBVA Risk Metricks --
        worksheet1 = workbook.add_worksheet("BBVA Risk Metricks")

        # -- Hoja de inputs --
        worksheet = workbook.add_worksheet("inputs")

        # --- TUS FORMATOS ---
        num = workbook.add_format({"num_format": "_-#,##0.00_-", "font_name":"Lato Light", 
                                            "align":"center", "valign":"vcenter"})

        pct = workbook.add_format({"num_format": '0.00%', "font_name":"Lato Light",
                                        "align":"center", "valign":"vcenter"})
        
        header = {"align":"left", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11,"bg_color":"#FFFFFF", "font_color":"#000000", "bold":True,
                                            }
        
        format_header = workbook.add_format(header)

        format_header_2 = workbook.add_format({"align":"left", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11, "font_color":"#000000", "bold":True, "bottom":1
                                            })
        
        format_header_3 = workbook.add_format({"align":"left", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11, "font_color":"#000000", "bold":True,
                                            })

        format_left = workbook.add_format({**header,'top': 1,'bottom': 1,'left': 1,'align': 'center','bold': True})

        format_right = workbook.add_format({**header,'top': 1,'bottom': 1,'right': 1,'align': 'center','bold': True})
      
        format_title = workbook.add_format({"align":"center", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11, "font_color":"#000000", "bg_color":"#DDEBF7",
                                            "bold":True,"text_wrap": True,"bottom": 5,"bottom_color": "#BFBFBF","bold":True})
        
        format_inferior = workbook.add_format({"bottom":1})

        format_superior = workbook.add_format({"top":1})

        format_trama = workbook.add_format({"pattern": 14})

        #                        --- DISEÑO ---

        #               -- Hoja BBVA Risk Metricks --

        worksheet1.write(1,1,"",format_inferior)
        worksheet1.write(0,2,"BBVA",format_header_3)
        worksheet1.write(0,4,"BBVA",format_header_3)
        worksheet1.write(1,2,"Absolute GT",format_header_2)
        worksheet1.write(1,4,"Strategic Eq",format_header_2)

        worksheet1.write(2,1,"Duration",format_header_3)

        for i,stat in enumerate(funds_cmmdty.columns.drop("VaR")):
            # -- Absolute --
            worksheet1.write(3+i,1,stat,format_header_3)
            if stat in ["Cumulative","Vol","Tracking Error","Max. Drawdown"]:
                worksheet1.write(3+i,2,funds_cmmdty[stat].iloc[0],pct)
            else:
                worksheet1.write(3+i,2,funds_cmmdty[stat].iloc[0],num)

            # -- Strategic --
            if stat in ["Cumulative","Vol","Tracking Error","Max. Drawdown"]:
                worksheet1.write(3+i,4,funds_cmmdty[stat].iloc[-1],pct)
            else:
                worksheet1.write(3+i,4,funds_cmmdty[stat].iloc[-1],num)

        worksheet1.write(2+len(funds_cmmdty.columns),1,"",format_superior)
        worksheet1.write(2+len(funds_cmmdty.columns),2,"",format_superior)
        worksheet1.write(2+len(funds_cmmdty.columns),4,"",format_superior)

        # Ajustar ancho de columnas
        worksheet1.set_column("A:A",3)
        worksheet1.set_column("B:B", 18)
        worksheet1.set_column("C:C",13)
        worksheet1.set_column("D:D",3)
        worksheet1.set_column("E:E",13)


        #                   -- Hoja de inputs --
        worksheet.hide_gridlines(2)

        for i in range(15):
             # -- linea de trama --
            worksheet.write(7,i,"",format_trama)

        worksheet.write(0,0,"BBVA - Absolute USD",format_header)
        worksheet.write(2,1,"Statistics",format_header)
        worksheet.write(10,0,"BBVA - Strategic USD",format_header)
        worksheet.write(12,1,"Statistics",format_header)

        worksheet.write(4,1,"Duration",format_left)
        worksheet.write(4,2,"",format_right)

        worksheet.write(14,1,"Duration",format_left)
        worksheet.write(14,2,"",format_right)

        for i,stat in enumerate(funds_cmmdty.columns.drop("VaR")):
            # -- Absolute --
            worksheet.write(1,2+i,stat,format_title)
            if stat in ["Cumulative","Vol","Tracking Error","Max. Drawdown"]:
                worksheet.write(2,2+i,funds_cmmdty[stat].iloc[0],pct)
            else:
                worksheet.write(2,2+i,funds_cmmdty[stat].iloc[0],num)

            # -- Strategic --
            worksheet.write(11,2+i,stat,format_title)
            if stat in ["Cumulative","Vol","Tracking Error","Max. Drawdown"]:
                worksheet.write(12,2+i,funds_cmmdty[stat].iloc[-1],pct)
            else:
                worksheet.write(12,2+i,funds_cmmdty[stat].iloc[-1],num)
            

        # Ajustar ancho de columnas
        worksheet.set_column("A:A", )
        worksheet.set_column("B:B", 10)
        worksheet.set_column("C:C", 15)
        worksheet.set_column("D:N", 13)

    #Extraer los datos del buffer
    data = output.getvalue()
    return data

@st.cache_data
def formato_morgan_stanley(funds_cmmdty):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        titulo=None
        if funds_cmmdty.index == "MSHRCZU Equity":
            titulo = "MS Risk Control USD"
        else:
            titulo = "MS Growth USD"

        worksheet = workbook.add_worksheet(titulo[:-4])

        # --- TUS FORMATOS ---
        num = workbook.add_format({"num_format": "_-#,##0.00_-", "font_name":"Lato Light", 
                                            "align":"center", "valign":"vcenter"})

        pct = workbook.add_format({"num_format": '0.00%', "font_name":"Lato Light",
                                        "align":"center", "valign":"vcenter"})
        
        header = {"align":"left", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11,"bg_color":"#FFFFFF", "font_color":"#000000", "bold":True,}
        
        format_header = workbook.add_format(header)

        format_title = workbook.add_format({"align":"center", "valign":"vcenter", "font_name":"Lato Light",
                                            "font_size":11, "font_color":"#000000", "bg_color":"#DDEBF7",
                                            "bold":True,"text_wrap": True,"bottom": 5,"bottom_color": "#BFBFBF","bold":True})

            # --- DISEÑO ---
        
        worksheet.hide_gridlines(2)

        worksheet.write(0,0,titulo,format_header)
        worksheet.write(2,1,"Statistics",format_header)
        
        for i,stat in enumerate(funds_cmmdty.columns.drop("VaR")):
            
            worksheet.write(1,2+i,stat,format_title)
            if stat in ["Cumulative","Vol","Tracking Error","Max. Drawdown"]:
                worksheet.write(2,2+i,funds_cmmdty[stat].iloc[0],pct)
            else:
                worksheet.write(2,2+i,funds_cmmdty[stat].iloc[0],num)

            
        # Ajustar ancho de columnas
        worksheet.set_column("A:A", )
        worksheet.set_column("B:B", 10)
        worksheet.set_column("C:C", 15)
        worksheet.set_column("D:N", 13)

    #Extraer los datos del buffer
    data = output.getvalue()
    return data,titulo

@st.cache_data
def formato_rothschild(funds_cmmdty):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Rothschild Risk Stats")
        worksheet.hide_gridlines(2)

        # --- DEFINICIÓN DE FORMATOS ---
        base_font = {"font_name": "Lato Light", "font_size": 11}
        
        fmt_yellow_hdr = workbook.add_format({**base_font, "bg_color": "#FFFF00", "bold": True, "align": "left"})
        
        fmt_section = workbook.add_format({**base_font, "bold": True, "top": 1})

        fmt_section_2 = workbook.add_format({**base_font, "bold": True, "top": 2, "top_color": "#F4B084"})

        fmt_red_row = workbook.add_format({**base_font, "bg_color": "#FF0000", "bold": False})
        
        num = workbook.add_format({**base_font, "num_format": "_-#,##0.00_-", "align": "center"})
        pct = workbook.add_format({**base_font, "num_format": "0.00%", "align": "center"})
        pct_2 = workbook.add_format({**base_font, "num_format": "0.00%", "align": "center", "bottom":1})
        
        label = workbook.add_format({**base_font, "align": "left"})

        # --- ESCRITURA DE DATOS ---

        worksheet.write("B3", "USD", fmt_yellow_hdr)
        worksheet.write("E3", "USD Spanish", fmt_yellow_hdr)
        worksheet.write("C3", "", fmt_yellow_hdr)
        worksheet.write("F3", "", fmt_yellow_hdr)
        
        worksheet.write("B6", "Return", fmt_section_2)
        worksheet.write("E6", "Retorno", fmt_section_2)
        worksheet.write("C6", "", fmt_section_2)
        worksheet.write("F6", "", fmt_section_2)
        
        worksheet.write("B7", "Total Return", label)
        worksheet.write("E7", "Retorno total", label)

        worksheet.write("C7", "", workbook.add_format({**base_font, "num_format": "0.00%", "align": "center","bottom":1}))
        worksheet.write("F7", "", workbook.add_format({**base_font, "num_format": "0.00%", "align": "center","bottom":1}))

        worksheet.write("B8", "Risk", fmt_section)
        worksheet.write("E8", "Riesgo", fmt_section)
        
        worksheet.write("B9", "Standard Deviation (Annualized)", label)
        worksheet.write("E9", "Desviación Estándar (Anualizado)", label)
        worksheet.write("C9", funds_cmmdty.get("Vol", [0])[0], pct)
        worksheet.write("F9", funds_cmmdty.get("Vol", [0])[0], pct)
        
        worksheet.write("B10", "VaR 95% (Daily)", label)
        worksheet.write("E10", "VaR 95% (Diario ex-post)", label)
        worksheet.write("C10", funds_cmmdty.get("VaR", [0])[0], pct)
        worksheet.write("F10", funds_cmmdty.get("VaR", [0])[0], pct)
        
        worksheet.write("B11", "Tracking Error (Annualized)", label)
        worksheet.write("E11", "Tracking Error (Anualizado)", label)
        worksheet.write("C11", funds_cmmdty.get("Tracking Error", [0])[0], pct_2)
        worksheet.write("F11", funds_cmmdty.get("Tracking Error", [0])[0], pct_2)

        worksheet.write("B12", "Risk/Return", fmt_section)
        worksheet.write("E12", "Riesgo/Retorno", fmt_section)
        
        metrics = [
            ("Sharpe Ratio", "Ratio Sharpe", "num"),
            ("Information Ratio", "Information Ratio", "num"),
            ("Sortino Ratio", "Sortino Ratio", "num"),
            ("Treynor Measure", "Medida Treynor", "num"),
            ("Beta (ex-post)", "Beta (ex-post)", "num"),
            ("Correlation", "Correlación", "num"),
            ("R^2", "R^2", "num"),
        ]

        row = 12
        for eng, esp, fmt_type in metrics:
            worksheet.write(row, 1, eng, label)
            worksheet.write(row, 4, esp, label)
            
            row += 1

        #valores de Risk/Return
        worksheet.write("C13", funds_cmmdty.get("Sharpe Ratio", [0])[0], num)
        worksheet.write("F13", funds_cmmdty.get("Sharpe Ratio", [0])[0], num)
        worksheet.write("C14", funds_cmmdty.get("Info. Ratio", [0])[0], num)
        worksheet.write("F14", funds_cmmdty.get("Info. Ratio", [0])[0], num)
        worksheet.write("C15", funds_cmmdty.get("Sortino Ratio", [0])[0], num)
        worksheet.write("F15", funds_cmmdty.get("Sortino Ratio", [0])[0], num)
        worksheet.write("C16", funds_cmmdty.get("Treynor Ratio", [0])[0], num)
        worksheet.write("F16", funds_cmmdty.get("Treynor Ratio", [0])[0], num)
        worksheet.write("C17", funds_cmmdty.get("Beta", [0])[0], num)
        worksheet.write("F17", funds_cmmdty.get("Beta", [0])[0], num)
        worksheet.write("C18", funds_cmmdty.get("Correlation", [0])[0], num)
        worksheet.write("F18", funds_cmmdty.get("Correlation", [0])[0], num)
        worksheet.write("C19", funds_cmmdty.get("R^2", [0])[0], num)
        worksheet.write("F19", funds_cmmdty.get("R^2", [0])[0], num)

        #Fila Especial: DURATION (Fila 19 - Resaltada en Rojo)
        worksheet.write(19, 1, "Duration", fmt_red_row)
        worksheet.write(19, 2, "", fmt_red_row)
        worksheet.write(19, 4, "Duration", fmt_red_row)
        worksheet.write(19, 5, "", fmt_red_row)

        # MAX DRAWDOWN y Cierre (Fila 20)
        worksheet.write(20, 1, "Max Drawdown             (S.I)", label)
        worksheet.write(20, 4, "Caída Maxima               (S.I)", label)
        worksheet.write("C21", funds_cmmdty.get("Max. Drawdown", [0])[0], pct)
        worksheet.write("F21", funds_cmmdty.get("Max. Drawdown", [0])[0], pct)
        
        # Línea final inferior
        worksheet.write(21, 1, "", workbook.add_format({"top": 1}))
        worksheet.write(21, 2, "", workbook.add_format({"top": 1}))
        worksheet.write(21, 4, "", workbook.add_format({"top": 1}))
        worksheet.write(21, 5, "", workbook.add_format({"top": 1}))

        # --- CONFIGURACIÓN DE COLUMNAS ---
        worksheet.set_column("A:A", 2)
        worksheet.set_column("B:B", 35)
        worksheet.set_column("C:C", 10)
        worksheet.set_column("D:D", 5)
        worksheet.set_column("E:E", 35)
        worksheet.set_column("F:F", 10)

    data = output.getvalue()
    return data

def generar_excel_fondos(assets,fnds_cmmdty,fecha_fin,periodo):
    if assets is None:
        assets = []

    # Mapeo de entidades a sus funciones de diseño
    config_fondos = {
        "Santander": {
            "tickers": ["BENIDUI Equity", "BELICUS Equity"],
            "diseno": formato_santander,
            "tipo": "agrupado"
        },
        "BBVA": {
            "tickers": ["BBSALIU Equity", "BBAGTIU Equity"],
            "diseno": formato_bbva,
            "tipo": "agrupado"
        },
        "Morgan Stanley": {
            "tickers": ["MSHRCZU Equity", "MSHZUSD Equity"],
            "diseno": formato_morgan_stanley,
            "tipo": "individual"
        },
        "Rothschild": {
            "tickers": ["RWMWICU Equity"],
            "diseno": formato_rothschild,
            "tipo": "agrupado"
        }
    }

    for entidad, info in config_fondos.items():
        activos_presentes = list(set(info["tickers"]).intersection(set(assets)))
        # st.write(activos_presentes)
        if activos_presentes:
            if info["tipo"] == "agrupado":
                # Generamos el excel pasando la función de diseño correspondiente
                excel_data = info["diseno"](fnds_cmmdty.loc[info["tickers"]])

                crear_boton(entidad, excel_data,fecha_fin,periodo)
                pass
            else:
                for ticker in activos_presentes:
                    excel_data, titulo = info["diseno"](fnds_cmmdty.loc[[ticker]])
                    crear_boton(titulo, excel_data,fecha_fin,periodo)
                    pass

def crear_boton(nombre, data,fecha_fin=None,periodo=None):
    st.download_button(
        label=f"Descargar Reporte {nombre}",
        data=data,
        file_name=f"{nombre} {fecha_fin} {periodo}.xlsx" if fecha_fin else f"{nombre} {periodo}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"btn_{nombre}"
    )

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