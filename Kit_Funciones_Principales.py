
import pandas as pd
import streamlit as st
import numpy as np
import Kit_Funciones_Secundarias as kit_f_secundarias
import Kit_Metricas as kit_metricas

def Funds_Commodity(_data, fecha_fin, periodicity=None,stats=None,assets=None,grafico=None,ticker_map=None,c_d=None):
    df_info = _data['Info']
    index_cols = df_info[df_info['Type'].isin(['Fund', 'Commodity'])]['Ticker'].tolist()

    df_prices = _data['Prices'].set_index('Date')
    df_ocw = _data['OCWHAUA LX Equity'].set_index('Date')
    df_dfaf = _data["FDAF"].set_index("Date")

    df_prices = df_prices.join(df_ocw, how='left')
    df_prices = df_prices.join(df_dfaf, how="left")

    df_prices_all = df_prices[index_cols]

    #ffill para ocw y dfaf con mask de welstg
    df_prices_all['OCWHAUA LX Equity'] = df_prices_all['OCWHAUA LX Equity'].ffill()
    df_prices_all['OCWHAUA LX Equity'] = df_prices_all['OCWHAUA LX Equity'].mask(df_prices_all['WELSTGD SW EQUITY'].isna())
    
    df_prices_all['FDAF'] = df_prices_all['FDAF'].ffill()
    df_prices_all['FDAF'] = df_prices_all['FDAF'].mask(df_prices_all['WELSTGD SW EQUITY'].isna())
    
    # Benchmarks
    returns_bmrk_pure, _, _ = kit_f_secundarias.calculus_bmrk(_data)
    returns_aux_all = kit_f_secundarias.bmrk_aux_funds(_data['Funds - BMRK'], returns_bmrk_pure, fecha_fin)

    # Definición de Fecha de Inicio según Periodicidad
    fecha_inicio_aux = kit_f_secundarias.start_dt(fecha_fin, periodicity,c_d)

    if fecha_inicio_aux == "INSUFFICIENT_DATA":
        st.warning(f"Insufficient historical data for {periodicity}. "
                f"Records start on: {pd.to_datetime('2015-12-03').date()}")
        st.stop()
    else:
        # Lógica de asignación final
        if periodicity == 'Since Inception':

            fecha_inicio = df_prices.index.min()

        else:

            idx_start = df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
            idx_start_prices = df_prices.index.get_indexer([fecha_inicio_aux], method='pad')[0]
            fecha_inicio = df_prices.index[idx_start]
            # fecha_inicio_precios = df_prices.index[idx_start_proces]

        
        st.success(f"Analysis period: {fecha_inicio.date()} to {pd.to_datetime(fecha_fin).date()}")


    #recorte del periodo seleccionado
    df_prices_RF=_data['Prices'].set_index('Date')
    df_prices_RF=df_prices_RF.loc[fecha_inicio:fecha_fin]

    #precios para el cálculo de los retornos
    prices_to_returns = df_prices_all.loc[:fecha_fin]
    returns_2,fechas_reales = kit_metricas.df_returns(prices_to_returns)

    #precios filtrados para graficar
    prices = df_prices_all.loc[fecha_inicio:fecha_fin]

    returns = returns_2.loc[fecha_inicio: fecha_fin]
    
    returns_z=returns.fillna(0)
    returns_aux = returns_aux_all.loc[fecha_inicio:fecha_fin]
     
    #_____________________________________________
    
    df_px_full = prices.ffill()

    # --- Retornos ---
    df_cumm_full = kit_metricas.cumm_return(returns_z) 
    df_rf_full = kit_metricas.RF(df_prices_RF)
    df_neg_full = kit_metricas.negative_returns(returns)

    # Metricas de riesgo y desempeño
    df_vol_full = kit_metricas.rolling_vol(returns, fechas_reales) 
    df_sharpe_full = kit_metricas.sharpe_ratio(returns, df_rf_full, df_vol_full, fechas_reales)
    df_dd_full = kit_metricas.downside_deviation(df_neg_full, fechas_reales)
    df_sortino = kit_metricas.Sortino(returns, df_rf_full, df_dd_full, fechas_reales)
    df_var_full = kit_metricas.VaR(returns, fechas_reales, conf_lvl=99)

    # -- Metricas que necesitan del benchmark --
    df_beta = kit_metricas.Beta(returns, returns_aux, fechas_reales)
    df_treynor = kit_metricas.Treynor_ratio(returns, df_rf_full, df_beta, fechas_reales)
    df_act_retrns=kit_metricas.daily_active_returns(returns,returns_aux,fechas_reales)
    df_te = kit_metricas.Tracking_Error(returns, returns_aux,fechas_reales)
    df_ir = kit_metricas.info_ratio(returns, returns_aux, df_te,fechas_reales)
    df_corr = kit_metricas.correlation(returns, returns_aux)
    df_R2=pd.DataFrame(df_corr**2).rename(index={"Correlation": 'R^2'})
    df_alpha = kit_metricas.Alpha(df_cumm_full, df_beta, kit_metricas.cumm_return(returns_aux), fechas_reales)
    df_jalpha = kit_metricas.J_Alpha(returns, df_rf_full, df_beta, returns_aux, fechas_reales)

    # -- Metricas Drowdown --
    df_dd = kit_metricas.Drawdown(returns, fechas_reales)
    df_mdd = kit_metricas.Max_Drawdown(returns, fechas_reales)


    fnds_cmmdty = pd.DataFrame(index=prices.columns)

    for fund in prices.columns:
        fecha = fechas_reales.get(fund)
        if pd.notnull(fecha):
            # Precios y retornos 
            fnds_cmmdty.at[fund, 'PX_LAST'] = df_px_full.loc[fecha, fund]
            fnds_cmmdty.at[fund, '% Change'] = returns.loc[fecha, fund]
            fnds_cmmdty.at[fund, 'Cumulative'] = df_cumm_full.loc[fecha, fund]
            fnds_cmmdty.at[fund, 'Negative Returns'] = df_neg_full.loc[fecha,fund]

            #Tasa libre de riesgo
            fnds_cmmdty.at[fund, 'RF'] =df_rf_full.loc[fecha]

            #Metricas de riesgo
            fnds_cmmdty.at[fund, 'Vol'] = df_vol_full.loc['Vol', fund]
            fnds_cmmdty.at[fund, 'Sharpe Ratio'] = df_sharpe_full.loc['Sharpe Ratio', fund]
            fnds_cmmdty.at[fund, 'Sortino Ratio'] = df_sortino.at['Sortino Ratio', fund]
            fnds_cmmdty.at[fund, 'Downside Deviation'] = df_dd_full.at['Downside Deviation', fund]
            fnds_cmmdty.at[fund, 'VaR'] = df_var_full.at['VaR 99%', fund]
            
            # #Métricas de Benchmark
            fnds_cmmdty.at[fund, 'Beta'] = df_beta.at['Beta', fund]
            fnds_cmmdty.at[fund, 'Correlation'] = df_corr.at['Correlation', fund]
            fnds_cmmdty.at[fund, 'R^2'] = df_R2.loc['R^2', fund]
            fnds_cmmdty.at[fund, 'Alpha'] = df_alpha.at['Alpha', fund]
            fnds_cmmdty.at[fund, 'J Alpha'] = df_jalpha.at['Jensen Alpha', fund]
            fnds_cmmdty.at[fund, 'Treynor Ratio'] = df_treynor.at['Treynor Ratio', fund]
            fnds_cmmdty.at[fund, 'Daily Active Return'] =df_act_retrns.loc[fecha,fund]
            fnds_cmmdty.at[fund, 'Tracking Error'] = df_te.at['Tracking Error', fund]
            fnds_cmmdty.at[fund, 'Info. Ratio'] = df_ir.at['Information Ratio', fund]
            
            #Drawdowns
            fnds_cmmdty.at[fund, 'Drawdown'] = df_dd.at['Drawdown', fund]
            fnds_cmmdty.at[fund, 'Max. Drawdown'] = df_mdd.at['Max Drawdown', fund]

            fnds_cmmdty.at[fund, 'Real Date'] = fecha
        else:
            # Si el fondo no tiene datos, llenamos con NaN
            fnds_cmmdty.at[fund, 'Real Date'] = pd.NaT
    

    formatos = {
        "PX_LAST": "${:,.3f}", #NO aplica para graficar
        "% Change": "{:.2%}", #NO aplica para graficar
        "Cumulative": "{:.2%}",
        "Negative Returns": "{:.2%}", #NO aplica para graficar
        "RF": "{:.2%}", #NO aplica para graficar
        "Alpha": "{:.2%}",
        "J Alpha": "{:.2%}",
        "Daily Active Return": "{:.2%}", #NO aplica para graficar
        "Vol": "{:.2%}",
        "Downside Deviation": "{:.2%}",
        "VaR": "{:.2%}",
        "Tracking Error": "{:.2%}",
        "Drawdown": "{:.2%}", #NO aplica para graficar
        "Max. Drawdown": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Treynor Ratio": "{:.2f}",
        "Info. Ratio": "{:.2f}",
        "Beta": "{:.2f}",
        "Correlation": "{:.2f}",
        "R^2": "{:.2f}",
        "Real Date": "{:%Y-%m-%d}"
        }
    
    # --- APARTADO DE GRÁFICOS ---
    if stats is not None and assets is not None and grafico:
        kit_f_secundarias.graficos_interactivos(fnds_cmmdty.loc[assets], prices[assets], stats,periodicity,
                                                ticker_map,formatos,returns[assets],"Funds")
    
    return fnds_cmmdty,formatos

def Portfolio(_data, fecha_fin, periodicity=None, stats=None, portfolios=None,grafico=None,ticker_map=None,c_d=None):
    """
    Genera la serie de precios combinada para los portafolios seleccionados.
    
    _data: Diccionario con las hojas 'Portfolio Prices', 'Prices', 'Nominals'.
    selected_portfolios: Lista de strings con los nombres de los portafolios.
    fecha_fin: Fecha de corte del análisis.
    """

    df_prices = _data['Prices'].set_index('Date')
    df_ocw = _data['OCWHAUA LX Equity'].set_index('Date')
    df_dfaf = _data["FDAF"].set_index("Date")

    df_prices_funds = df_prices.join(df_ocw, how='left')
    df_prices_funds = df_prices_funds.join(df_dfaf, how="left")

    #ffill para ocw y dfaf con mask de welstg
    df_prices_funds['OCWHAUA LX Equity'] = df_prices_funds['OCWHAUA LX Equity'].ffill()
    df_prices_funds['OCWHAUA LX Equity'] = df_prices_funds['OCWHAUA LX Equity'].mask(df_prices_funds['WELSTGD SW EQUITY'].isna())
    
    df_prices_funds['FDAF'] = df_prices_funds['FDAF'].ffill()
    df_prices_funds['FDAF'] = df_prices_funds['FDAF'].mask(df_prices_funds['WELSTGD SW EQUITY'].isna())
    

    df_fixed_portfolios = _data['Portfolio Prices'].set_index('Date')
    df_nominals = _data['Nominals'].copy()

    # Precios de los portafolios
    df_prices_all=kit_f_secundarias.portfolio_Prices(df_prices_funds,df_fixed_portfolios,df_nominals)

    # Definición de Fecha de Inicio según Periodicidad
    fecha_inicio_aux = kit_f_secundarias.start_dt(fecha_fin, periodicity,c_d)

    if fecha_inicio_aux == "INSUFFICIENT_DATA":
        st.warning(f"Insufficient historical data for {periodicity}. "
                f"Records start on: {pd.to_datetime('2015-12-03').date()}")
        st.stop()
    else:
        # Lógica de asignación final
        if periodicity == 'Since Inception':

            fecha_inicio = df_prices.index.min()

        else:

            idx_start = df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
            fecha_inicio = df_prices.index[idx_start]
        
        st.success(f"Analysis period: {fecha_inicio.date()} to {pd.to_datetime(fecha_fin).date()}")

    #recorte del periodo seleccionado
    df_prices_RF=_data['Prices'].set_index('Date')
    df_prices_RF=df_prices_RF.loc[fecha_inicio:fecha_fin]

    prices_to_returns = df_prices_all.loc[:fecha_fin]

    returns,fechas_reales = kit_metricas.df_returns(prices_to_returns)
    returns = returns.loc[fecha_inicio:fecha_fin]
    prices = df_prices_all.loc[fecha_inicio:fecha_fin]

    #returns con NaN para el cálculo de las métricas
    returns=returns.replace(0,np.nan)
    
    returns_z=returns.fillna(0)
    
    df_px_full = prices.ffill()

    # --- Retornos ---
    df_cumm_full = kit_metricas.cumm_return(returns_z) 
    df_rf_full = kit_metricas.RF(df_prices_RF)
    df_neg_full = kit_metricas.negative_returns(returns)

    # Metricas de riesgo y desempeño
    df_vol_full = kit_metricas.rolling_vol(returns, fechas_reales) 
    df_sharpe_full = kit_metricas.sharpe_ratio(returns, df_rf_full, df_vol_full, fechas_reales)
    df_dd_full = kit_metricas.downside_deviation(df_neg_full, fechas_reales)
    df_sortino = kit_metricas.Sortino(returns, df_rf_full, df_dd_full, fechas_reales)
    df_var_full = kit_metricas.VaR(returns, fechas_reales, conf_lvl=99)

    # -- Metricas Drowdown --
    df_dd = kit_metricas.Drawdown(returns, fechas_reales)
    df_mdd = kit_metricas.Max_Drawdown(returns, fechas_reales)


    final_portfolios = pd.DataFrame(index=prices.columns)

    for fund in prices.columns:
        fecha = fechas_reales.get(fund)
        if pd.notnull(fecha):
            # Precios y retornos 
            final_portfolios.at[fund, 'PX_LAST'] = df_px_full.loc[fecha, fund]
            final_portfolios.at[fund, '% Change'] = returns.loc[fecha, fund]
            final_portfolios.at[fund, 'Cumulative'] = df_cumm_full.loc[fecha, fund]
            final_portfolios.at[fund, 'Negative Returns'] = df_neg_full.loc[fecha,fund]

            #Tasa libre de riesgo
            final_portfolios.at[fund, 'RF'] =df_rf_full.loc[fecha]

            #Metricas de riesgo
            final_portfolios.at[fund, 'Vol'] = df_vol_full.loc['Vol', fund]
            final_portfolios.at[fund, 'Sharpe Ratio'] = df_sharpe_full.loc['Sharpe Ratio', fund]
            final_portfolios.at[fund, 'Sortino Ratio'] = df_sortino.at['Sortino Ratio', fund]
            final_portfolios.at[fund, 'Downside Deviation'] = df_dd_full.at['Downside Deviation', fund]
            final_portfolios.at[fund, 'VaR'] = df_var_full.at['VaR 99%', fund]
            
            #Drawdowns
            final_portfolios.at[fund, 'Drawdown'] = df_dd.at['Drawdown', fund]
            final_portfolios.at[fund, 'Max. Drawdown'] = df_mdd.at['Max Drawdown', fund]

            final_portfolios.at[fund, 'Real Date'] = fecha
        else:
            # Si el fondo no tiene datos, llenamos con NaN
            final_portfolios.at[fund, 'Real Date'] = pd.NaT
    
    #formatos según la métrica
    formatos = {
    "PX_LAST": "${:,.3f}",
    "% Change": "{:.2%}",
    "Cumulative": "{:.2%}",
    "Negative Returns": "{:.2%}",
    "RF": "{:.2%}",         
    "Vol": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Sortino Ratio": "{:.2f}",
    "Downside Deviation": "{:.2%}",
    "VaR": "{:.2%}",
    "Drawdown": "{:.2%}",
    "Max. Drawdown": "{:.2%}",
    "Real Date": "{:%Y-%m-%d}"}

    # --- APARTADO DE GRÁFICOS ---
    if stats is not None and portfolios is not None and grafico:
        
        kit_f_secundarias.graficos_interactivos(final_portfolios.loc[portfolios], prices[portfolios], stats,periodicity,
                                                ticker_map,formatos,returns[portfolios],"Portfolios")

    return final_portfolios, formatos

def procesar_analisis(topic, data, selection, stats, assets,ticker_map):
    """
    Maneja la lógica de fechas, comparativas y carga de métricas 
    para Funds y Portfolios de forma unificada.
    """
    start_date = None
    selected_date = None
    
    if selection == "Custom Date":
        start_date, selected_date = kit_f_secundarias.calendar(data["Prices"]["Date"], mode="range")
    else:
        selected_date = kit_f_secundarias.calendar(data["Prices"]["Date"], mode="single")

    #Comparativa
    toggle_button = st.toggle("Comparative", key=f"toggle_{topic}_{selection}")
    grafico= True if toggle_button else None
    
    # -- session state para los botones de los fondos --
    current_params = f"{selected_date}_{sorted(assets)}"
    
    if "last_params_2" not in st.session_state:
        st.session_state.last_params_2 = current_params
        st.session_state.cargar_tabla_2 = False

    # Si los parámetros cambian, apagamos la tabla
    if st.session_state.last_params_2 != current_params:
        st.session_state.cargar_tabla_2 = False
        st.session_state.last_params_2 = current_params
    
    # Botón de Carga
    if st.button("Load Process", key=f"btn_{topic}_{selection}_{toggle_button}"):
        st.session_state.cargar_tabla_2 = True

    if st.session_state.cargar_tabla_2:

        # --- validación ---
        if not assets:
            st.warning("⚠️ Please select at least one asset/portfolio to continue.")
            return
        
        if not stats:
            st.warning("⚠️ Please select at least one metric to calculate.")
            return
        # ----------------------------

        with st.container():
            func_principal = (Funds_Commodity if topic == "Funds" else Portfolio)
            
            if selection == "Custom Date":
                results, formatos = func_principal(data, selected_date, selection, stats, assets,grafico,ticker_map, start_date)
                periodo_excel=f"{start_date} to {selected_date}"
                fecha_excel=None

            else:
                results, formatos = func_principal(data, selected_date, selection, stats, assets,grafico,ticker_map)
                periodo_excel="SI" if selection == "Since Inception" else selection
                fecha_excel=selected_date

            cols_to_show = [c for c in (stats + ['Real Date']) if c in results.columns]
            final_df = results[cols_to_show].loc[assets]
            
            #se cambia el nombre de ticker al nombre del fondo en la tabla final
            map_names = {v: k for k, v in ticker_map.items()}
            final_df = final_df.rename(index=map_names)

            st.dataframe(final_df.style.format(formatos, na_rep="-"))
            
            # Generar excel
            kit_f_secundarias.generar_excel_fondos(assets,results[stats],fecha_excel,periodo_excel) 
            st.success("You can download the Reports!")

def tabla_rendimientos(_data,fecha_fin,portfolio_select,periodicity="YTD"):
    df_prices = _data['Prices'].set_index('Date')
    df_ocw = _data['OCWHAUA LX Equity'].set_index('Date')
    df_dfaf = _data["FDAF"].set_index("Date")

    df_prices_funds = df_prices.join(df_ocw, how='left')
    df_prices_funds = df_prices_funds.join(df_dfaf, how="left")

    #ffill para ocw y dfaf con mask de welstg
    df_prices_funds['OCWHAUA LX Equity'] = df_prices_funds['OCWHAUA LX Equity'].ffill()
    df_prices_funds['OCWHAUA LX Equity'] = df_prices_funds['OCWHAUA LX Equity'].mask(df_prices_funds['WELSTGD SW EQUITY'].isna())
    
    df_prices_funds['FDAF'] = df_prices_funds['FDAF'].ffill()
    df_prices_funds['FDAF'] = df_prices_funds['FDAF'].mask(df_prices_funds['WELSTGD SW EQUITY'].isna())
    

    df_fixed_portfolios = _data['Portfolio Prices'].set_index('Date')
    df_nominals = _data['Nominals'].copy()

    #returns de los benchamrks, para poder usar en los gráficos 
    _, _, returns_bmrk_z = kit_f_secundarias.calculus_bmrk(_data)

    # Precios de los portafolios
    df_portfolio_prices=kit_f_secundarias.portfolio_Prices(df_prices_funds,df_fixed_portfolios,df_nominals)
    
    # Definición de Fecha de Inicio según Periodicidad
    fecha_inicio_aux = pd.Timestamp(year=pd.to_datetime(fecha_fin).year - 1, month=12, day=31) 
    start_dt_ocw_aux = pd.Timestamp(year=pd.to_datetime(df_ocw.index[-1]).year - 1, month=12, day=31) 
    start_dt_dfaf_aux = pd.Timestamp(year=pd.to_datetime(df_dfaf.index[-1]).year - 1, month=12, day=31)

    if fecha_inicio_aux == "INSUFFICIENT_DATA":
        st.warning(f"Insufficient historical data for {periodicity}. "
                f"Records start on: {pd.to_datetime('2015-12-03').date()}")
        st.stop()
    else:
        # Lógica de asignación final
        if periodicity == 'Since Inception':

            fecha_inicio = df_prices.index.min()

        else:

            idx_start = df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
            fecha_inicio = df_prices.index[idx_start]

            #se extraer las fechas inicios de OCW y DFAF para lenar la tabla de excel
            idx_start_ocw = df_ocw.index.get_indexer([start_dt_ocw_aux], method='backfill')[0]
            idx_start_fdaf = df_dfaf.index.get_indexer([start_dt_dfaf_aux], method='backfill')[0]
            start_dt_ocw = df_ocw.index[idx_start_ocw]
            start_dt_dfaf = df_dfaf.index[idx_start_fdaf]

        
        st.success(f"Analysis period: {fecha_inicio.date()} to {pd.to_datetime(fecha_fin).date()}")

    #precios filtrados de los fondos
    prices = df_prices_funds.loc[fecha_inicio:fecha_fin]
    
    #extraccion de los bnchmarkas asociados a cada portafolio
    df_info=_data["Info"]
    df_filt_bmrk_port = df_info.loc[df_info['Ticker'].isin(portfolio_select), ['Ticker', 'Associate Benchmark']]
    
    ##retornos acumulados y filtrados de los portafolios y de los benchmarks para el gráfico
    # -- bmrk acumulado y filtrado --
    returns_bmrk_z = returns_bmrk_z.loc[fecha_inicio:fecha_fin]
    accm_bmrk = kit_metricas.cumm_return(returns_bmrk_z)
    accm_bmrk = accm_bmrk[df_filt_bmrk_port["Associate Benchmark"].to_list()]
    
    # -- portafolio acumulado y filtrado --
    returns_port,_ = kit_metricas.df_returns(df_portfolio_prices)
    
    #returns con NaN para el cálculo del acumulado
    returns_port=returns_port.replace(0,np.nan)
    returns_port_z=returns_port.fillna(0)
    returns_port_z = returns_port_z.loc[fecha_inicio:fecha_fin]
    accm_port = kit_metricas.cumm_return(returns_port_z)
    accm_port = accm_port[portfolio_select]

    #extrear la relacion de cada portafolio con su benchmark
    dict_relation= dict(zip(df_filt_bmrk_port['Ticker'], df_filt_bmrk_port['Associate Benchmark']))

    #diccionario de los portafolios y benchmarks
    dict_grafico={
        "Portafolio" : accm_port,
        "Benchmark" : accm_bmrk
        }

    with st.container():
        st.write("### Reports generated")

        for port in portfolio_select:

            current_nominals=df_nominals[df_nominals["Portfolio"]==port]
            df_curnt_nominals=current_nominals.pivot_table(
                        index='Start Date', 
                        columns='Ticker', 
                        values='Nominal', 
                        aggfunc='sum'
                        ) 

            #fecha maxima
            fecha_max=df_curnt_nominals.index.max()

            #nominales mas recientes
            df_curnt_nominals=df_curnt_nominals.loc[[fecha_max]]

            #nombre de las columnas de cada portafolio y nombre largo del portafolio
            cols_name=df_curnt_nominals.columns.to_list()
            large_name_port = df_info[df_info["Ticker"].isin(cols_name)].set_index("Ticker")["Long Name"].to_dict()
            # large_name_port=df_info["Long Name"][df_info["Ticker"].isin(cols_name)].reset_index(drop=True)
            
            #precios de los fondos de cada portafolio
            start_prices=prices[cols_name].loc[[prices.index.min()]]
            final_prices=prices[cols_name].loc[[prices.index.max()]]
            
            #precios de portafolios
            portfolio_start_price=df_portfolio_prices.loc[fecha_inicio,port]
            portfolio_final_price=df_portfolio_prices.loc[fecha_fin,port]
            
            #rendimiento total
            total_portfolio=portfolio_final_price/portfolio_start_price - 1

            #allocation de los fondos
            allocation=final_prices.reset_index(drop=True)*df_curnt_nominals.reset_index(drop=True)/portfolio_final_price

            # -- generación del grafico del portafolio --
            data_port = dict_grafico["Portafolio"][port]
            ticker_bmrk = dict_relation.get(port)
            data_bmrk = dict_grafico["Benchmark"].get(ticker_bmrk)
            # img_buffer = kit_f_secundarias.generar_grafico_tablas(data_port, data_bmrk, port)
            img_buffer = kit_f_secundarias.generar_grafico_linea_suave(data_port, data_bmrk, port)

            # generar el excel de cada portafolio
            excel_file=kit_f_secundarias.crear_excel(large_name_port,cols_name, total_portfolio,start_prices,
                                                    final_prices,allocation,port,prices,img_buffer,
                                                    df_ocw,df_dfaf,start_dt_ocw,start_dt_dfaf)

            
            # boton de descarga
            st.download_button(
                label=f"Descargar {port[:6]}",
                data=excel_file,
                file_name=f"Funds Rendimientos -{port[5:6]} {fecha_fin} {periodicity}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"Reporte_{port}")
        
    return