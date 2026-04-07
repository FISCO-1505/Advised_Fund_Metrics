
import pandas as pd
import streamlit as st
import numpy as np
import Kit_Funciones_Secundarias as kit_f_secundarias
import Kit_Metricas as kit_metricas

def Funds_Commodity(_data, fecha_fin, periodicity=None,stats=None,assets=None,grafico=None,c_d=None):
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
            fecha_inicio = df_prices.index[idx_start]
        
        st.success(f"Analysis period: {fecha_inicio.date()} to {pd.to_datetime(fecha_fin).date()}")


    #recorte del periodo seleccionado
    df_prices_RF=_data['Prices'].set_index('Date')
    df_prices_RF=df_prices_RF.loc[fecha_inicio:fecha_fin]

    prices = df_prices_all.loc[fecha_inicio:fecha_fin]
    returns,fechas_reales = kit_metricas.df_returns(prices)
    
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
    df_dd = kit_metricas.Drawdown(prices, fechas_reales)
    df_mdd = kit_metricas.Max_Drawdown(prices, fechas_reales)


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
    
    # --- APARTADO DE GRÁFICOS ---
    if stats is not None and assets is not None and grafico:
        kit_f_secundarias.graficos_interactivos(fnds_cmmdty.loc[assets], prices[assets], stats,periodicity,"Funds")

    formatos = {
        "PX_LAST": "${:,.3f}",
        "% Change": "{:.2%}",
        "Cumulative": "{:.2%}",
        "Negative Returns": "{:.2%}",
        "RF": "{:.2%}",
        "Alpha": "{:.2%}",
        "J Alpha": "{:.2%}",
        "Daily Active Return": "{:.2%}",
        "Vol": "{:.2%}",
        "Downside Deviation": "{:.2%}",
        "VaR": "{:.2%}",
        "Tracking Error": "{:.2%}",
        "Drawdown": "{:.2%}",
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
    # st.dataframe(stats)
    # st.dataframe(fnds_cmmdty[stats])
    
    # st.dataframe(fnds_cmmdty.columns)
    return fnds_cmmdty,formatos

def Portfolio(_data, fecha_fin, periodicity=None, stats=None, portfolios=None, c_d=None):
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

    prices = df_prices_all.loc[fecha_inicio:fecha_fin]
    returns,fechas_reales = kit_metricas.df_returns(prices)

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
    df_dd = kit_metricas.Drawdown(prices, fechas_reales)
    df_mdd = kit_metricas.Max_Drawdown(prices, fechas_reales)


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
    if stats is not None and portfolios is not None:
        
        kit_f_secundarias.graficos_interactivos(final_portfolios.loc[portfolios], prices[portfolios], stats,periodicity,"Portfolios")

    # st.dataframe(returns)
    return final_portfolios, formatos

def procesar_analisis(topic, data, selection, stats, assets):
    """
    Maneja la lógica de fechas, comparativas y carga de métricas 
    para Funds y Portfolios de forma unificada.
    """
    # Manejo de Fechas
    start_date = None
    selected_date = None
    
    if selection == "Custom Date":
        start_date, selected_date = kit_f_secundarias.calendar(data["Prices"]["Date"], mode="range")
    else:
        selected_date = kit_f_secundarias.calendar(data["Prices"]["Date"], mode="single")

    # Configuración de Comparativa (Toggle)
    toggle_button = st.toggle("Comparative", key=f"toggle_{topic}_{selection}")
    stats_aux = stats if toggle_button else stats #None
    assets_aux = assets if toggle_button else assets #None
    grafico= True if toggle_button else None

    # Botón de Carga
    if st.button("Load metrics", key=f"btn_{topic}_{selection}"):

        # --- validación ---
        if not assets:
            st.warning("⚠️ Please select at least one asset/portfolio to continue.")
            return
        
        if not stats:
            st.warning("⚠️ Please select at least one metric to calculate.")
            return
        # ----------------------------

        with st.spinner("Calculating metrics..."):
            # Seleccionar la función según el topic
            func_principal = (Funds_Commodity if topic == "Funds" else Portfolio)
            
            # Llamada a la función según selection
            if selection == "Custom Date":
                results, formatos = func_principal(data, selected_date, selection, stats_aux, assets_aux,grafico, start_date)

            else:
                results, formatos = func_principal(data, selected_date, selection, stats_aux, assets_aux,grafico)

            # Filtrado y Visualización
            cols_to_show = [c for c in (stats + ['Real Date']) if c in results.columns]
            final_df = results[cols_to_show].loc[assets]
            
            st.dataframe(final_df.style.format(formatos, na_rep="-"))
            
            # Generar excel
            kit_f_secundarias.generar_excel_fondos(assets,results[stats])

#falta realizar el gráfico y los casos de la fechas e info de DFAF y OCW (wilshire)
#solo mostrar los portafolios 6,7,8
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

    ####_____________________faltaaaaaaaaa___________________________#####
    #returns de los benchamrks, para poder usar en los gráficos 
    _, _, returns_bmrk_z = kit_f_secundarias.calculus_bmrk(_data)

    # Precios de los portafolios
    df_portfolio_prices=kit_f_secundarias.portfolio_Prices(df_prices_funds,df_fixed_portfolios,df_nominals)

    # Definición de Fecha de Inicio según Periodicidad
    fecha_inicio_aux = kit_f_secundarias.start_dt(fecha_fin, periodicity,None)

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

    #precios filtrados
    prices = df_prices_funds.loc[fecha_inicio:fecha_fin]

    #bmrk acumulado filtrado
    accm_bmrk = kit_metricas.cumm_return(returns_bmrk_z)
    accm_bmrk = accm_bmrk.loc[fecha_inicio:fecha_fin]

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

            #nombre de las columnas de cada portafolio
            cols_name=df_curnt_nominals.columns.to_list()

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

            excel_file=kit_f_secundarias.crear_excel(cols_name, total_portfolio,start_prices,
                                                    final_prices,allocation,port,prices)

            st.download_button(
                label=f"Descargar {port}",
                data=excel_file,
                file_name=f"reporte_portafolio{port}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"Reporte_{port}")
        
    return