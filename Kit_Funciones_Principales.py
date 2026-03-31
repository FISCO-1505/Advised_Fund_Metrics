
import pandas as pd
import streamlit as st
import Kit_Funciones_Secundarias as kit_f_secundarias
import Kit_Metricas as kit_metricas

def Funds_Commodity_test(_data, fecha_fin, periodicity=None,stats=None,assets=None,c_d=None):
    df_info = _data['Info']
    index_cols = df_info[df_info['Type'].isin(['Fund', 'Commodity'])]['Ticker'].tolist()

    df_prices = _data['Prices'].set_index('Date')
    df_ocw = _data['OCWHAUA LX Equity'].set_index('Date')
    df_prices = df_prices.join(df_ocw, how='left')

    df_prices_all = df_prices[index_cols]
    
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


    # En tu script principal, al armar la tabla final:
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
    if stats is not None and assets is not None:
        kit_f_secundarias.graficos_interactivos(fnds_cmmdty.loc[assets], prices[assets], stats,periodicity)

    # st.dataframe(fnds_cmmdty)
    return fnds_cmmdty