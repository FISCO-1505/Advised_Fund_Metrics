
import pandas as pd
import streamlit as st
import numpy as np
import Kit_Funciones as kit_funciones
import Kit_Metricas as kit_metricas

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
                    aggfunc='sum'  # Si hay duplicados, los suma. Si no, los deja igual.
                    )   
        
        # Pesos alineados (ffill para mantener el peso hasta el siguiente cambio)
        w_aligned = w_pivot.reindex(df_prices.index).ffill().shift(1)
        
        # Retornos de componentes SIN fillna (mantener NaNs originales)
        comp_tickers = w_pivot.columns
        comp_returns = df_prices[comp_tickers].pct_change() 
        
        # Cálculo del Benchmark:
        # Usamos min_count=1 para que si TODOS los componentes son NaN, 
        # el resultado sea NaN (no 0).
        dict_bmks_returns[bmk_name] = (comp_returns * w_aligned).sum(axis=1, min_count=1)

    # 1. Retornos con NaNs (Para Volatilidad, Sharpe, Beta, Tracking Error)
    returns_bmrk = pd.DataFrame(dict_bmks_returns)
    
    # 2. Retornos con Zeros (Para Precios e Índices acumulados)
    returns_bmrk_z = returns_bmrk.fillna(0)
    
    # 3. Precios Acumulados (Basados en los retornos con ceros)
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


def Funds_Commodity_test(_data, fecha_fin, periodicity=None,c_d=None):
    df_info = _data['Info']
    index_cols = df_info[df_info['Type'].isin(['Fund', 'Commodity'])]['Ticker'].tolist()

    df_prices = _data['Prices'].set_index('Date')
    df_ocw = _data['OCWHAUA LX Equity'].set_index('Date')
    df_prices = df_prices.join(df_ocw, how='left')

    # Precios y retornos de los fondos seleccionados
    df_prices_all = df_prices[index_cols]
    # returns_all = kit_metricas.df_returns(df_prices_all)#df_prices_all.pct_change()
    
    # Benchmarks (Calculados una sola vez para toda la historia)
    returns_bmrk_pure, _, _ = calculus_bmrk(_data)
    returns_aux_all = bmrk_aux_funds(_data['Funds - BMRK'], returns_bmrk_pure, fecha_fin)


    # Definición de Fecha de Inicio según Periodicidad
    fecha_inicio_aux = kit_funciones.start_dt(fecha_fin, periodicity,c_d)

    if fecha_inicio_aux == "INSUFFICIENT_DATA":
        st.warning(f"Insufficient historical data for {periodicity}. "
                f"Records start on: {pd.to_datetime('2015-12-03').date()}")
        st.stop()
    else:
        # Lógica de asignación final
        if periodicity == 'Since Inception':
            fecha_inicio = df_prices.index.min()
        else:
            # fecha_res contiene el objeto Timestamp válido
            idx_start = df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
            fecha_inicio = df_prices.index[idx_start]
        
        st.success(f"Analysis period: {fecha_inicio.date()} to {pd.to_datetime(fecha_fin).date()}")


    ####### parece ser que ya podemos eliminar esta parte, solo falta corrborarlo

    # # start_date(end_date, period="1Y", custom_start=None)
    # if periodicity == 'Since Inception':
    #     fecha_inicio = df_prices.index.min()
    #     st.write(f"Date Since Inception{fecha_inicio.date()}")
    #     #ver si se agrega el since inception de cada fondo

    # elif periodicity == 'MTD':
    #     fecha_inicio_aux = kit_funciones.start_dt(fecha_fin, periodicity)
    #     st.write(fecha_inicio_aux)
    #     idx_start = df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
    #     fecha_inicio = df_prices.index[idx_start]
    #     st.write(fecha_inicio)

    #     if fecha_inicio <= pd.to_datetime('2015-12-03'):
    #         return st.warning(
    #         f"Insufficient historical data for MTD. "
    #         f"Required start date: {fecha_inicio.date()}, "
    #         f"but the file starts on: {pd.to_datetime('2015-12-03').date()}")
        
    # elif periodicity == 'YTD':
    #     fecha_inicio_aux = kit_funciones.start_dt(fecha_fin, periodicity)
    #     st.write(fecha_inicio_aux)
    #     idx_start = df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
    #     fecha_inicio = df_prices.index[idx_start]
    #     st.write(fecha_inicio)

    #     if fecha_inicio <= pd.to_datetime('2015-12-03'):
    #         return st.warning(
    #         f"Insufficient historical data for MTD. "
    #         f"Required start date: {fecha_inicio.date()}, "
    #         f"but the file starts on: {pd.to_datetime('2015-12-03').date()}")
        
    # elif periodicity == '1Y':
    #     fecha_inicio_aux = kit_funciones.start_dt(fecha_fin, periodicity)
    #     st.write(fecha_inicio_aux)
    #     idx_start = df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
    #     fecha_inicio = df_prices.index[idx_start]
    #     st.write(fecha_inicio)
    #     if fecha_inicio <= pd.to_datetime('2015-12-03'):
    #         return st.warning(f"Insufficient history for 1Y. Start: {fecha_inicio}")
            
    # #aun pendiente por revisar las fechas 
    # elif periodicity == 'YTD':
    #     fecha_inicio = kit_funciones.start_dt(fecha_fin, YTD=True)
    #     if fecha_inicio <= pd.to_datetime('2015-12-03'):
    #         return st.warning(f"Insufficient history for YTD. Start: {fecha_inicio}")

    # elif periodicity == 'Custome Date':
    #     fecha=c_d

    #recorte del periodo seleccionado
    df_prices_RF=_data['Prices'].set_index('Date')
    df_prices_RF=df_prices_RF.loc[fecha_inicio:fecha_fin]

    prices = df_prices_all.loc[fecha_inicio:fecha_fin]
    returns = kit_metricas.df_returns(prices)
    st.dataframe(prices)
    st.write(prices.shape)

    # st.dataframe(returns_all)
    # st.write(returns_all.shape)
    
    st.dataframe(returns)
    st.write(returns.shape)

    returns_z=returns.fillna(0)
    returns_aux = returns_aux_all.loc[fecha_inicio:fecha_fin]
    
    # # Risk Free Rate para el periodo
    # rf_df = 

    # 5. Cálculo de Métricas (Llamando a tus funciones simplificadas)
    df_final = {}
    df_final['PX_LAST'] = prices.ffill()
    df_final['% Change'] = returns
    df_final['Cumulative'] = kit_metricas.cumm_return(returns_z,fecha_fin)
    df_final['RF'] = kit_metricas.RF(df_prices_RF, periodicity, fecha_fin)#modificar para que tenga periodo de entrada y quitar fecha fin
    
    # Métricas de Riesgo y Retorno
    df_final['Vol'] = kit_metricas.rolling_vol(returns, periodicity, fecha_fin)
    df_final['Sharpe Ratio'] = kit_metricas.sharpe_ratio(returns, df_final['RF'] , df_final['Vol'], fecha_fin, periodicity)
    
    # # Downside
    # df_final['Negative Returns'] = kit_metricas.negative_returns(returns, fecha_fin, periodicity)
    # df_final['Downside Deviation'] = kit_metricas.downside_deviation(df_final['Negative Returns'], fecha_fin, periodicity)
    # df_final['Sortino Ratio'] = kit_metricas.Sortino(returns, df_final['RF'], df_final['Downside Deviation'], fecha_fin, periodicity)
    # df_final['VaR']=kit_metricas.VaR(df_final['% Change'],fecha_fin,periodicity)

    # # Benchmark Related
    # df_final['Beta'] = kit_metricas.Beta(returns, returns_aux, fecha_fin) # Tu función Beta
    # df_final['Daily Active Return'] = returns - returns_aux
    # df_final['Tracking Error'] = df_final['Daily Active Return'].std() * np.sqrt(252)
    
    # # Drawdowns (Usando precios del periodo)
    # df_final['Max. Drawdown'] = kit_metricas.Max_Drawdown(df_final['PX_LAST'], fecha_fin)

            
    fnds_cmmdty=pd.DataFrame()
    for metric,col in df_final.items():
        fnds_cmmdty[metric]=df_final[metric].loc[fecha_fin]
    
    
    return fnds_cmmdty