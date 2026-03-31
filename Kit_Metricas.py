import pandas as pd
import numpy as np
import warnings
from datetime import datetime
# import calendar
import streamlit as st
import Kit_Funciones_Secundarias as kit_funciones

#display options
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.6f}'.format(x))
pd.set_option('display.max_columns', None)


@st.cache_data
def df_returns(prices):
    '''
    Calcula retornos manteniendo los NaNs originales de la matriz de precios.
    '''
    returns = prices.ffill().pct_change()
    returns = returns.mask(prices.isna())
    fechas_reales=returns.apply(lambda x: x.last_valid_index())
    
    return returns.iloc[1:], fechas_reales

@st.cache_data
def cumm_return(returns,fecha_fin=None):
    """
    returns: Series with returns
    """
    df_cumm=((1+returns).cumprod() - 1)

    if fecha_fin!=None:
        total_ret = df_cumm.iloc[-1] 
        df_result = pd.DataFrame(total_ret).T
        df_result.index = [fecha_fin]

        return df_result
        
    return df_cumm
        
@st.cache_data
def RF(df_prices, fecha_fin=None):
    '''
    df_prices: DataFrame que ya contiene la columna 'USGG3M Index' filtrada.
    '''

    rf_expanding = df_prices['USGG3M Index'].expanding().mean()
    
    # Retornamos la serie completa para poder indexar después por fecha_real
    return rf_expanding/100

@st.cache_data
def rolling_vol(returns, fechas_reales):
    '''
    returns: DataFrame de retornos (con NaNs).
    fechas_reales: Serie con la última fecha válida por fondo.
    '''
    # vol de toda la matriz
    vol_full = returns.expanding().std() * np.sqrt(252)
    # 2. Extraemos el valor solo si la fecha existe y es válida
    vol_final = {}
    for col in returns.columns:
        fecha = fechas_reales.get(col)
        # Verificamos que la fecha no sea NaT y que exista en el índice de vol_full
        if pd.notnull(fecha) and fecha in vol_full.index:
            vol_final[col] = vol_full.loc[fecha, col]
        else:
            vol_final[col] = np.nan 
    
    return pd.DataFrame(vol_final, index=["Vol"])

@st.cache_data
def sharpe_ratio(returns, rf_series, vol_df, fechas_reales):
    '''
    returns: DataFrame de retornos (con NaNs).
    rf_series: La serie devuelta por la función RF.
    vol_df: El DataFrame devuelto por rolling_vol.
    fechas_reales: Serie con la última fecha válida por fondo.
    '''
    avg_daily = returns.mean()
    
    ann_return = ((1 + avg_daily)**252) - 1
    
    #Manejo de NaT para evitar KeyError
    rf_vals_dict = {}
    for col in returns.columns:
        fecha = fechas_reales.get(col)

        if pd.notnull(fecha) and fecha in rf_series.index:
            rf_vals_dict[col] = rf_series.loc[fecha]
        else:
            rf_vals_dict[col] = np.nan
            
    rf_vals = pd.Series(rf_vals_dict)

    vols = vol_df.iloc[0] 

    sharpe = (ann_return - rf_vals) / vols
    
    # Reemplazar infinitos o valores erróneos por NaN
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    
    return pd.DataFrame(sharpe).T.rename(index={0: 'Sharpe Ratio'})

@st.cache_data
def downside_deviation(negative_returns, fechas_reales):
    '''
    Calcula la desviación a la baja anualizada de forma vectorizada.
    '''
    downside_returns = negative_returns

    dd_full = np.sqrt((downside_returns**2).expanding().mean()) * np.sqrt(252)
    
    # 3. Extraemos el valor en la fecha real
    dd_final = {col: dd_full.loc[fechas_reales[col], col] 
                if pd.notnull(fechas_reales.get(col)) else np.nan 
                for col in negative_returns.columns}
    
    return pd.DataFrame(dd_final, index=["Downside Deviation"])

@st.cache_data
def negative_returns(returns):
    '''
    Retorna el DataFrame original pero solo con los valores negativos (el resto NaN).
    '''
    return returns[returns < 0]

@st.cache_data
def VaR(returns, fechas_reales, conf_lvl=99):
    '''
    Calcula el VaR Histórico (Percentil) de forma vectorizada.
    '''
    alpha = 1 - (conf_lvl / 100)
    
    # Calculamos el cuantil expansivo para toda la matriz
    var_full = returns.expanding().quantile(alpha)
    
    # Extraemos el valor exacto para cada fondo
    var_final = {col: var_full.loc[fechas_reales[col], col] 
                 if pd.notnull(fechas_reales.get(col)) else np.nan 
                 for col in returns.columns}
    
    return pd.DataFrame(var_final, index=[f"VaR {conf_lvl}%"])

@st.cache_data
def Sortino(returns, rf_series, dd_df, fechas_reales):
    avg_daily = returns.mean()
    ann_return = ((1 + avg_daily)**252) - 1
    
    # Alineamos RF y Downside Deviation a la fecha real
    rf_vals = pd.Series({c: rf_series.loc[fechas_reales[c]] if pd.notnull(fechas_reales[c]) else np.nan for c in returns.columns})
    down_devs = dd_df.iloc[0]
    
    sortino = (ann_return - rf_vals) / down_devs
    return pd.DataFrame(sortino).T.rename(index={0: 'Sortino Ratio'})
    
@st.cache_data
def Beta(returns, returns_bmrk, fechas_reales=None):
    beta_values = {}
    
    for col in returns.columns:
        fecha_corte = fechas_reales.get(col) if fechas_reales is not None else None
        
        if pd.notnull(fecha_corte) and fecha_corte in returns.index:
            fondo_r = returns.loc[:fecha_corte, col]
            bmrk_r = returns_bmrk.loc[:fecha_corte, col]
        else:
            fondo_r = returns[col]
            bmrk_r = returns_bmrk[col]

        cov_m = fondo_r.cov(bmrk_r)
        var_bmrk_m = bmrk_r.var()
        
        # Cálculo de la Beta
        if var_bmrk_m != 0 and pd.notna(cov_m):
            beta_values[col] = cov_m / var_bmrk_m
        else:
            beta_values[col] = np.nan
            
    return pd.DataFrame(beta_values, index=['Beta'])

@st.cache_data
def Alpha(df_cumm, beta_df, bmrk_cumm, fechas_reales):
    '''
    Calcula el Alpha: Retorno acumulado del fondo - (Beta * Retorno acumulado Benchmark).
    Considera los valores acumulados en la fecha de corte específica de cada fondo.
    '''
    alpha_values = {}
    betas = beta_df.iloc[0]
    
    for col in df_cumm.columns:
        fecha_corte = fechas_reales.get(col)
        
        if pd.notnull(fecha_corte) and fecha_corte in df_cumm.index:
            fondo_c = df_cumm.loc[fecha_corte, col]
            bmrk_c = bmrk_cumm.loc[fecha_corte, col]
            beta_fnd = betas[col]
            
            # Alpha = Retorno Fondo - (Beta * Retorno Mercado)
            alpha_values[col] = fondo_c - (beta_fnd * bmrk_c)
        else:
            alpha_values[col] = np.nan
            
    return pd.DataFrame(alpha_values, index=['Alpha'])

@st.cache_data
def J_Alpha(returns, rf_series, beta_df, returns_bmrk, fechas_reales):
    '''
    Calcula el Jensen's Alpha anualizado:
    J_Alpha = R_anualizado - [RF + Beta * (R_bmrk_anualizado - RF)]
    '''
    j_alpha_values = {}
    betas = beta_df.iloc[0]
    
    for col in returns.columns:
        fondo_r = returns[col]
        bmrk_r = returns_bmrk[col]
        fecha_corte = fechas_reales.get(col)
        
        if pd.notnull(fecha_corte):
            fondo_r = fondo_r.loc[:fecha_corte]
            bmrk_r = bmrk_r.loc[:fecha_corte]
            
        s_f_aligned = fondo_r.reindex(bmrk_r.index)#.fillna(0)
        
        # Anualización de Retornos (Fondo y Benchmark)
        ann_ret_fnd = ((1 + s_f_aligned.mean())**252) - 1
        ann_ret_bmrk = ((1 + bmrk_r.mean())**252) - 1
        
        rf_val = rf_series.loc[fecha_corte] if pd.notnull(fecha_corte) else np.nan
        beta_fnd = betas[col]
        
        if pd.notna(rf_val) and pd.notna(beta_fnd):
            # Alpha = RetornoFondo - [Rf + Beta * (RetornoMkt - Rf)]
            j_alpha_values[col] = ann_ret_fnd - (rf_val + beta_fnd * (ann_ret_bmrk - rf_val))
        else:
            j_alpha_values[col] = np.nan
            
    return pd.DataFrame(j_alpha_values, index=['Jensen Alpha'])

@st.cache_data
def Treynor_ratio(returns, rf_series, beta_df, fechas_reales):
    avg_daily = returns.mean()
    ann_return = ((1 + avg_daily)**252) - 1
    rf_vals = pd.Series({c: rf_series.loc[fechas_reales[c]] if pd.notnull(fechas_reales[c]) else np.nan for c in returns.columns})
    betas = beta_df.iloc[0]
    
    treynor = (ann_return - rf_vals) / betas
    return pd.DataFrame(treynor).T.rename(index={0: 'Treynor Ratio'})

@st.cache_data
def daily_active_returns(returns, returns_bmrk, fechas_reales=None):
    '''
    Calcula la matriz de retornos activos diarios (Fondo - Benchmark)
    alineando calendarios y tratando los NaNs como 0% de movimiento.
    '''
    active_returns_dict = {}

    for col in returns.columns:
        fondo_r = returns[col]
        bmrk_r = returns_bmrk[col]

        fecha_corte = fechas_reales.get(col) if fechas_reales is not None else None
        
        if pd.notnull(fecha_corte):
            fondo_r = fondo_r.loc[:fecha_corte]
            bmrk_r = bmrk_r.loc[:fecha_corte]
        
        # (returns con 0s) - (Benchmark)
        active_returns_dict[col] = fondo_r.reindex(bmrk_r.index).fillna(0) - bmrk_r

    df_active_returns = pd.DataFrame(active_returns_dict)

    return df_active_returns

@st.cache_data
def Tracking_Error(returns, returns_bmrk, fechas_reales=None):
    '''
    Calcula el Tracking Error: std(daily_active_rtrns)*sqrt(252).
    Usa fillna(0) para el caso de los daily active returns.
    '''

    te_values = {}
    
    for col in returns.columns:
        fecha_corte = fechas_reales.get(col) if fechas_reales is not None else None
        
        if pd.notnull(fecha_corte) and fecha_corte in returns.index:
            fondo_r = returns.loc[:fecha_corte, col]
            bmrk_r = returns_bmrk.loc[:fecha_corte, col]
        
        else:
            fondo_r = returns[col]
            bmrk_r = returns_bmrk[col]

        active_diff = (fondo_r.reindex(bmrk_r.index).fillna(0) - bmrk_r)
        
        #Cálculo del Tracking Error Anualizado
        if len(active_diff) > 1:
             te_values[col]= active_diff.std() * np.sqrt(252)
        else:
            te_values[col] = np.nan
            
    return pd.DataFrame(te_values, index=['Tracking Error'])

@st.cache_data
def info_ratio(returns, returns_bmrk, df_te, fechas_reales=None):
    '''
    Calcula el Information Ratio: (Retorno Activo Anualizado) / Tracking Error.
    Usa fillna(0) para que el retorno activo sea consistente con el TE calculado.
    '''
    ir_values = {}
    
    for col in returns.columns:
        fondo_r = returns[col]
        bmrk_r = returns_bmrk[col]
        
        fecha_corte = fechas_reales.get(col) if fechas_reales is not None else None
        if pd.notnull(fecha_corte):
            fondo_r = fondo_r.loc[:fecha_corte]
            bmrk_r = bmrk_r.loc[:fecha_corte]
            
        active_diff = fondo_r.reindex(bmrk_r.index).fillna(0) - bmrk_r

        ann_active_ret = ((1 + active_diff.mean())**252) - 1
        current_te = df_te[col].iloc[0]
        
        if current_te > 0:
            ir_values[col] = ann_active_ret / current_te
        else:
            ir_values[col] = np.nan
            
    return pd.DataFrame(ir_values, index=['Information Ratio'])

@st.cache_data
def correlation(returns, returns_bmrk, fechas_reales=None):
    '''
    Calcula la correlación de Pearson considerando la historia real 
    de cada fondo
    '''
    corr_values = {}
    
    for col in returns.columns:
        fondo_r = returns[col]
        bmrk_r = returns_bmrk[col]
        
        fecha_corte = fechas_reales.get(col) if fechas_reales is not None else None
        if pd.notnull(fecha_corte):
            fondo_r = fondo_r.loc[:fecha_corte]
            bmrk_r = bmrk_r.loc[:fecha_corte]
            
        val_corr = fondo_r.corr(bmrk_r)
        
        corr_values[col] = val_corr if pd.notna(val_corr) else 0

    return pd.DataFrame(corr_values, index=['Correlation'])

@st.cache_data
def Drawdown(prices, fechas_reales):
    '''
    Calcula el Drawdown actual (en la fecha real) respecto al máximo histórico previo.
    '''
    pico_historico = prices.max()
    all_drawdowns = (prices / pico_historico) - 1
    
    #validación de NaT
    dd_final = {}
    for c in prices.columns:
        fecha = fechas_reales.get(c)
        
        
        if pd.notnull(fecha) and fecha in all_drawdowns.index:
            dd_final[c] = all_drawdowns.loc[fecha, c]
        else:
            dd_final[c] = np.nan
            
    return pd.DataFrame(dd_final, index=["Drawdown"])

@st.cache_data
def Max_Drawdown(prices, fechas_reales):
    '''
    Calcula la caída máxima histórica (el punto más bajo del drawdown) de forma segura.
    '''
    pico_historico = prices.cummax()
    all_drawdowns = (prices / pico_historico) - 1
    

    #captura la peor caída registrada hasta cada punto en el tiempo
    mdd_full = all_drawdowns.expanding().min()
    
    mdd_final = {}
    for c in prices.columns:
        fecha = fechas_reales.get(c)

        if pd.notnull(fecha) and fecha in mdd_full.index:
            mdd_final[c] = mdd_full.loc[fecha, c]
        else:
            mdd_final[c] = np.nan
            
    return pd.DataFrame(mdd_final, index=["Max Drawdown"])


#____ Funciones Generales

@st.cache_data
#si, COPIA
def Benchmark_test(_data,fecha_fin,periodicity=None):

    df_final={}
    df_prices=_data['Prices'].set_index('Date')

    returns,px_last,returns_z=calculus_bmrk(_data)
    rf_df=RF(_data)
    df_cumm_returns=cumm_return(returns_z)
    
    if periodicity=='1Y':
        
        fecha_inicio=(datetime.strptime(fecha_fin, '%Y-%m-%d').date()-pd.Timedelta(days=bisiesto(fecha_fin))).strftime('%Y-%m-%d')
        if fecha_inicio>'2015-12-03':
            # print('Sí es mayor a 365 días')
            returns=returns.loc[fecha_inicio:fecha_fin]
            px_last=px_last.loc[fecha_inicio:fecha_fin]
            returns_z=returns_z.loc[fecha_inicio:fecha_fin]
            # print(px_last)
            #IMPORTANTE: usan la columna de los precios para el acumulado, revisar esta parte
            df_cumm_returns=cumm_return(px_last,fecha_fin)
            rf_df=RF(_data,fecha_inicio,fecha_fin)
            
        else: 
            return f'It is necessary to enter a higher date than 2015-12-03. Your initial date is: {fecha_inicio}'
            
    elif periodicity=='YTD':
        year_date=datetime.strptime(fecha_fin, "%Y-%m-%d").year
        fecha_inicio=datetime.strptime(f'{year_date}-01-01', "%Y-%m-%d").strftime('%Y-%m-%d')
        if fecha_inicio>'2015-12-03':
            # print('Sí es mayor a 365 días')
            returns=returns.loc[fecha_inicio:fecha_fin]
            px_last=px_last.loc[fecha_inicio:fecha_fin]
            returns_z=returns_z.loc[fecha_inicio:fecha_fin]

            rf_df=RF(_data,fecha_inicio,fecha_fin)
            
        else:
            return f'It is necessary to enter a higher date than 2015-12-03. Your initial date is: {fecha_inicio}'
        
    else:
        
        pass
    
    #                                     _______________metrics for benchmarks_______________
    
    #get the PX_LAST
    df_final['PX_LAST']=px_last
    
    #get the % Change
    df_final['% Change']=returns
    
    #get the cummulative df
    df_final['Cumulative']=df_cumm_returns

    #get the RF df
    df_final['RF']=rf_df#RF(data)
    
    #get the volatility df
    df_final['Vol']=rolling_vol(returns,fecha_fin)
    
    #get the Sharpe Ratio
    df_final['Sharpe Ratio']=sharpe_ratio(df_final['% Change'],df_final['RF'],df_final['Vol'],fecha_fin)
    
    #get the Negative Returns
    df_final['Negative Returns']=df_final['% Change'].where(df_final['% Change']<0)
    
    #get the Downside Deviation
    df_final['Downside Deviation']=downside_deviation(df_final['Negative Returns'],fecha_fin)
    
    #get the sortino ratio
    df_final['Sortino Ratio']=Sortino(df_final['% Change'],df_final['RF'],df_final['Downside Deviation'],fecha_fin)
    
    #get the VaR
    df_final['VaR']=VaR(df_final['% Change'],fecha_fin)
    
    #Create the dataframe of benchmark final results 
    bmrk=pd.DataFrame()
    for metric,col in df_final.items():
        bmrk[metric]=df_final[metric].loc[fecha_fin]

    return bmrk

@st.cache_data
#si
def Benchmark(_data,fecha_fin):

    df_final={}
    df_prices=_data['Prices'].set_index('Date')

    returns,px_last,returns_z=calculus_bmrk(_data)
    
    #                                     _______________metrics for benchmarks_______________
    
    #get the PX_LAST
    df_final['PX_LAST']=px_last
    
    #get the % Change
    df_final['% Change']=returns
    
    #get the cummulative df
    df_final['Cumulative']=cumm_return(returns_z)
    
    #get the RF df
    df_final['RF']=df_prices['USGG3M Index'].expanding().mean().iloc[1:]
    
    #get the volatility df
    df_final['Vol']=rolling_vol(returns,fecha_fin)
    
    #get the Sharpe Ratio
    df_final['Sharpe Ratio']=sharpe_ratio(df_final['% Change'],df_final['RF'],df_final['Vol'],fecha_fin)
    
    #get the Negative Returns
    df_final['Negative Returns']=df_final['% Change'].where(df_final['% Change']<0)
    
    #get the Downside Deviation
    df_final['Downside Deviation']=downside_deviation(df_final['Negative Returns'],fecha_fin)
    
    #get the sortino ratio
    df_final['Sortino Ratio']=Sortino(df_final['% Change'],df_final['RF'],df_final['Downside Deviation'],fecha_fin)
    
    #get the VaR
    df_final['VaR']=VaR(df_final['% Change'],fecha_fin)
    
    #Create the dataframe of benchmark final results 
    bmrk=pd.DataFrame()
    for metric,col in df_final.items():
        bmrk[metric]=df_final[metric].loc[fecha_fin]

    return bmrk

@st.cache_data
#test
def Index_test(_data,fecha_fin,periodicity=None):
    df_final={}
    
    df_info=_data['Info']
    index_cols=df_info[df_info['Type']=='Index']['Ticker'].to_list()
    df_prices=_data['Prices'][index_cols+['Date']].set_index('Date')
    
    returns=df_returns(df_prices)
    returns_z=returns.fillna(0)

    rf_df=RF(_data)
    df_cumm_returns=cumm_return(returns_z)

    #_______________
    if periodicity=='1Y':
        
        fecha_inicio=(datetime.strptime(fecha_fin, '%Y-%m-%d').date()-pd.Timedelta(days=bisiesto(fecha_fin))).strftime('%Y-%m-%d')
        if fecha_inicio>'2015-12-03':
            # print('Sí es mayor a 365 días')
            returns=returns.loc[fecha_inicio:fecha_fin]
            df_prices=df_prices.loc[fecha_inicio:fecha_fin]
            returns_z=returns_z.loc[fecha_inicio:fecha_fin]

            #corrborar que si se oucpe df_prices
            df_cumm_returns=cumm_return(df_prices,fecha_fin)
            rf_df=RF(_data,fecha_inicio,fecha_fin)
            
        else: 
            return f'It is necessary to enter a higher date than 2015-12-03. Your initial date is: {fecha_inicio}'
            
    elif periodicity=='YTD':
        year_date=datetime.strptime(fecha_fin, "%Y-%m-%d").year
        fecha_inicio=datetime.strptime(f'{year_date}-01-01', "%Y-%m-%d").strftime('%Y-%m-%d')
        if fecha_inicio>'2015-12-03':
            # print('Sí es mayor a 365 días')
            returns=returns.loc[fecha_inicio:fecha_fin]
            df_prices=df_prices.loc[fecha_inicio:fecha_fin]
            returns_z=returns_z.loc[fecha_inicio:fecha_fin]

            # IMPORTANTE: Considrar que para los indices se maneja diferente el calculo del acumulado para el YTD
            rf_df=RF(_data,fecha_inicio,fecha_fin)
            
        else:
            return f'It is necessary to enter a higher date than 2015-12-03. Your initial date is: {fecha_inicio}'
        
    else:
        
        pass
    
        #                                     _______________metrics for index_______________
    
    #get the PX_LAST
    df_final['PX_LAST']=df_prices
    
    #get the % Change
    df_final['% Change']=returns
    
    #get the cummulative df
    df_final['Cumulative']=df_cumm_returns
    
    #get the RF df
    df_final['RF']=rf_df
    
    #get the volatility df
    df_final['Vol']=rolling_vol(returns,fecha_fin)
    
    #get the Sharpe Ratio
    df_final['Sharpe Ratio']=sharpe_ratio(df_final['% Change'],df_final['RF'],df_final['Vol'],fecha_fin)
    
    #Create the dataframe of Index final results 
    indx=pd.DataFrame()
    for metric,col in df_final.items():
        indx[metric]=df_final[metric].loc[fecha_fin]

    return indx

@st.cache_data
def Index(_data,fecha_fin):
    df_final={}
    
    df_info=_data['Info']
    index_cols=df_info[df_info['Type']=='Index']['Ticker'].to_list()
    df_prices=_data['Prices'][index_cols+['Date']].set_index('Date')
    
    returns=df_returns(df_prices)
    returns_z=returns.fillna(0)
    
        #                                     _______________metrics for index_______________
    
    #get the PX_LAST
    df_final['PX_LAST']=df_prices
    
    #get the % Change
    df_final['% Change']=returns
    
    #get the cummulative df
    df_final['Cumulative']=cumm_return(returns_z)
    
    #get the RF df
    df_final['RF']=RF(_data)
    
    #get the volatility df
    df_final['Vol']=rolling_vol(returns,fecha_fin)
    
    #get the Sharpe Ratio
    df_final['Sharpe Ratio']=sharpe_ratio(df_final['% Change'],df_final['RF'],df_final['Vol'],fecha_fin)
    
    #Create the dataframe of Index final results 
    indx=pd.DataFrame()
    for metric,col in df_final.items():
        indx[metric]=df_final[metric].loc[fecha_fin]

    return indx

@st.cache_data
#si COPIA
# def Funds_Commodity_test(_data,fecha_fin,periodicity=None):
#     df_final={}
    
#     df_info=_data['Info']
    
#     #get the info about the prices of funds and commodity
#     index_cols=df_info[(df_info['Type']=='Fund') | (df_info['Type']=='Commodity')]['Ticker'].to_list()
#     df_prices=_data['Prices'].set_index('Date')
#     df_OCW=_data['OCWHAUA LX Equity'].set_index('Date')
#     df_prices=df_prices.join(df_OCW,how='left')
#     ######considerar poner un ffill para que auqnue sea día festivo arroje la fehca
#     ###### pero no afecte directamente a los calculos de las metricas pero ya en el rasultado final
#     df_prices=df_prices[index_cols]#.ffill()
    
#     #get the funds returns
#     returns=df_returns(df_prices)
#     returns_z=returns.fillna(0)
    
#     #__
#     # get the columns and prices related to indexs
#     index_cols_index=df_info[df_info['Type']=='Index']['Ticker'].to_list()
#     df_prices_index=_data['Prices'][index_cols_index+['Date']].set_index('Date')
    
#     #se obtiene la información de los fondos que tienen bmrks
#     df_funds_bmrk=_data['Funds - BMRK']
#     df_funds_bmrk=df_funds_bmrk.replace('2015-12-03','2015-12-04')
   
#     #get the bmrk returns
#     returns_bmrk,_,_=calculus_bmrk(_data)
#     returns_aux=bmrk_aux_funds(df_funds_bmrk,returns_bmrk,df_prices_index)
    

#     fecha_inicio=None
#     #_______________________________________________
#     rf_df=RF(_data)
#     df_cumm_returns=cumm_return(returns_z)

#     if periodicity=='1Y':

#         fecha_inicio_aux=kit_funciones.start_dt(fecha_fin,n_months=12)
#         id_proximo=df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
#         fecha_inicio=df_prices.index[id_proximo]

#         if fecha_inicio>pd.to_datetime('2015-12-03'):
#             st.write(fecha_inicio)
#             st.write(fecha_fin)
#             df_prices=df_prices.loc[fecha_inicio:fecha_fin]
#             returns=df_returns(df_prices)
#             returns_z=returns.fillna(0)

#             st.dataframe(df_prices)
#             st.write(df_prices.shape)

#             # st.dataframe(returns_all)
#             # st.write(returns_all.shape)
            
#             st.dataframe(returns)
#             st.write(returns.shape)

#             returns_aux=returns_aux.loc[fecha_inicio:fecha_fin]
#             df_cumm_returns=cumm_return(returns_z,fecha_fin)
#             rf_df=RF(_data,fecha_inicio,fecha_fin)

#             # vol=rolling_vol(returns,periodicity,fecha_fin)
#             # st.write("_______________vol")
#             # st.dataframe(vol)
#             # print(vol)
#             # sharpe=sharpe_ratio(returns,rf_df,vol,fecha_fin,periodicity)
#             # st.write("sharpeeee")
#             # st.dataframe(sharpe)
#             # st.dataframe(vol.style.format("{:.8f}"))

#             # st.dataframe(rf_df)
#             # st.dataframe(returns)

#         else: 
#             return st.warning(f'It is necessary to enter at least {periodicity} higher than 2015-12-03. Your initial date is: {fecha_inicio}')
            
#     elif periodicity=='YTD':
#         # year_date=datetime.strptime(fecha_fin, "%Y-%m-%d").year
#         # fecha_inicio=datetime.strptime(f'{year_date}-01-01', "%Y-%m-%d").strftime('%Y-%m-%d')
#         fecha_inicio=kit_funciones.start_dt(fecha_fin,YTD=True)
#         if fecha_inicio>'2015-12-03':
#             # print('Sí es mayor a 365 días')
#             returns=returns.loc[fecha_inicio:fecha_fin]
#             #NOTA: para el caclculo del max drawdown puede haber diferenecias ya que para crear el primer dato de drawdown, sera NA
#             df_prices=df_prices.loc[fecha_inicio:fecha_fin]
#             returns_z=returns_z.loc[fecha_inicio:fecha_fin]
#             returns_aux=returns_aux.loc[fecha_inicio:fecha_fin]

#             # IMPORTANTE: Considrar que para los indices se maneja diferente el calculo del acumulado para el YTD
#             rf_df=RF(_data,fecha_inicio,fecha_fin)
            
#         else:
#             return f'It is necessary to enter a higher date than 2015-12-03. Your initial date is: {fecha_inicio}'
        
#     else:
        
#         pass
#         #                                     _______________metrics for funds and commodity_______________
#     #get the PX_LAST
#     df_final['PX_LAST']=df_prices.ffill()  #aqui a tendrá los ffill
    
#     #get the % Change
#     df_final['% Change']=returns #aqui ya tendra los retornos con NaN
    
#     #get the cummulative df
#     df_final['Cumulative']=df_cumm_returns  #aqui usa los retornos con 0
    
#     #get the RF df
#     df_final['RF']=rf_df
    
#     #get the volatility df
#     df_final['Vol']=rolling_vol(df_final['% Change'],periodicity,fecha_fin)#usa los retornos con NaN
    
#     #get the Sharpe Ratio
#     df_final['Sharpe Ratio']=sharpe_ratio(df_final['% Change'],df_final['RF'],df_final['Vol'],fecha_fin,periodicity)

#     #get the Negative Returns
#     df_final['Negative Returns']=negative_returns(df_final['% Change'],fecha_fin,periodicity) #df_final['% Change'].where(df_final['% Change']<0)
    
#     #get the Downside Deviation
#     df_final['Downside Deviation']=downside_deviation(df_final['Negative Returns'],fecha_fin,periodicity)
    
#     #get the sortino ratio
#     df_final['Sortino Ratio']=Sortino(df_final['% Change'],df_final['RF'],df_final['Downside Deviation'],fecha_fin,periodicity)

#     #get the VaR
#     df_final['VaR']=VaR(df_final['% Change'],fecha_fin,periodicity)
    
#     #get the Beta
#     df_final['Beta']=Beta(returns,returns_aux,fecha_fin)
    
#     #get the Treynor Ratio
#     df_final['Treynor Ratio']=Treynor_ratio(df_final['% Change'],df_final['RF'],df_final['Beta'],fecha_fin)
    
#     # get the daily active return
#     df_final['Daily Active Return']=df_final['% Change'] - returns_aux
    
#     #get the tracking error
#     df_final['Tracking Error']=Tracking_Error(df_final['Daily Active Return'],fecha_fin)
    
#     #get the info ratio
#     df_final['Info. Ratio']=info_ratio(df_final['Daily Active Return'],df_final['Tracking Error'],fecha_fin)
    
#     #get the Alpha
#     df_final['Alpha']=Alpha(df_final['Cumulative'],df_final['Beta'],cumm_return(returns_aux),fecha_fin)
    
#     #get the J Alpha
#     df_final['J Alpha']=J_Alpha(df_final['% Change'],df_final['RF'],df_final['Beta'],returns_aux,fecha_fin)
    
#     #get the correlation
#     df_final['Correlation']=correlation(df_final['% Change'],returns_aux,fecha_fin)
    
#     # get the R^2
#     df_final['R^2']=df_final['Correlation']**2
    
#     # get the Drawdown
#     df_final['Drawdown']=Drawdown(df_final['PX_LAST'],fecha_fin)
    
#     #get the Max Drawdown
#     df_final['Max. Drawdown']=Max_Drawdown(df_final['PX_LAST'],fecha_fin)
    
#     #Create the dataframe of Index final results 
#     fnds_cmmdty=pd.DataFrame()
#     for metric,col in df_final.items():
#         fnds_cmmdty[metric]=df_final[metric].loc[fecha_fin]
    
#     return fnds_cmmdty

@st.cache_data
def Funds_Commodity(_data,fecha_fin):
    df_final={}
    
    df_info=_data['Info']
    
    #get the info about the prices of funds and commodity
    index_cols=df_info[(df_info['Type']=='Fund') | (df_info['Type']=='Commodity')]['Ticker'].to_list()
    df_prices=_data['Prices'].set_index('Date')
    df_OCW=_data['OCWHAUA LX Equity'].set_index('Date')
    df_prices=df_prices.join(df_OCW,how='left')
    df_prices=df_prices[index_cols]
    
    #get the funds returns
    returns=df_returns(df_prices)
    returns_z=returns.fillna(0)
    
    #__
    # get the columns and prices related to indexes
    index_cols_index=df_info[df_info['Type']=='Index']['Ticker'].to_list()
    df_prices_index=_data['Prices'][index_cols_index+['Date']].set_index('Date')
    
    #se obtiene la información de los fondos que tienen bmrks
    df_funds_bmrk=_data['Funds - BMRK']
    df_funds_bmrk=df_funds_bmrk.replace('2015-12-03','2015-12-04')
    #get the bmrk returns
    returns_bmrk,_,_=calculus_bmrk(_data)
    returns_aux=bmrk_aux_funds(df_funds_bmrk,returns_bmrk,df_prices_index)
    
    
        #                                     _______________metrics for funds and commodity_______________
    
    #get the PX_LAST
    df_final['PX_LAST']=df_prices
    
    #get the % Change
    df_final['% Change']=returns
    
    #get the cummulative df
    df_final['Cumulative']=cumm_return(returns_z)
    
    #get the RF df
    df_final['RF']=RF(_data)
    
    #get the volatility df
    df_final['Vol']=rolling_vol(returns,fecha_fin)
    
    #get the Sharpe Ratio
    df_final['Sharpe Ratio']=sharpe_ratio(df_final['% Change'],df_final['RF'],df_final['Vol'],fecha_fin)

    #get the VaR
    df_final['VaR']=VaR(df_final['% Change'],fecha_fin)
    
    #get the Negative Returns
    df_final['Negative Returns']=df_final['% Change'].where(df_final['% Change']<0)
    
    #get the Downside Deviation
    df_final['Downside Deviation']=downside_deviation(df_final['Negative Returns'],fecha_fin)
    
    #get the sortino ratio
    df_final['Sortino Ratio']=Sortino(df_final['% Change'],df_final['RF'],df_final['Downside Deviation'],fecha_fin)
    
    #get the Beta
    df_final['Beta']=Beta(returns,returns_aux,fecha_fin)
    
    #get the Treynor Ratio
    df_final['Treynor Ratio']=Treynor_ratio(df_final['% Change'],df_final['RF'],df_final['Beta'],fecha_fin)
    
    # get the daily active return
    df_final['Daily Active Return']=df_final['% Change'] - returns_aux
    
    #get the tracking error
    df_final['Tracking Error']=Tracking_Error(df_final['Daily Active Return'],fecha_fin)
    
    #get the info ratio
    df_final['Info. Ratio']=info_ratio(df_final['Daily Active Return'],df_final['Tracking Error'],fecha_fin)
    
    #get the Alpha
    df_final['Alpha']=Alpha(df_final['Cumulative'],df_final['Beta'],cumm_return(returns_aux),fecha_fin)
    
    #get the J Alpha
    df_final['J Alpha']=J_Alpha(df_final['% Change'],df_final['RF'],df_final['Beta'],returns_aux,fecha_fin)
    
    #get the correlation
    df_final['Correlation']=correlation(df_final['% Change'],returns_aux,fecha_fin)
    
    # get the R^2
    df_final['R^2']=df_final['Correlation']**2
    
    # get the Drawdown
    df_final['Drawdown']=Drawdown(df_final['PX_LAST'],fecha_fin)
    
    #get the Max Drawdown
    df_final['Max. Drawdown']=Max_Drawdown(df_final['PX_LAST'],fecha_fin)
    
    
    #Create the dataframe of Index final results 
    fnds_cmmdty=pd.DataFrame()
    for metric,col in df_final.items():
        fnds_cmmdty[metric]=df_final[metric].loc[fecha_fin]
    
    return fnds_cmmdty


#estas ya se podrían ir descartando

# @st.cache_data
# #ya podríamos irlas quitando
# def returns_refered_bmrk(df,returns_bmrk):
#     final=pd.DataFrame()
#     for i in range(len(df)):
#         bmrk_name=df['Associate Benchmark'][i]
#         if i == len(df)-1:
#             end_flag=df['Start Date'][i]
#             fin_aux=pd.DataFrame(returns_bmrk.loc[end_flag:,bmrk_name])
#             fin_aux.columns=['Price']
#             next
            
#         else:
#             start_flag=df['Start Date'][i]
#             end_flag=df['Start Date'][i+1] - pd.Timedelta(days=1)
#             fin_aux=pd.DataFrame(returns_bmrk.loc[start_flag:end_flag,bmrk_name])
#             fin_aux.columns=['Price']
    
#         final=pd.concat([final,fin_aux],axis=0)

#     return final

# @st.cache_data
# #ya podríamos irlas quitando
# def calculus_bmrk(_data):

#     df_prices=_data['Prices'].set_index('Date')
#     df_info=_data['Info']
#     df_weights=_data['Weights']

#     #make a split for Calculus ID and weights
#     df_weights['ID_MSTR']=df_weights['Calculus ID'].str.split(',')
#     df_weights['wgts']=df_weights['Weights'].str.split(',')
    
#     #explode the columns and convert the wgts column to float type
#     df_weights=df_weights.explode(['ID_MSTR','wgts'])
    
#     #format for the id of index and weights
#     df_weights['ID_MSTR']=df_weights['ID_MSTR'].str.strip()
#     df_weights['wgts']=df_weights['wgts'].astype(float)
#     df_weights.reset_index(drop=True,inplace=True)
    
#     #get the names of the index that use in the BMRK
#     df_weights=df_weights.merge(df_info[['ID_MSTR','Ticker']],on='ID_MSTR',how='left')

#     #This section replace the previous weight function 
#     #create a dictionary with their own weights refered to the BMRK
#     dict_bmks_wgts = dict()
#     for bmrk in df_weights['Ticker_x'].unique():
#         df_aux=df_weights[df_weights['Ticker_x']==bmrk].pivot_table(index='Start Date',columns='Ticker_y',values='wgts').rename_axis(None, axis=1)#.reset_index()
#         dict_bmks_wgts[bmrk]=df_aux

    
#     # Calculate index returns to get the bmrk returns
    
#     #df with NaN
#     df_bmk_returns = df_returns(df_prices[df_weights['Ticker_y'].unique()])
#     #df with zeros to use for acumulative calculus
#     df_bmk_returns_z=df_bmk_returns.fillna(0)
    
#     # Calculate benchmrk historical returns
#     # Dictionray with bmk returns
#     dict_bmks_returns = dict()
#     dict_bmks_returns_z = dict()
#     # Iterate each benchmark
#     for bmk in dict_bmks_wgts.keys():
#         df_wgt = dict_bmks_wgts[bmk]

#         # Reindex and ffill
#         df_wgt = (df_wgt.reindex(df_prices.index)
#                         .shift(1)
#                         .ffill()
#                         .dropna()
#                  )

#         # Prod wgt*return
#         df_wgt_z = df_wgt * df_bmk_returns_z.loc[df_wgt.index, df_wgt.columns]
#         df_wgt = df_wgt * df_bmk_returns.loc[df_wgt.index, df_wgt.columns]
        
#         # Append results in dictionary
#         dict_bmks_returns[bmk] = df_wgt.sum(axis=1).rename("Return")
#         dict_bmks_returns_z[bmk] = df_wgt_z.sum(axis=1).rename("Return")

#         returns=pd.DataFrame(dict_bmks_returns)/100
#         returns_z=pd.DataFrame(dict_bmks_returns_z)/100
        
#         #px_last for all benchmarks
#         n=100000
#         #inverse of the percentage change considering n=100,000
#         px_last=pd.DataFrame(n*(1+returns).cumprod())

#     return returns,px_last,returns_z

# @st.cache_data
# #ya podríamos irlas quitando
# def bmrk_aux_funds(df_funds_bmrk,returns_bmrk,df_prices):
#     not_aply_bmrk=df_funds_bmrk['Ticker'][df_funds_bmrk['Associate Benchmark']=='No Aplica'].unique()

#     #get the bmrk info for each fund
#     df_aux=dict()
#     for i in df_funds_bmrk['Ticker'].unique():
#         if i in not_aply_bmrk:
#             df_aux[i]=np.nan
#         else:
#             df_aux[i]=df_funds_bmrk[(df_funds_bmrk['Ticker']==i)][['Associate Benchmark','Start Date']].reset_index(drop=True)#.set_index('Start Date')

#     #excludes the funds columns that don't have a related bmrk 
#     df_aux_filter = {clave: valor for clave, valor in df_aux.items() if clave not in not_aply_bmrk}
    
#     #________________________________
#     #get the index returns that are use it as a bmrk
#     returns_index=df_returns(df_prices)
#     returns_merge=returns_bmrk.join(returns_index)
    
#     final_df=pd.DataFrame()
#     for name in df_aux_filter:
#         final_df[name]=returns_refered_bmrk(df_aux_filter[name],returns_merge)
    

#     return final_df.join(pd.DataFrame(df_aux,columns=not_aply_bmrk))

# @st.cache_data
# def bisiesto(fecha_fin):
#     year=datetime.strptime(fecha_fin, "%Y-%m-%d").year
#     if calendar.isleap(year):
#         print(f"El año {year} es bisiesto y tiene 366 días.")
#         return 366
#     else:
#         print(f"El año {year} no es bisiesto y tiene 365 días.")
#         return 365
    