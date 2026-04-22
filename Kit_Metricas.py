import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def df_returns(prices):
    """
    Calcula retornos porcentuales manteniendo la posición de los NaNs originales.

    Parámetros:
    prices (DataFrame): Matriz de precios.

    Retorna:
    DataFrame: Retornos calculados a partir de la segunda fila.
    Serie: Última fecha con datos válidos para cada activo.
    """
    returns = prices.ffill().pct_change()
    returns = returns.mask(prices.isna())
    fechas_reales=returns.apply(lambda x: x.last_valid_index())
    
    return returns.iloc[1:], fechas_reales

@st.cache_data
def cumm_return(returns,fecha_fin=None):
    """
    Calcula el retorno acumulado histórico o a una fecha límite

    Parámetros:
    returns (DataFrame): Dataframe de retornos.
    fecha_fin (str/datetime, opcional): Fecha de corte para obtener solo el retorno total.

    Retorna:
    DataFrame: Serie de retornos acumulados o el resultado final transpuesto del dataframe.
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
    """
    Calcula la tasa libre de riesgo (Risk-Free) promedio acumulada.

    Parámetros:
    df_prices (DataFrame): Datos que incluyen la columna 'USGG3M Index'.
    fecha_fin (opcional): No se utiliza actualmente en el cálculo, pero se mantiene por consistencia.

    Retorna:
    Series: Promedio expansivo del índice USGG3M expresado en decimales.
    """

    rf_expanding = df_prices['USGG3M Index'].expanding().mean()
    
    # Retornamos la serie completa para poder indexar después por fecha_real
    return rf_expanding/100

@st.cache_data
def rolling_vol(returns, fechas_reales):
    """
    Calcula la volatilidad anualizada expandida basada en la última fecha válida de cada activo.

    Parámetros:
    returns (DataFrame): Dataframe de retornos.
    fechas_reales (Series): Serie que contiene la última fecha de datos para cada columna.

    Retorna:
    DataFrame: Un renglón con la volatilidad anualizada (Vol) calculada para cada activo.
    """
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
    """
    Calcula el Sharpe Ratio anualizado.

    Parámetros:
    returns (DataFrame): Retornos de los activos.
    rf_series (Series): Tasa libre de riesgo (RF) calculada.
    vol_df (DataFrame): Volatilidad anualizada calculada.
    fechas_reales (Series): Última fecha válida por activo.

    Retorna:
    DataFrame: El ratio Sharpe por activo en una sola fila.
    """
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
    """
    Calcula la desviación estándar de los retornos negativos (anualizada).

    Parámetros:
    negative_returns (DataFrame): Dataframe que solo contiene retornos menores a cero.
    fechas_reales (Series): Última fecha válida por activo.

    Retorna:
    DataFrame: La desviación a la baja (Downside Deviation) por activo.
    """
    downside_returns = negative_returns

    dd_full = np.sqrt((downside_returns**2).expanding().mean()) * np.sqrt(252)
    
    # 3. Extraemos el valor en la fecha real
    dd_final = {col: dd_full.loc[fechas_reales[col], col] 
                if pd.notnull(fechas_reales.get(col)) else np.nan 
                for col in negative_returns.columns}
    
    return pd.DataFrame(dd_final, index=["Downside Deviation"])

@st.cache_data
def negative_returns(returns):
    """
    Filtra la matriz de retornos para obetener solo los valores negativos.

    Parámetros:
    returns (DataFrame): Dataframe original de retornos.

    Retorna:
    DataFrame: Misma estructura que la entrada, pero con NaN en los valores positivos.
    """
    return returns[returns < 0]

@st.cache_data
def VaR(returns, fechas_reales, conf_lvl=99):
    """
    Calcula el Valor en Riesgo (VaR) histórico mediante el método de percentiles.

    Parámetros:
    returns (DataFrame): Retornos de los activos.
    fechas_reales (Series): Última fecha válida por activo.
    conf_lvl (int): Nivel de confianza (por defecto 99%).

    Retorna:
    DataFrame: El VaR histórico correspondiente al nivel de confianza.
    """
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
    """
    Calcula el Sortino Ratio.

    Parámetros:
    returns (DataFrame): Retornos de los activos.
    rf_series (Series): Tasa libre de riesgo (RF) calculada.
    dd_df (DataFrame): Desviación a la baja calculada.
    fechas_reales (Series): Última fecha válida por activo.

    Retorna:
    DataFrame: El ratio Sortino por activo.
    """
    avg_daily = returns.mean()
    ann_return = ((1 + avg_daily)**252) - 1
    
    # Alineamos RF y Downside Deviation a la fecha real
    rf_vals = pd.Series({c: rf_series.loc[fechas_reales[c]] if pd.notnull(fechas_reales[c]) else np.nan for c in returns.columns})
    down_devs = dd_df.iloc[0]
    
    sortino = (ann_return - rf_vals) / down_devs
    return pd.DataFrame(sortino).T.rename(index={0: 'Sortino Ratio'})
    
@st.cache_data
def Beta(returns, returns_bmrk, fechas_reales=None):
    """
    Calcula la Beta de cada activo respecto a su benchmark de referencia.

    Parámetros:
    returns (DataFrame): Retornos de los activos.
    returns_bmrk (DataFrame): Retornos del benchmark.
    fechas_reales (Series, opcional): Última fecha para delimitar el cálculo.

    Retorna:
    DataFrame: La Beta (sensibilidad) calculada para cada activo.
    """
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
    """
    Calcula el Alpha: Retorno acumulado del fondo - (Beta * Retorno acumulado Benchmark). 

    Parámetros:
    df_cumm (DataFrame): Retornos acumulados de los fondos.
    beta_df (DataFrame): Betas calculadas previamente.
    bmrk_cumm (DataFrame): Retornos acumulados del benchmark.
    fechas_reales (Series): Última fecha válida por activo.
    """
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
    """
    Calcula el Alpha de Jensen anualizado ajustado por riesgo (CAPM).
    J_Alpha = R_anualizado - [RF + Beta * (R_bmrk_anualizado - RF)]

    Parámetros:
    returns (DataFrame): Retornos diarios de los fondos.
    rf_series (Series): Tasa libre de riesgo (RF).
    beta_df (DataFrame): Betas de los activos.
    returns_bmrk (DataFrame): Retornos diarios del benchmark.
    """

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
    """
    Mide el retorno excedente por cada unidad de riesgo sistemático (Beta).

    Parámetros:
    returns (DataFrame): Retornos de los fondos.
    rf_series (Series): Tasa libre de riesgo.
    beta_df (DataFrame): Betas calculadas.
    fechas_reales (Series): Última fecha válida.
    """
    avg_daily = returns.mean()
    ann_return = ((1 + avg_daily)**252) - 1
    rf_vals = pd.Series({c: rf_series.loc[fechas_reales[c]] if pd.notnull(fechas_reales[c]) else np.nan for c in returns.columns})
    betas = beta_df.iloc[0]
    
    treynor = (ann_return - rf_vals) / betas
    return pd.DataFrame(treynor).T.rename(index={0: 'Treynor Ratio'})

@st.cache_data
def daily_active_returns(returns, returns_bmrk, fechas_reales=None):
    """
    Genera la serie de tiempo de retornos activos (Fondo - Benchmark).

    Parámetros:
    returns (DataFrame): Retornos de los fondos.
    returns_bmrk (DataFrame): Retornos del benchmark.
    """
    active_returns_dict = {}

    for col in returns.columns:
        fondo_r = returns[col]
        bmrk_r = returns_bmrk[col]

        fecha_corte = fechas_reales.get(col) if fechas_reales is not None else None
        
        if pd.notnull(fecha_corte):
            fondo_r = fondo_r.loc[:fecha_corte]
            bmrk_r = bmrk_r.loc[:fecha_corte]
        
        # (returns con 0s) - (Benchmark)
        active_returns_dict[col] =  fondo_r.fillna(0) - bmrk_r#fondo_r.reindex(bmrk_r.index).fillna(0) - bmrk_r

    df_active_returns = pd.DataFrame(active_returns_dict)

    return df_active_returns

@st.cache_data
def Tracking_Error(returns, returns_bmrk, fechas_reales=None):
    """
    Calcula el Tracking Error: Calcula la volatilidad anualizada de los retornos activos (error de seguimiento).
    std(daily_active_rtrns)*sqrt(252)
    Usa fillna(0) para el caso de los retornos diarios.

    Parámetros:
    returns (DataFrame): Retornos de los fondos.
    returns_bmrk (DataFrame): Retornos del benchmark.
    """
    '''
    
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

        active_diff = (fondo_r.fillna(0) - bmrk_r) #(fondo_r.reindex(bmrk_r.index).fillna(0) - bmrk_r)

        #Cálculo del Tracking Error Anualizado
        if len(active_diff) > 1:
             te_values[col]= active_diff.std() * np.sqrt(252)
        else:
            te_values[col] = np.nan

        
            
    return pd.DataFrame(te_values, index=['Tracking Error'])

@st.cache_data
def info_ratio(returns, returns_bmrk, df_te, fechas_reales=None):
    """
    Calcula el Information Ratio: Mide la capacidad del gestor para generar retornos excedentes sobre el Tracking Error.
    (Retorno Activo Anualizado) / Tracking Error. Usa fillna(0) para que el retorno activo sea consistente con el TE calculado.

    Parámetros:
    returns (DataFrame): Retornos de los fondos.
    returns_bmrk (DataFrame): Retornos del benchmark.
    df_te (DataFrame): Tracking Error calculado.
    """
    '''
    
    
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
    """
    Calcula el coeficiente de correlación de Pearson respecto al benchmark.

    Parámetros:
    returns (DataFrame): Retornos de los activos.
    returns_bmrk (DataFrame): Retornos del benchmark.
    """
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
def Drawdown(returns, fechas_reales):
    """
    Calcula el Drawdown: la caída porcentual actual desde el máximo histórico alcanzado.

    Parámetros:
    returns (DataFrame): Retornos de los fondos.
    fechas_reales (Series): Fecha de corte para evaluar el nivel actual.
    """
    serie_precios = (1 + returns).cumprod()
    previous_pick = serie_precios.cummax()

    all_drawdowns = (serie_precios / previous_pick) - 1
    
    #validación de NaT
    dd_final = {}
    for c in returns.columns:
        fecha = fechas_reales.get(c)
        
        
        if pd.notnull(fecha) and fecha in all_drawdowns.index:
            dd_final[c] = all_drawdowns.loc[fecha, c]
        else:
            dd_final[c] = np.nan
            
    return pd.DataFrame(dd_final, index=["Drawdown"])

@st.cache_data
def Max_Drawdown(returns, fechas_reales):
    """
    Calcula el Max Drowdown: Identifica la caída máxima (peor racha de pérdidas) en el periodo analizado.

    Parámetros:
    returns (DataFrame): Retornos de los fondos.
    fechas_reales (Series): Fecha límite para la búsqueda histórica.
    """
    
    serie_precios = (1 + returns).cumprod()
    previous_pick = serie_precios.cummax()

    all_drawdowns = (serie_precios / previous_pick) - 1

    mdd_final = {}
    for c in returns.columns:
        fecha = fechas_reales.get(c)

        if pd.notnull(fecha):
            
            # Filtrar lso drawdown hasta la fecha
            mdd_full = all_drawdowns.loc[:fecha,c]

            if not mdd_full.empty:
                mdd_final[c] = mdd_full.min()
            else:
                mdd_final[c] = np.nan
        else:
            mdd_final[c] = np.nan
            
    return pd.DataFrame(mdd_final, index=["Max Drawdown"])

