import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import calendar
import streamlit as st
import Kit_Funciones as kit_funciones

#display options
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.6f}'.format(x))
pd.set_option('display.max_columns', None)

# @st.cache_data
# #si
# def df_returns(prices):
#     '''
#     Function to tcalculate the return of the prices.
#     '''
#     aux_nan={}
#     for i in prices.columns:
#         aux_nan[i]=prices[i][prices[i].isna()==True].index[1:]

#     #get df with NaN
#     returns=prices.ffill().pct_change()[1:]#.dropna()
#     for i,j in aux_nan.items():
#         returns.loc[aux_nan[i],i]=np.nan
         
#     return returns

@st.cache_data
def df_returns(prices):
    '''
    Calcula retornos manteniendo los NaNs originales de la matriz de precios.
    '''
    # 1. Calculamos retornos sobre una copia con ffill para evitar NaNs intermedios
    # que rompan la cadena de cálculo, pero pct_change generará el retorno.
    returns = prices.ffill().pct_change()
    
    returns = returns.mask(prices.isna())
    
    return returns.iloc[1:]

@st.cache_data
#ya podríamos irlas quitando
def returns_refered_bmrk(df,returns_bmrk):
    final=pd.DataFrame()
    for i in range(len(df)):
        bmrk_name=df['Associate Benchmark'][i]
        if i == len(df)-1:
            end_flag=df['Start Date'][i]
            fin_aux=pd.DataFrame(returns_bmrk.loc[end_flag:,bmrk_name])
            fin_aux.columns=['Price']
            next
            
        else:
            start_flag=df['Start Date'][i]
            end_flag=df['Start Date'][i+1] - pd.Timedelta(days=1)
            fin_aux=pd.DataFrame(returns_bmrk.loc[start_flag:end_flag,bmrk_name])
            fin_aux.columns=['Price']
    
        final=pd.concat([final,fin_aux],axis=0)

    return final

@st.cache_data
#ya podríamos irlas quitando
def calculus_bmrk(_data):

    df_prices=_data['Prices'].set_index('Date')
    df_info=_data['Info']
    df_weights=_data['Weights']

    #make a split for Calculus ID and weights
    df_weights['ID_MSTR']=df_weights['Calculus ID'].str.split(',')
    df_weights['wgts']=df_weights['Weights'].str.split(',')
    
    #explode the columns and convert the wgts column to float type
    df_weights=df_weights.explode(['ID_MSTR','wgts'])
    
    #format for the id of index and weights
    df_weights['ID_MSTR']=df_weights['ID_MSTR'].str.strip()
    df_weights['wgts']=df_weights['wgts'].astype(float)
    df_weights.reset_index(drop=True,inplace=True)
    
    #get the names of the index that use in the BMRK
    df_weights=df_weights.merge(df_info[['ID_MSTR','Ticker']],on='ID_MSTR',how='left')

    #This section replace the previous weight function 
    #create a dictionary with their own weights refered to the BMRK
    dict_bmks_wgts = dict()
    for bmrk in df_weights['Ticker_x'].unique():
        df_aux=df_weights[df_weights['Ticker_x']==bmrk].pivot_table(index='Start Date',columns='Ticker_y',values='wgts').rename_axis(None, axis=1)#.reset_index()
        dict_bmks_wgts[bmrk]=df_aux

    
    # Calculate index returns to get the bmrk returns
    
    #df with NaN
    df_bmk_returns = df_returns(df_prices[df_weights['Ticker_y'].unique()])
    #df with zeros to use for acumulative calculus
    df_bmk_returns_z=df_bmk_returns.fillna(0)
    
    # Calculate benchmrk historical returns
    # Dictionray with bmk returns
    dict_bmks_returns = dict()
    dict_bmks_returns_z = dict()
    # Iterate each benchmark
    for bmk in dict_bmks_wgts.keys():
        df_wgt = dict_bmks_wgts[bmk]

        # Reindex and ffill
        df_wgt = (df_wgt.reindex(df_prices.index)
                        .shift(1)
                        .ffill()
                        .dropna()
                 )

        # Prod wgt*return
        df_wgt_z = df_wgt * df_bmk_returns_z.loc[df_wgt.index, df_wgt.columns]
        df_wgt = df_wgt * df_bmk_returns.loc[df_wgt.index, df_wgt.columns]
        
        # Append results in dictionary
        dict_bmks_returns[bmk] = df_wgt.sum(axis=1).rename("Return")
        dict_bmks_returns_z[bmk] = df_wgt_z.sum(axis=1).rename("Return")

        returns=pd.DataFrame(dict_bmks_returns)/100
        returns_z=pd.DataFrame(dict_bmks_returns_z)/100
        
        #px_last for all benchmarks
        n=100000
        #inverse of the percentage change considering n=100,000
        px_last=pd.DataFrame(n*(1+returns).cumprod())

    return returns,px_last,returns_z

@st.cache_data
#ya podríamos irlas quitando
def bmrk_aux_funds(df_funds_bmrk,returns_bmrk,df_prices):
    not_aply_bmrk=df_funds_bmrk['Ticker'][df_funds_bmrk['Associate Benchmark']=='No Aplica'].unique()

    #get the bmrk info for each fund
    df_aux=dict()
    for i in df_funds_bmrk['Ticker'].unique():
        if i in not_aply_bmrk:
            df_aux[i]=np.nan
        else:
            df_aux[i]=df_funds_bmrk[(df_funds_bmrk['Ticker']==i)][['Associate Benchmark','Start Date']].reset_index(drop=True)#.set_index('Start Date')

    #excludes the funds columns that don't have a related bmrk 
    df_aux_filter = {clave: valor for clave, valor in df_aux.items() if clave not in not_aply_bmrk}
    
    #________________________________
    #get the index returns that are use it as a bmrk
    returns_index=df_returns(df_prices)
    returns_merge=returns_bmrk.join(returns_index)
    
    final_df=pd.DataFrame()
    for name in df_aux_filter:
        final_df[name]=returns_refered_bmrk(df_aux_filter[name],returns_merge)
    

    return final_df.join(pd.DataFrame(df_aux,columns=not_aply_bmrk))

@st.cache_data
def bisiesto(fecha_fin):
    year=datetime.strptime(fecha_fin, "%Y-%m-%d").year
    if calendar.isleap(year):
        print(f"El año {year} es bisiesto y tiene 366 días.")
        return 366
    else:
        print(f"El año {year} no es bisiesto y tiene 365 días.")
        return 365
    
@st.cache_data
#si
def cumm_return(returns,fecha_fin=None):
    """
    Function to calculate the compounded return
    *returns: Series with returns
    """
    df_cumm=((1+returns).cumprod() - 1)

    if fecha_fin!=None:
        total_ret = df_cumm.iloc[-1] 
        df_result = pd.DataFrame(total_ret).T
        df_result.index = [fecha_fin]

        return df_result
        
    return df_cumm

# @st.cache_data
# def RF(_data,fecha_fin=None):
#     df=_data['Prices'].set_index('Date')
#     df=df['USGG3M Index'].expanding().mean().iloc[1:]

#     if fecha_fin:
#         rf_df=_data['Prices'].set_index('Date')
#         rf_df=rf_df.loc[:fecha_fin]
#         rf_df=rf_df['USGG3M Index'].expanding().mean().iloc[1:]
#         st.write("RF table")

#         st.dataframe(rf_df)
#         return rf_df

#     else:
#         return df

#por corroborar
@st.cache_data
def RF(df, periodo=None, fecha_fin=None):
    '''
    df: dataframe que ya tiene los precios filtrados por la fecha fin
    '''
    df_prices=df.loc[:fecha_fin]
    
    if periodo == '1Y':
        # Tomamos los últimos 252 registros (días hábiles aprox)
        data_periodo = df_prices.iloc[-252:]
        rf_df=data_periodo['USGG3M Index'].expanding().mean().iloc[1:]
        
        return rf_df
    else:
        # Para Since Inception, YTD, MTD, Custom: 
        # Usamos lo que venga en la serie (que ya debería estar filtrada)
        data_periodo = df_prices
        f_df=data_periodo['USGG3M Index'].expanding().mean().iloc[1:]
        
        return f_df
        

@st.cache_data
def rolling_vol(returns, periodo=None, fecha_fin=None):
    df = returns.copy()
    resultados = {}

    for col in df.columns:
        serie = df.loc[:fecha_fin, col]

        if periodo == '1Y':
            data_clean = serie.dropna().iloc[-252:]
        else:
            # Cubre 'Since Inception', 'YTD', 'MTD' y 'Custom' 
            # porque el filtro ya se aplicó antes de entrar aquí.
            data_clean = serie.dropna()

        if len(data_clean) < 2:
            resultados[col] = np.nan
        else:
            
            resultados[col] = data_clean.std() * np.sqrt(252)
            
    df_final = pd.DataFrame(resultados, index=[fecha_fin])
    df_final.index.name = 'Date'
    
    return df_final

@st.cache_data
def sharpe_ratio(returns, RF, volat, fecha_fin, periodo=None):
    df = returns.copy()
    resultados = {}
    
    for col in df.columns:
        serie = df.loc[:fecha_fin, col]

        if periodo == '1Y':
            data_clean = serie.dropna().iloc[-252:]
        else:
            # valido para: Since Inception, YTD, MTD y Custom Date
            # Porque el DataFrame ya viene filtrado desde fuera
            data_clean = serie.dropna()

        # Cálculo del Sharpe Ratio (Validando que existan datos suficientes)
        if len(data_clean) < 2:
            resultados[col] = np.nan
        else:
            avg_daily = data_clean.mean()
            ann_return = ((1 + avg_daily)**252) - 1
            
            rf_val = RF.loc[fecha_fin]
            volat_val = volat.loc[fecha_fin, col]
            
            #consideradno divisiones entre cero
            if volat_val != 0 and not np.isnan(volat_val):
                resultados[col] = (ann_return - rf_val) / volat_val
            else:
                resultados[col] = np.nan
            
    # Resultado final
    df_final = pd.DataFrame(resultados, index=[fecha_fin])
    df_final.index.name = 'Date'
    return df_final



@st.cache_data
def downside_deviation(returns, fecha_fin, periodo=None):
    df = returns.copy()
    resultados = {}
    
    for col in df.columns:
        serie = df.loc[:fecha_fin, col]
        
        if periodo == '1Y':
            data_subset = serie.dropna().iloc[-252:]
        else:
            data_subset = serie.dropna()

        neg_returns = data_subset[data_subset < 0]
        
        if len(neg_returns) < 2:
            resultados[col] = np.nan
        else:
            aux = (neg_returns**2).sum() / len(data_subset) # Nota: Excel suele dividir entre N total, no solo N negativos
            down_dev = np.sqrt(aux) * np.sqrt(252)
            resultados[col] = down_dev
            
    df_final = pd.DataFrame(resultados, index=[fecha_fin])
    df_final.index.name = 'Date'
    return df_final

@st.cache_data
def negative_returns(returns, fecha_fin, periodo=None):
    df = returns.copy()
    resultados = {}
    

    for col in df.columns:
        serie = df.loc[:fecha_fin, col]

        if periodo == '1Y':
            data_subset = serie.dropna().iloc[-252:]
        else:
            data_subset = serie.dropna()

        neg_returns = data_subset[data_subset < 0]
        resultados[col] = neg_returns

    df_final = pd.concat(resultados, axis=1)
    df_final.index.name = 'Date'
    
    return df_final

@st.cache_data
def VaR(returns, fecha_fin, periodo=None,conf_lvl=99, ):
    df = returns.copy()
    resultados = {}
    
    # El nivel de confianza suele ser 95 o 99. 
    # Para el cuantil usamos alpha (p.ej. 0.05 para 95%)
    alpha = 1 - (conf_lvl / 100)

    for col in df.columns:
        serie = df.loc[:fecha_fin, col]
        
        if periodo == '1Y':
            data_subset = serie.dropna().iloc[-252:]
        else:
            data_subset = serie.dropna()

        if len(data_subset) < 2:
            resultados[col] = np.nan
        else:
            # Cálculo del cuantil histórico directo
            resultados[col] = data_subset.quantile(alpha)
            
    df_final = pd.DataFrame(resultados, index=[fecha_fin])
    df_final.index.name = 'Date'
    return df_final

# @st.cache_data
# #si
# def Sortino(returns,RF,down_dev,fecha_fin):
#     i=returns.index.get_loc(fecha_fin)
#     sortino=pd.DataFrame()
#     aux=returns.loc[:fecha_fin].rolling(i+1,min_periods=1).mean()
#     sortino=(((1+aux[-1:])**252 - 1) - RF.loc[fecha_fin])/down_dev
    
#     return sortino

@st.cache_data
def Sortino(returns, RF, down_dev, fecha_fin, periodo=None):
    df = returns.copy()
    resultados = {}

    for col in df.columns:
        serie = df.loc[:fecha_fin, col]
        
        if periodo == '1Y':
            data_subset = serie.dropna().iloc[-252:]
        else:
            data_subset = serie.dropna()

        if len(data_subset) < 2:
            resultados[col] = np.nan
        else:
            avg_daily = data_subset.mean()
            ann_return = ((1 + avg_daily)**252) - 1
            
            rf_val = RF.loc[fecha_fin]
            dd_val = down_dev.loc[fecha_fin, col]
            
            if dd_val != 0 and not np.isnan(dd_val):
                resultados[col] = (ann_return - rf_val) / dd_val
            else:
                resultados[col] = np.nan

    df_final = pd.DataFrame(resultados, index=[fecha_fin])
    df_final.index.name = 'Date'
    
    return df_final


@st.cache_data
#si solo recuerda cambiar los ddof=1 para que sea muestral
def Beta(returns,returns_aux,fecha_fin):
    beta={}
    for i in returns.columns.unique():

        aux_1=returns[i].loc[:fecha_fin]
        aux_2=returns_aux[i].loc[:fecha_fin]
    
        #se utiliza la covarianza muestral ddof=1, no coincidirá con los calculos del excel
        beta[i]=aux_1.cov(aux_2,ddof=0)/aux_2.var()

    beta['Start Date']=fecha_fin
    beta=pd.DataFrame(beta,index=['Start Date'])
    beta.set_index('Start Date',inplace=True, drop=True)
    beta.index = pd.to_datetime(beta.index)
    beta.index = beta.index.strftime('%Y-%m-%d')
    
    return beta
@st.cache_data
#si
def Alpha(df_cumm,beta,bmrk_cumm,fecha_fin):
    aux=df_cumm.loc[fecha_fin]-beta.loc[fecha_fin]*bmrk_cumm.loc[fecha_fin]
    aux=pd.DataFrame(aux).T
    alpha=aux.rename(index={0:fecha_fin})
    
    return alpha
@st.cache_data
#si
def J_Alpha(returns,RF,beta,returns_bmrk,fecha_fin):
    i=returns.index.get_loc(fecha_fin)
    aux=returns.loc[:fecha_fin].rolling(i+1,min_periods=1).mean()
    aux_2=returns_bmrk.loc[:fecha_fin].rolling(i+1,min_periods=1).mean()

    j_alpha=((1+aux[-1:])**252-1) - (RF.loc[fecha_fin]+beta.loc[fecha_fin]*((1+aux_2[-1:])**252-1-RF.loc[fecha_fin]))
    return j_alpha
@st.cache_data
#si
def Treynor_ratio(returns,RF,beta,fecha_fin):
    i=returns.index.get_loc(fecha_fin)
    aux=returns.loc[:fecha_fin].rolling(i+1,min_periods=1).mean()
    treynor=(((1+aux[-1:])**252-1)-RF.loc[fecha_fin])/beta.loc[fecha_fin]
    
    return treynor
@st.cache_data
#si
def Tracking_Error(d_active_rtrns,fecha_fin):
    i=d_active_rtrns.index.get_loc(fecha_fin)
    t_e=d_active_rtrns.loc[:fecha_fin].rolling(i+1,min_periods=1).std()*np.sqrt(252)
    
    return t_e[-1:]
@st.cache_data
#si
def info_ratio(d_active_rtrns,tracking_e,fecha_fin):
    i=d_active_rtrns.index.get_loc(fecha_fin)
    aux=d_active_rtrns.loc[:fecha_fin].rolling(i+1,min_periods=1).mean()
    info_r=((1+aux[-1:])**252-1)/tracking_e

    return info_r
@st.cache_data
#si
def correlation(returns,returns_bmrk,fecha_fin):
    aux=returns.loc[:fecha_fin]
    aux_2=returns_bmrk.loc[:fecha_fin]

    corr=aux.corrwith(aux_2)
    corr=pd.DataFrame(corr).T
    corr=corr.rename(index={0:fecha_fin})
    
    return corr
@st.cache_data
#si
def Drawdown(price,fecha_fin):
    new_fecha_fin=(datetime.strptime(fecha_fin, '%Y-%m-%d').date()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    aux=price.loc[fecha_fin]
    aux2=price.loc[:new_fecha_fin].max()
    drawdown=aux/aux2 - 1
    drawdown=pd.DataFrame(drawdown).T
    drawdown=drawdown.rename(index={0:fecha_fin})
    
    return drawdown
@st.cache_data
#si
def Max_Drawdown(prices,fecha_fin):
    flag=prices.index.get_loc(fecha_fin)
    date_range=prices.index[:flag+1].strftime('%Y-%m-%d').to_list()
    
    max_drawdown=pd.DataFrame()
    for date in date_range:
        aux=Drawdown(prices,date)
        max_drawdown=pd.concat([max_drawdown,aux])
    # print(max_drawdown)
    max_drawdown=pd.DataFrame(max_drawdown.min()).T
    max_drawdown=max_drawdown.rename(index={0:fecha_fin})
    
    return max_drawdown

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
def Funds_Commodity_test(_data,fecha_fin,periodicity=None):
    df_final={}
    
    df_info=_data['Info']
    
    #get the info about the prices of funds and commodity
    index_cols=df_info[(df_info['Type']=='Fund') | (df_info['Type']=='Commodity')]['Ticker'].to_list()
    df_prices=_data['Prices'].set_index('Date')
    df_OCW=_data['OCWHAUA LX Equity'].set_index('Date')
    df_prices=df_prices.join(df_OCW,how='left')
    ######considerar poner un ffill para que auqnue sea día festivo arroje la fehca
    ###### pero no afecte directamente a los calculos de las metricas pero ya en el rasultado final
    df_prices=df_prices[index_cols]#.ffill()
    
    #get the funds returns
    returns=df_returns(df_prices)
    returns_z=returns.fillna(0)
    
    #__
    # get the columns and prices related to indexs
    index_cols_index=df_info[df_info['Type']=='Index']['Ticker'].to_list()
    df_prices_index=_data['Prices'][index_cols_index+['Date']].set_index('Date')
    
    #se obtiene la información de los fondos que tienen bmrks
    df_funds_bmrk=_data['Funds - BMRK']
    df_funds_bmrk=df_funds_bmrk.replace('2015-12-03','2015-12-04')
   
    #get the bmrk returns
    returns_bmrk,_,_=calculus_bmrk(_data)
    returns_aux=bmrk_aux_funds(df_funds_bmrk,returns_bmrk,df_prices_index)
    

    fecha_inicio=None
    #_______________________________________________
    rf_df=RF(_data)
    df_cumm_returns=cumm_return(returns_z)

    if periodicity=='1Y':

        fecha_inicio_aux=kit_funciones.start_dt(fecha_fin,n_months=12)
        id_proximo=df_prices.index.get_indexer([fecha_inicio_aux], method='backfill')[0]
        fecha_inicio=df_prices.index[id_proximo]

        if fecha_inicio>pd.to_datetime('2015-12-03'):
            st.write(fecha_inicio)
            st.write(fecha_fin)
            df_prices=df_prices.loc[fecha_inicio:fecha_fin]
            returns=df_returns(df_prices)
            returns_z=returns.fillna(0)

            st.dataframe(df_prices)
            st.write(df_prices.shape)

            # st.dataframe(returns_all)
            # st.write(returns_all.shape)
            
            st.dataframe(returns)
            st.write(returns.shape)

            returns_aux=returns_aux.loc[fecha_inicio:fecha_fin]
            df_cumm_returns=cumm_return(returns_z,fecha_fin)
            rf_df=RF(_data,fecha_inicio,fecha_fin)

            # vol=rolling_vol(returns,periodicity,fecha_fin)
            # st.write("_______________vol")
            # st.dataframe(vol)
            # print(vol)
            # sharpe=sharpe_ratio(returns,rf_df,vol,fecha_fin,periodicity)
            # st.write("sharpeeee")
            # st.dataframe(sharpe)
            # st.dataframe(vol.style.format("{:.8f}"))

            # st.dataframe(rf_df)
            # st.dataframe(returns)

        else: 
            return st.warning(f'It is necessary to enter at least {periodicity} higher than 2015-12-03. Your initial date is: {fecha_inicio}')
            
    elif periodicity=='YTD':
        # year_date=datetime.strptime(fecha_fin, "%Y-%m-%d").year
        # fecha_inicio=datetime.strptime(f'{year_date}-01-01', "%Y-%m-%d").strftime('%Y-%m-%d')
        fecha_inicio=kit_funciones.start_dt(fecha_fin,YTD=True)
        if fecha_inicio>'2015-12-03':
            # print('Sí es mayor a 365 días')
            returns=returns.loc[fecha_inicio:fecha_fin]
            #NOTA: para el caclculo del max drawdown puede haber diferenecias ya que para crear el primer dato de drawdown, sera NA
            df_prices=df_prices.loc[fecha_inicio:fecha_fin]
            returns_z=returns_z.loc[fecha_inicio:fecha_fin]
            returns_aux=returns_aux.loc[fecha_inicio:fecha_fin]

            # IMPORTANTE: Considrar que para los indices se maneja diferente el calculo del acumulado para el YTD
            rf_df=RF(_data,fecha_inicio,fecha_fin)
            
        else:
            return f'It is necessary to enter a higher date than 2015-12-03. Your initial date is: {fecha_inicio}'
        
    else:
        
        pass
        #                                     _______________metrics for funds and commodity_______________
    #get the PX_LAST
    df_final['PX_LAST']=df_prices.ffill()  #aqui a tendrá los ffill
    
    #get the % Change
    df_final['% Change']=returns #aqui ya tendra los retornos con NaN
    
    #get the cummulative df
    df_final['Cumulative']=df_cumm_returns  #aqui usa los retornos con 0
    
    #get the RF df
    df_final['RF']=rf_df
    
    #get the volatility df
    df_final['Vol']=rolling_vol(df_final['% Change'],periodicity,fecha_fin)#usa los retornos con NaN
    
    #get the Sharpe Ratio
    df_final['Sharpe Ratio']=sharpe_ratio(df_final['% Change'],df_final['RF'],df_final['Vol'],fecha_fin,periodicity)

    #get the Negative Returns
    df_final['Negative Returns']=negative_returns(df_final['% Change'],fecha_fin,periodicity) #df_final['% Change'].where(df_final['% Change']<0)
    
    #get the Downside Deviation
    df_final['Downside Deviation']=downside_deviation(df_final['Negative Returns'],fecha_fin,periodicity)
    
    #get the sortino ratio
    df_final['Sortino Ratio']=Sortino(df_final['% Change'],df_final['RF'],df_final['Downside Deviation'],fecha_fin,periodicity)

    #get the VaR
    df_final['VaR']=VaR(df_final['% Change'],fecha_fin,periodicity)
    
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
