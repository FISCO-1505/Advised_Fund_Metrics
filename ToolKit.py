# Finance Tool Kit

# Import libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
import yfinance as yf
import datetime as dt
from pandas.tseries.offsets import BDay

# Download Prices or Returns from Yahoo Finance
def get_prices(tickers, start=None, end=None, period="1y", freq="1d", rename=None, only_close=True):
	"""
	Descargar precios de los tickers dado el periodo y frecuencia.
	* tickers: list
	* start and end: str or datetime YYYY-MM-DD
	* period: str (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
	* freq: str (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo) intraday only if period < 60 days
	* rename: Optional list
	* only_close: Optional to download only close prices
	"""
	# Download prices
	if start is not None:
		prices = yf.download(tickers=tickers, start=start, end=end, interval=freq, progress=False).dropna(thresh=len(tickers)*.99).fillna(method="ffill")
	else:
		prices = yf.download(tickers=tickers, period=period, interval=freq, progress=False).dropna(thresh=len(tickers)*.99).fillna(method="ffill")
	
	# Drop all columns except Adjusted Close
	if only_close == True:
		if len(tickers) > 1:
			prices = prices["Adj Close"]
		else:
			cols = prices.columns
			prices = prices.drop(columns = cols[cols!="Adj Close"])
	
	# Rename it
	if rename is not None:
		if len(tickers) > 1:
			dic_rename = {}
			for i in range(len(tickers)):
				dic_rename[tickers[i]] = rename[i]
			prices = prices.rename(columns=dic_rename)
		else:
			dic_rename = {"Adj Close": rename}
			prices = prices.rename(columns=dic_rename)
	
	# Reindex to period
	if freq == "1d":
		prices = prices.to_period("D")
	if freq == "1mo":
		prices = prices.to_period("M")
	if freq == "1y":
		prices = prices.to_period("Y")
	
	return prices

def get_returns(tickers, start=None, end=None, period="1y", freq="1d", rename=None, only_close=True):
	"""
	Descargar rendimientos de tickers dado el periodo y la frecuencia.
	* tickers: list
	* start and end: str or datetime YYYY-MM-DD
	* period: str (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
	* freq: str (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo) intraday only if period < 60 days
	* rename: Optional list
	* only_close: Optional to download only close prices
	"""
	# Restar 1 al tiempo para calcular rendimientos
	if start is not None:
		# Convertir string to datetime
		if type(start) is str:
			start = dt.datetime.fromisoformat(start)
		
		if freq == "1d" or freq == "5d":
			delta_time = BDay(1)
			start -= delta_time
		if freq == "1wk":
			delta_time = BDay(5)
			start -= delta_time
		if freq == "1mo":
			delta_time = dt.timedelta(days=22)
			start -= delta_time
	
	# Calcular rendimientos
	returns = get_prices(tickers, start, end, period, freq, rename, only_close).pct_change().dropna()
	return returns

# Risk Measures
def drawdown(returns_series: pd.Series):
	"""
	Takes time series of assets  returns
	Computes and returns DataFrame contains:
	wealth index
	previous peaks
	percent drawdowns
	"""
	wealth_index = 100*(1+returns_series).cumprod()
	prev_peaks = wealth_index.cummax()
	drawdowns = (wealth_index - prev_peaks)/prev_peaks
	return pd.DataFrame({"Wealth" : wealth_index, 
						 "Peaks" : prev_peaks,
						 "Drawdown" : drawdowns})

def max_drawdown(returns):
	"""
	Returns only de max drawdown in absolute terms.
	"""
	dd = returns.aggregate(lambda returns: drawdown(returns).Drawdown.min())
	return np.abs(dd)


def beta(returns, mkt="IPC"):
	"""
	Calcular la beta de los tickers dados los rendimientos.
	* returns: DataFrame con rendimientos
	* mkt: nombre de ticker del mercado
	"""	
	# Calcular varianza mkt y beta
	mkt_var = returns.iloc[:,-1].var()
	betas = returns.cov().iloc[:,-1] / mkt_var
	
	return betas.drop(betas.index[-1]).rename("Beta")

def semideviation(returns):
	"""
	Returns the semideviation, negative semideviation of returns
	"""
	is_negative = returns < 0
	return returns[is_negative].std(ddof=1)

def var_historic(returns, level=5):
	"""
	VaR Historic
	* returns: Series or DataFrame
	* level: (1-confidence level) must be int 0-100
	"""
	if isinstance(returns, pd.DataFrame):
		return returns.aggregate(var_historic, level=level)
	elif isinstance(returns, pd.Series):
		return -np.percentile(returns, level)
	else:
		raise TypeError("Expected returns to be Series or DataFrame")

def var_gaussian(returns, level=5, modified=False):
	"""
	Returns the Parametric Gaussian VaR of a Series or DataFrame
	* returns: Series or DataFrame
	* level: (1-confidence level) must be int 0-100
	"""
	# Compute de Z score assuming it was Gaussian
	z = stats.norm.ppf(level/100)
	
	if modified:
		# Modify the Z score based on observed skewness and kurtosis
		s = skewness(returns)
		k = kurtosis(returns)
		z = (z+
				(z**2 - 1)*s/6 +
				(z**3 -3*z)*(k-3)/24 -
				(2*z**3 - 5*z)*(s**2)/36
			)
		
	return -(returns.mean() + z*returns.std(ddof=1))

def cvar_historic(returns, level=5):
	"""
	Computes the Conditional VaR of Series or DataFrame
	* returns: Series or DataFrame
	* level: (1-confidence level) must be int 0-100
	"""
	if isinstance(returns, pd.Series):
		is_beyond = returns <= -var_historic(returns, level=level)
		return -returns[is_beyond].mean()
	elif isinstance(returns, pd.DataFrame):
		return (returns.aggregate(cvar_historic, level=level)).rename("CVaR Hist")
	else:
		raise TypeError("Expecetd returns to be a Series or a DataFrame")

# Statistic Measures
def skewness(returns):
	"""
	Calculate skewness
	"""
	if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
		return returns.skew()
	else:
		raise TypeError("Expected returns to be a Series or DataFrame")

def kurtosis(returns):
	"""
	Calculate kurtosis
	"""
	if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
		return returns.kurtosis()
	else:
		raise TypeError("Expected returns to be a Series or DataFrame")

def is_normal(returns, level=0.01):
	"""
	Applies Jarque-Bera test to determine if a series is normal or not
	Test is applied at the 1% level by default
	Returns True if the hypothesis is accepted, False otherwise
	"""
	if isinstance(returns, pd.Series):
		statistic, p_value = stats.jarque_bera(returns)
		return p_value > level
	elif isinstance(returns, pd.DataFrame):
		p_value = returns.aggregate(is_normal)
		return p_value > level
	else:
		raise TypeError("Expected returns to be a Series or DataFrame")

# Returns Measures
def ann_vol(returns, periods):
	"""
	Annualizes the vol of a set of returns
	We should infer the periods per year
	"""
	return returns.std()*np.sqrt(periods)

def ann_return(returns, periods, actual_days=False):
	"""
	Annualizes the acum return of a set of returns
	We should infer the periods per year
	"""
	prod = (1+returns).prod()
	if actual_days:
		n_periods = returns.index[-1] - returns.index[0]
		n_periods = n_periods.days
	else:
		n_periods = returns.shape[0]
	return (prod)**(periods/n_periods)-1

def sharpe_ratio(returns, riskfree, periods, actual_days=False):
	"""
	Computes the annualized Sharpe Ratio of a set of returns
	"""
	annreturn = ann_return(returns, periods, actual_days)
	annvol = ann_vol(returns, periods)
	if isinstance(returns, pd.DataFrame):
		return ((annreturn - riskfree)/annvol).rename("Sharpe Ratio")
	else:
		return (annreturn - riskfree)/annvol

def treynor_ratio(returns, riskfree, periods, Beta=None, actual_days=False):
	"""
	Return the Traynor Ratio of a set of returns
	"""
	# Calcular Traynor Ratio
	if isinstance(returns, pd.DataFrame):
		annreturn = ann_return(returns, periods, actual_days)
		return ((annreturn - riskfree) / Beta).rename("Traynor Ratio")
	elif isinstance(returns, pd.Series):
		annreturn = ann_return(returns, periods, actual_days)
		return (annreturn - riskfree)/Beta
	else:
		raise TypeError("Returns must be DataFrame or Series")

def romad(returns, periods, actual_days=False):
	"""
	Returns the Return Over MaxDrawdown
	"""
	if isinstance(returns, pd.DataFrame):
		annreturn = ann_return(returns, periods, actual_days)
		maxdd = max_drawdown(returns)
		return (annreturn/maxdd).rename("RoMaD")
	elif isinstance(returns, pd.Series):
		annreturn = ann_return(returns, periods, actual_days)
		maxdd = max_drawdown(returns)
		return (annreturn/maxdd)
	else:
		raise TypeError("Returns must be DataFrame or Series")

def sortino(returns, mar, periods, only_negative=False, actual_days=False):
	"""
	Returns Sortino Ratio given the returns and the Min acceptale return
	"""
	annreturn = ann_return(returns, periods, actual_days)
	if only_negative:
		stdmar = semideviation(returns) * np.sqrt(periods)
	else:
		adj_mar = (1+mar)**(1/periods)-1
		ret_mar = returns < adj_mar
		stdmar = (returns[ret_mar]).std() * np.sqrt(periods)
	
	# Calculate Sortino
	if isinstance(returns, pd.DataFrame):
		return ((annreturn - mar)/stdmar).rename("Sortino Ratio")
	elif isinstance(returns, pd.Series):
		return (annreturn - mar)/stdmar
	else:
		raise TypeError("Returns must be DataFrame or Series")

def alpha_jensen(returns, riskfree, periods, mkt="IPC", Beta=None, actual_days=False):
	"""
	Returns Jensen's alpha of returns
	"""
	# Calculate CAPM
	if Beta is None:
		Beta = beta(list(returns.columns), mkt=mkt, start=returns.index[0], end=returns.index[-1], freq=str("1"+returns.index.freqstr))
	
	if mkt == "IPC":
		mkt_ret = get_returns(tickers="^MXX", start=returns.index[0], end=returns.index[-1], freq=str("1"+returns.index.freqstr))
	elif mkt == "S&P":
		mkt_ret = get_returns(tickers="^GSPX", start=returns.index[0], end=returns.index[-1], freq=str("1"+returns.index.freqstr))
	else:
		mkt_ret = get_returns(tickers=mkt, start=returns.index[0], end=returns.index[-1], freq=str("1"+returns.index.freqstr))
	
	annmkt = ann_return(mkt_ret, periods, actual_days)
	capm = riskfree + Beta*(annmkt - riskfree)
	
	# Calculate Jensen's Alpha
	j_alpha = ann_return(returns, periods, actual_days) - capm
	if isinstance(returns, pd.DataFrame):
		return j_alpha.rename("Jensen's Alpha")
	elif isinstance(returns, pd.Series):
		return j_alpha
	else:
		raise TypeError("Returns must be DataFrame or Series")

# Portfolio Functions
def port_return(weights, returns):
	"""
	Weights --> Return
	"""
	return weights.T @ returns

def port_vol(weights, covmat):
	"""
	Weights --> Volatility
	""" 
	return np.sqrt(weights.T @ covmat @ weights)


def min_vol(target_return, exp_returns, cov):
	"""
	Target_return --> W
	* target_return: float
	* exp_returns: array with expected returns
	* cov: covariance matrix
	"""
	n = exp_returns.shape[0]
	init_guess = np.repeat(1/n, n)
	bounds = ((0.0, 1.0),)*n
	return_is_target = {
		'type': 'eq',
		'args': (exp_returns,), 
		'fun': lambda weights, exp_returns: target_return - port_return(weights, exp_returns)
	}
	weights_sum = {
		'type': 'eq',
		'fun': lambda weights: np.sum(weights)-1
	}
    
	results = minimize(port_vol, init_guess,
						args=(cov,), method="SLSQP",
						options={'disp': False},
						constraints=(return_is_target, weights_sum),
						bounds=bounds
						)
	return results.x

def optimal_weights(n_points, exp_returns, cov):
	"""
	List of weights to run the optimizer on to minimize the vol
	* n_points: int
	* exp_returns: array of expected returns
	* cov: covariance matrix
	"""
	target_ret = np.linspace(exp_returns.min(), exp_returns.max(), n_points)
	weights = [min_vol(target_returns, exp_returns, cov) for target_returns in target_ret]
	return weights

def max_sharpe(riskfree, exp_returns, cov):
	"""
	RiskFree rate + ER + COV --> W
	* riskfree: float
	* exp_returns: array of expected returns
	* cov: covariance matrix
	"""
	n = exp_returns.shape[0]
	init_guess = np.repeat(1/n, n)
	bounds = ((0.0, 1.0),)*n

	weights_sum = {
		'type': 'eq',
		'fun': lambda weights: np.sum(weights)-1
	}
    
	def neg_sharpe(weights, riskfree, exp_returns, cov):
		"""
		Returns the negative of Sharpe Ratio
		"""
		returns = port_return(weights, exp_returns)
		vol = port_vol(weights, cov)
		return -((returns - riskfree)/vol)
        
	results = minimize(neg_sharpe, init_guess,
						args=(riskfree, exp_returns, cov,), method="SLSQP",
						options={'disp': False},
						constraints=(weights_sum),
						bounds=bounds
						)
	return results.x

def gmv(cov):
	"""
	Returns the weights of Global Minimum Vol Portfolio
	"""
	n_assets = cov.shape[0]
	weights = max_sharpe(0, np.repeat(1, n_assets), cov)
	return weights

def plot_ef(n_points, exp_returns, cov, show_cml=False, style=".-", riskfree=0, show_ew=False, show_gmv=False):
	"""
	Plots the N-asset efficient frontier
	"""
	weights = optimal_weights(n_points, exp_returns, cov)
	returns = [port_return(w, exp_returns) for w in weights]
	vols = [port_vol(w, cov) for w in weights]
	eff_frontier = pd.DataFrame({
					"Returns":returns,
					"Vols":vols
		})
	ax = eff_frontier.plot.line(x="Vols", y="Returns", style=style)
	
	if show_ew:
		n_assets = exp_returns.shape[0]
		w_ew = np.repeat(1/n_assets, n_assets)
		returns_ew = port_return(w_ew, exp_returns)
		vol_ew = port_vol(w_ew, cov)
		
		# display EqWgt
		ax.plot([vol_ew], [returns_ew], color="goldenrod", marker="o", markersize=5)
		
	if show_gmv:
		w_gmv = gmv(cov)
		returns_gmv = port_return(w_gmv, exp_returns)
		vol_gmv = port_vol(w_gmv, cov)
		
		# display GMV
		ax.plot([vol_gmv], [returns_gmv], color="midnightblue", marker="o", markersize=5)
		
	if show_cml:
		ax.set_xlim(left = 0)
		w_maxSharpe = max_sharpe(riskfree, exp_returns, cov)
		r_maxSharpe = port_return(w_maxSharpe, exp_returns)
		vol_maxSharpe = port_vol(w_maxSharpe, cov)

		# Add CML
		cml_x = [0, vol_maxSharpe]
		cml_y = [riskfree, r_maxSharpe]
		ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=5, linewidth=2)
		
		return ax

# Summary Stats
def summary_stats(returns, riskfree=0.03, periods=252, actual_days=False):
	"""
	Return a DataFrame that contain aggregated summary stats for the returns in the columns of returns.
	"""
	annreturn = returns.aggregate(ann_return, periods=periods, actual_days=actual_days)
	annvol = returns.aggregate(ann_vol, periods=periods)
	annSharpe = returns.aggregate(sharpe_ratio, riskfree=riskfree, periods=periods)
	dd = returns.aggregate(lambda returns: drawdown(returns).Drawdown.min())
	skew = returns.aggregate(skewness)
	kurt = returns.aggregate(kurtosis)
	cf_var5 = returns.aggregate(var_gaussian, modified=True)
	hist_cvar5 = returns.aggregate(cvar_historic)
	
	return pd.DataFrame({
		"Annualized Return" : annreturn,
		"Annualized Vol" : annvol,
		"Skewness" : skew,
		"Kurtosis" : kurt,
		"Cornish-Fisher VaR (5%)" : cf_var5,
		"Historic CVaR (5%)" : hist_cvar5,
		"Sharpe Ratio" : annSharpe,
		"Max Drawdown" : dd})

# Análisis Técnico
def sma(prices, periods=20, multiindex=False):
	"""
	Calcular SMA dado el número de periodos.
	* prices: Series or DataFrame
	* sma_period: int
	"""
	if isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series):
		sma_df = prices.rolling(periods).mean().dropna()
		if multiindex:
			sma_df.columns = pd.MultiIndex.from_product([["SMA {}".format(periods)], sma_df.columns])
		return sma_df
	else:
		raise TypeError("Prices must be Series or DataFrame")

def ema(prices, periods=20, multiindex=False):
	"""
	Calcular EMA dado el número de periodos.
	* prices: Series or DataFrame
	* period: int
	"""
	if isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series):
		ema = prices.copy()
		ema.iloc[:periods] = sma(prices, periods).iloc[0:periods]
		ema_df = ema.ewm(span=periods, adjust=False).mean().dropna()
		if multiindex:
			ema_df.columns = pd.MultiIndex.from_product([["EMA {}".format(periods)], ema_df.columns])
		return ema_df
	else:
		raise TypeError("Prices must be Series or DataFrame")

def roc(prices, periods=14, multiindex=False):
	"""
	Calcular ROC dado el número de periodos.
	* prices: Series or DataFrame
	* periods: int
	"""
	if isinstance (prices, pd.DataFrame) or isinstance(prices, pd.Series):
		roc_df = (prices / prices.shift(periods) - 1).dropna()
		if multiindex:
			roc_df.columns = pd.MultiIndex.from_product([["ROC {}".format(periods)], roc_df.columns])
		return roc_df
	else:
		raise TypeError("Prices must be Series or DataFrame")

def macd(prices, emaLong=26, emaShort=12, signal=9, multiindex=False):
	"""
	Calcular MACD
	* prices: Series or DataFrame
	* ema: int periodos de las EMA
	* signal: int periodos de la señal del MACD
	"""
	if isinstance (prices, pd.DataFrame) or isinstance(prices, pd.Series):
		ema_long = ema(prices, emaLong)
		ema_short = ema(prices, emaShort)
		macd_line = (ema_long - ema_short).dropna()
		signal_line = ema(macd_line, signal)
		if multiindex:
			return pd.concat([macd_line, signal_line], axis=1, keys=["MACD {}-{}".format(emaShort, emaLong),"Signal {}".format(signal)]).dropna()
		else:
			return pd.concat([macd_line, signal_line], axis=1, keys=["MACD","Signal"]).dropna()
	else:
		raise TypeError("Prices must be Series or DataFrame")