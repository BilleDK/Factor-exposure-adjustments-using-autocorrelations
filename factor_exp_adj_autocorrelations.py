import numpy as np
import pandas as pd
import pandas_datareader as pdr
import pandas_datareader.data as web
import statsmodels.formula.api as sm
import statsmodels.stats.api as sms
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import math
import bs4 as bs
import requests
import yfinance as yf
import datetime


ff_5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start='1-1-1926')[0]
ff_mom = pdr.get_data_famafrench('F-F_Momentum_Factor', start='1-1-1926')[0]
ff_data = pd.merge(ff_5, ff_mom, how = 'left', left_index=True, right_index=True)
ff_data = ff_data.rename(columns={'Mom   ': 'WML'}) # As FF extracted data contains "Mom" col name with 3 spaces at end

SP500 = web.get_data_yahoo('^GSPC', '01/1985', interval = 'm')['Adj Close']
SP500.index = SP500.index.to_period('M')
SP500 = pd.merge(SP500, ff_data, how = 'left', left_index=True, right_index=True)
SP500 = SP500[['Adj Close', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML']]
SP500.rename(columns = {'Adj Close':'Price', 'Mkt-RF':'Mkt_RF'}, inplace = True)
SP500['SPreturn'] = SP500['Price'].pct_change() * 100
SP500 = SP500.dropna() # drop NA's
SP500_factors = sm.ols(formula = "SPreturn ~ Mkt_RF + SMB + HML + RMW + CMA + WML", data = SP500).fit()
years = len(SP500) / 12
SP_ann_ret = (1 + SP500['SPreturn'] / 100).prod() ** (1/years) - 1
SP_ann_volatility = (SP500['SPreturn'] / 100).std() * (12**0.5)
SP_SR = SP_ann_ret / SP_ann_volatility

# sets all factor exposures to SP500 factor exposures initially
# beta coefficients in the sm.ols function can be accessed through params as a df with 1 row
ff_data['Mkt_exp'] = SP500_factors.params['Mkt_RF']
ff_data['SMB_exp'] = SP500_factors.params['SMB']
ff_data['HML_exp'] = SP500_factors.params['HML']
ff_data['RMW_exp'] = SP500_factors.params['RMW']
ff_data['CMA_exp'] = SP500_factors.params['CMA']
ff_data['WML_exp'] = SP500_factors.params['WML']

# Adding rolling autocorrelations with a rolliwng window and lag variable
RW = 240
lag = 1

ff_data['Mkt_rACF'] = ff_data['Mkt-RF'].rolling(RW).apply(lambda x: pd.Series(x).autocorr(lag))
ff_data['SMB_rACF'] = ff_data['SMB'].rolling(RW).apply(lambda x: pd.Series(x).autocorr(lag))
ff_data['HML_rACF'] = ff_data['HML'].rolling(RW).apply(lambda x: pd.Series(x).autocorr(lag))
ff_data['RMW_rACF'] = ff_data['RMW'].rolling(RW).apply(lambda x: pd.Series(x).autocorr(lag))
ff_data['CMA_rACF'] = ff_data['CMA'].rolling(RW).apply(lambda x: pd.Series(x).autocorr(lag))
ff_data['WML_rACF'] = ff_data['WML'].rolling(RW).apply(lambda x: pd.Series(x).autocorr(lag))

# factor exposure adjustment formula
ff_data['Mkt_exp_formula'] = (ff_data['Mkt_rACF'].shift() * ff_data['Mkt-RF'].shift()) / 10
ff_data['SMB_exp_formula'] = (ff_data['SMB_rACF'].shift() * ff_data['SMB'].shift()) / 10
ff_data['HML_exp_formula'] = (ff_data['HML_rACF'].shift() * ff_data['HML'].shift()) / 10
ff_data['RMW_exp_formula'] = (ff_data['RMW_rACF'].shift() * ff_data['RMW'].shift()) / 10
ff_data['CMA_exp_formula'] = (ff_data['CMA_rACF'].shift() * ff_data['CMA'].shift()) / 10
ff_data['WML_exp_formula'] = (ff_data['WML_rACF'].shift() * ff_data['WML'].shift()) / 10

# Slicing the ff dataframe to make time horizon comparable with length of S&P500 data
ff_data = ff_data['1985-02':]

# adjusts the factor exposures based on factor return and rolling factor autocorrelation
# np.clip inputs an array-like and sets upper and lower boundaries
# boundaries set at -0,5 and +0,5 due to empirical limitations for factors, boundary set at 0,7 to 1,3 for market beta
min_b = -0.5
max_b = 0.5

for i in range(1, len(ff_data)):
    ff_data['Mkt_exp'].values[i] = np.clip(ff_data['Mkt_exp'].values[i-1] + ff_data['Mkt_exp_formula'].values[i], 0.7, 1.3)

for i in range(1, len(ff_data)):
    ff_data['SMB_exp'].values[i] = np.clip(ff_data['SMB_exp'].values[i-1] + ff_data['SMB_exp_formula'].values[i], min_b, max_b)

for i in range(1, len(ff_data)):
    ff_data['SMB_exp'].values[i] = np.clip(ff_data['SMB_exp'].values[i-1] + ff_data['SMB_exp_formula'].values[i], min_b, max_b)

for i in range(1, len(ff_data)):
    ff_data['HML_exp'].values[i] = np.clip(ff_data['HML_exp'].values[i-1] + ff_data['HML_exp_formula'].values[i], min_b, max_b)

for i in range(1, len(ff_data)):
    ff_data['RMW_exp'].values[i] = np.clip(ff_data['RMW_exp'].values[i-1] + ff_data['RMW_exp_formula'].values[i], min_b, max_b)
    
for i in range(1, len(ff_data)):
    ff_data['CMA_exp'].values[i] = np.clip(ff_data['CMA_exp'].values[i-1] + ff_data['CMA_exp_formula'].values[i], min_b, max_b)

for i in range(1, len(ff_data)):
    ff_data['WML_exp'].values[i] = np.clip(ff_data['WML_exp'].values[i-1] + ff_data['WML_exp_formula'].values[i], min_b, max_b)

# Setting other benchmarks using static factor exposures
# factor boundaries in 0.10 intervals: Mkt-RF = (0.0 - 1.0). All other factor (-0.5 - 0.5)
return_list = [] #creates empty list
return_df = pd.DataFrame() #creates empty df
# using np arange as basic python range only takes integers
Mkt_exp = np.arange(0.0, 1.1, 0.1)
factor_exp = np.arange(-0.5, 0.6, 0.1)
# create DF with benchmark returns
for i in Mkt_exp:
    for j in factor_exp:
        r_series = ff_data['Mkt-RF'] * i + (ff_data['SMB'] + ff_data['HML'] + ff_data['RMW'] + ff_data['CMA'] + ff_data['WML']) * j
        r_float = (1 + (r_series / 100)).prod()
        return_list.append(r_float) # adds each result to the list
    return_series = pd.Series(return_list, name = i) #transforms the list into a series to be used in .concat
    return_df = pd.concat([return_df, return_series], axis = 1) #concat is faster than .insert, inserts the result_series as a new column
    # empties the list so it is ready for next inner loop
    return_list.clear()
BM_returns = return_df ** (1/years) - 1
BM_returns.set_index(factor_exp, inplace = True)

vol_list = [] #creates empty list
BM_vol = pd.DataFrame() #creates empty df

# create DF with benchmark volatility
for i in Mkt_exp:
    for j in factor_exp:
        r_series = ff_data['Mkt-RF'] * i + (ff_data['SMB'] + ff_data['HML'] + ff_data['RMW'] + ff_data['CMA'] + ff_data['WML']) * j
        ann_vol = (r_series / 100).std() * (12**0.5)
        vol_list.append(ann_vol) # adds each result to the list
    vol_series = pd.Series(vol_list, name = i) #transforms the list into a series to be used in .concat
    BM_vol = pd.concat([BM_vol, vol_series], axis = 1) #concat is faster than .insert, inserts the result_series as a new column
    # empties the list so it is ready for next inner loop
    vol_list.clear()
BM_vol.set_index(factor_exp, inplace = True)

BM_SR = BM_returns / BM_vol

# Calculating portfolio return, volatility and Sharpe Ratio using the factor exposures based on autocorrelations and past factor returns
ff_data['return'] = ff_data['Mkt-RF'] * ff_data['Mkt_exp'] + ff_data.SMB * ff_data.SMB_exp + ff_data.HML * ff_data.HML_exp + ff_data.RMW * ff_data.RMW_exp + ff_data.CMA * ff_data.CMA_exp + ff_data.WML * ff_data.WML_exp
ff_data['return_factor'] = 1 + (ff_data['return'] / 100)
ann_ret = ff_data['return_factor'].prod() ** (1/years) - 1
ann_volatility = (ff_data['return'] / 100).std() * (12**0.5)
Portf_SR = ann_ret / ann_volatility

# Plotting factor exposures and returns to see how well the model adjust exposure to capture/avoid positive/negative returns
fig, axes = plt.subplots(3, 2, figsize=(24,12)) #creates 1 figure with 6 axes, in a 3x2 matrix, each axes can be chosen through indexing
(1 + ff_data['Mkt-RF'] / 100).cumprod().plot(ax=axes[0, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
ff_data['Mkt_exp'].plot.line(ax=axes[0, 0], title = "Market / Beta", legend=True, label="Factor exposure")
(1 + ff_data['SMB'] / 100).cumprod().plot(ax=axes[0, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
ff_data['SMB_exp'].plot.line(ax=axes[0, 1], title = "Size / SMB", legend=True, label="Factor exposure")
(1 + ff_data['HML'] / 100).cumprod().plot(ax=axes[1, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
ff_data['HML_exp'].plot.line(ax=axes[1, 0], title = "Value / HML", legend=True, label="Factor exposure")
(1 + ff_data['RMW'] / 100).cumprod().plot(ax=axes[1, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
ff_data['RMW_exp'].plot.line(ax=axes[1, 1], title = "Profitability / RMW", legend=True, label="Factor exposure")
(1 + ff_data['CMA'] / 100).cumprod().plot(ax=axes[2, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
ff_data['CMA_exp'].plot.line(ax=axes[2, 0], title = "Investments / CMA", legend=True, label="Factor exposure")
(1 + ff_data['WML'] / 100).cumprod().plot(ax=axes[2, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
ff_data['WML_exp'].plot.line(ax=axes[2, 1], title = "Momentum / WML", legend=True, label="Factor exposure")
fig.suptitle("Factor exposures & cumulative factor return")   
fig.tight_layout() # adds spacing between suptitle and graphs

# Plotting factor and portfolio returns to see how well the model adjust exposure to capture/avoid positive/negative returns
fig, axes = plt.subplots(3, 2, figsize=(24,12)) #creates 1 figure with 6 axes, in a 3x2 matrix, each axes can be chosen through indexing
(1 + ff_data['Mkt-RF'] / 100).cumprod().plot(ax=axes[0, 0], colormap='PRGn', legend=True, label="Cum. factor return")
(1 + ((ff_data['Mkt-RF'] / 100) * ff_data['Mkt_exp'])).cumprod().plot.line(ax=axes[0, 0], title = "Market / Beta", legend=True, label="Cum. Portfolio factor return")
(1 + ff_data['SMB'] / 100).cumprod().plot(ax=axes[0, 1], colormap='PRGn', legend=True, label="Cum. factor return")
(1 + ((ff_data['SMB'] / 100) * ff_data['SMB_exp'])).cumprod().plot.line(ax=axes[0, 1], title = "Size / SMB", legend=True, label="Cum. Portfolio factor return")
(1 + ff_data['HML'] / 100).cumprod().plot(ax=axes[1, 0], colormap='PRGn', legend=True, label="Cum. factor return")
(1 + ((ff_data['HML'] / 100) * ff_data['HML_exp'])).cumprod().plot.line(ax=axes[1, 0], title = "Value / HML", legend=True, label="Cum. Portfolio factor return")
(1 + ff_data['RMW'] / 100).cumprod().plot(ax=axes[1, 1], colormap='PRGn', legend=True, label="Cum. factor return")
(1 + ((ff_data['RMW'] / 100) * ff_data['RMW_exp'])).cumprod().plot.line(ax=axes[1, 1], title = "Profitability / RMW", legend=True, label="Cum. Portfolio factor return")
(1 + ff_data['CMA'] / 100).cumprod().plot(ax=axes[2, 0], colormap='PRGn', legend=True, label="Cum. factor return")
(1 + ((ff_data['CMA'] / 100) * ff_data['CMA_exp'])).cumprod().plot.line(ax=axes[2, 0], title = "Investments / CMA", legend=True, label="Cum. Portfolio factor return")
(1 + ff_data['WML'] / 100).cumprod().plot(ax=axes[2, 1], colormap='PRGn', legend=True, label="Cum. factor return")
(1 + ((ff_data['WML'] / 100) * ff_data['WML_exp'])).cumprod().plot.line(ax=axes[2, 1], title = "Momentum / WML", legend=True, label="Cum. Portfolio factor return")
fig.suptitle("Factor & portfolio factor cumulative returns")   
fig.tight_layout() # adds spacing between suptitle and graphs

# Plotting cumulative returns for SP500 and the portfolio
plot_frame = pd.DataFrame()
plot_frame['SP500'] = (1 + SP500['SPreturn'] / 100).cumprod()
plot_frame['Portfolio'] = ff_data['return_factor'].cumprod()
plot_frame.plot()

# performance after trading costs - assuming 5bps each month as it is not the entire portfolio that is turned around each month but only a fraction
ff_data['Trade_cost'] = 0.0005
ff_data['return_AC'] = 1 + ((ff_data['return'] / 100) - ff_data['Trade_cost'])
ann_ret_AC = ff_data['return_AC'].prod() ** (1/years) - 1
ann_volatility_AC = ((ff_data['return'] / 100) - ff_data['Trade_cost']).std() * (12**0.5)
Portf_AC_SR = ann_ret_AC / ann_volatility_AC