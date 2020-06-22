#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:29:49 2020

@author: mac
"""
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randrange
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.api import tsa
#%%
import pandas as pd
import os
#%%
print (os.path.abspath('.'))
#%%
df = pd.read_csv('train.csv',encoding="utf-8") 
ef = pd.read_csv('test.csv',encoding="utf-8")  
#%%
df.dtypes
ef.dtypes
#%%
df2 = df.set_index('transaction_created_utc')  
ef2 = ef.set_index('transaction_created_utc')
#%%
sns.jointplot(x='transaction_created_utc',y='transaction_total',data=df2)
#%%
df2.plot()
plt.show()# 绘制时序图
#%%
data_diff = df2.diff()#差分–转换为平稳序列,默认差分阶数为1
#%%
data_diff = df2.diff(2)#差分阶数为2
#%%
data_diff = data_diff.dropna()#差分后需要排空
data_diff.plot()
plt.show()
#%%
plot_acf(data_diff).show()
plot_pacf(data_diff).show()

#%%
from pyramid.arima import auto_arima
#%%
arima = auto_arima(df2, trace=True, error_action='ignore', suppress_warnings=True)
arima.fit(df2)

# =============================================================================
# arima = auto_arima(df2, start_p=1, start_q=1,
#                            max_p=3, max_q=3, m=12,
#                            start_P=0, seasonal=True,
#                            d=1, D=1, trace=True,
#                            error_action='ignore',  
#                            suppress_warnings=True, 
#                            stepwise=True)
# arima.fit(df2)
# =============================================================================

#%%
df.index = pd.to_datetime(df.index)
ef.index = pd.to_datetime(ef.index)
#%%
train = df2.loc['2018-08-31':'2019-05-31']
test = ef2.loc['2019-06-01':'2019-09-20']
#%%
arima.fit(train)
#%%
# =============================================================================
# forecast = arima.predict(n_periods=len(test))
# print(forecast)
# =============================================================================
#%%
forecast = arima.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
#%%
# =============================================================================
# forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
# pd.concat([test,forecast],axis=1).plot()
# =============================================================================

#%%
plt.plot(train, label='Train')
plt.plot(test, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()

#pd.concat([train,test,forecast],axis=1).plot()

# =============================================================================
# #%%
# forecast = pd.DataFrame(forecast,index = ef.index,columns=['Prediction'])
# #%%
# plt.plot(df2, label='Train')
# plt.plot(ef2, label='Test')
# plt.plot(forecast, label='Prediction')
# plt.show()
# =============================================================================
#%%
#模型训练
# =============================================================================
# model = ARIMA(df2, order=(5,1,1))
# result = arima.fit(disp=False)
# print(result.aic, result.bic, result.hqic)
# 
# plt.plot(data_diff)
# plt.plot(result.fittedvalues, color='red')
# plt.title('ARIMA RSS: %.4f' % sum(result.fittedvalues - data_diff['transaction_total']) ** 2)
# plt.show()
# =============================================================================

#%%
from math import sqrt
from sklearn.metrics import mean_squared_error
 
rms = sqrt(mean_squared_error(test,forecast))
print(rms)





