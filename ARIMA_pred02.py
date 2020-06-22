#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:06:22 2020

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
import os
#%%
print (os.path.abspath('.'))
#%%
df = pd.read_csv('train.csv',encoding="utf-8")   
#%%
df.dtypes
#%%
df2 = df.set_index('transaction_created_utc')  
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
#模型训练
arima = ARIMA(df2, order=(2, 1, 2))
result = arima.fit(disp=False)
print(result.aic, result.bic, result.hqic)

plt.plot(data_diff)
plt.plot(result.fittedvalues, color='red')
plt.title('ARIMA RSS: %.4f' % sum(result.fittedvalues - data_diff['transaction_total']) ** 2)
plt.show()
#%%
pred = result.predict('20190531', '20190920',dynamic=True,typ='levels')
print(pred)
#%%
x = pd.date_range('20180831', '20190920')
#%%
plt.plot(x[:274], df2['transaction_total'])
plt.plot(pred)
plt.show()
#%%
dict_pred = {'transaction_created_utc':pred.index,'transaction_total':pred.values}
df_pred = pd.DataFrame(dict_pred)
#%%
df_pred.to_csv('df_pred.csv',index=False)
