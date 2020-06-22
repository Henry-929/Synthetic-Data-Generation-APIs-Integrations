# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:56:02 2019

@author: acer
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
df = pd.read_csv('F:\sales_history_dataset\sales_history_dataset.csv',encoding="utf-8")
#%%
df.dtypes

#%%
df2 = df.set_index('DATE')  
#%%
sns.jointplot(x='DATE',y='transaction_total',data=df2)
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
arima = ARIMA(df2, order=(2, 1, 4))
result = arima.fit(disp=False)
print(result.aic, result.bic, result.hqic)

plt.plot(data_diff)
plt.plot(result.fittedvalues, color='red')
plt.title('ARIMA RSS: %.4f' % sum(result.fittedvalues - data_diff['transaction_total']) ** 2)
plt.show()
#%%
pred = result.predict('20190921', '20200301',dynamic=True,typ='levels')
print(pred)
#%%
x = pd.date_range('20180831', '20200301')
#%%
plt.plot(x[:386], df2['transaction_total'])
plt.plot(pred)
plt.show()
#%%
compare_data=pd.concat([df2,pred],axis=0) #合并列 
print(compare_data)
#%%
import statsmodels.api as sm
#%%
train_results = sm.tsa.arma_order_select_ic(df2, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
 
print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

