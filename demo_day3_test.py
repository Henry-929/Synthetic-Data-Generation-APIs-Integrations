# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:41:07 2019

@author: acer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.api import tsa
#%%
def generate_data(start_date, end_date):
    df = pd.DataFrame([300 + i * 30 + randrange(50) for i in range(31)], columns=['income'],
                      index=pd.date_range(start_date, end_date, freq='D'))

    return df


data = generate_data('20170601', '20170701')
# 这里要将数据类型转换为‘float64’
data['income'] = data['income'].astype('float64')
#%%
print(data)
#%%
data.dtypes
print(data)

x = pd.date_range('20180831', '20191010')
plt.plot(x[:386], df2['transaction_total'])
# lenth = len()
plt.plot(pred)
plt.show()
print('end')