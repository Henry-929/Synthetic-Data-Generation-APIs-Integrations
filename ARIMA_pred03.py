#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:24:34 2020

@author: mac
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
#%%
print (os.path.abspath('.'))

#%%
df_1 = pd.read_csv('train.csv',encoding="utf-8")  
df_2 = pd.read_csv('test.csv',encoding="utf-8")
df_3 = pd.read_csv('df_pred.csv',encoding="utf-8")
#%%
id_1 = df_1.set_index('transaction_created_utc')  
id_2 = df_2.set_index('transaction_created_utc')
id_3 = df_3.set_index('transaction_created_utc')
#%%
x_1 = pd.date_range('20180831', '20190531')
x_2 = pd.date_range('20190601', '20190920')
##x_1 = id_1.index
##x_2 = id_2.index
##x_3 = id_3.index
#%%
y_1 = id_1['transaction_total']
y_2 = id_2['transaction_total']
y_3 = id_3['transaction_total']
#%%
plt.plot(x_1,y_1,label="Train",color='red')
plt.plot(x_2,y_2,label="Test",color='blue',linestyle="--")
plt.plot(x_2,y_3,label="pred",color='green')
plt.legend()

plt.xlabel('transaction_created_utc')
plt.ylabel('transaction_total')
plt.show()
#%%
from math import sqrt
from sklearn.metrics import mean_squared_error
 
rms = sqrt(mean_squared_error(id_2,id_3))
print(rms)

