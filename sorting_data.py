# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 03:01:02 2019

@author: acer
"""

#%%
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from datetime import datetime
#%%
df = pd.read_csv('F:\sales_history_dataset\sales.csv',encoding="utf-8")#读取csv格式文件
#%%
df['DATE'] = pd.to_datetime(df['transaction_created_utc'])#将df表中日期转为datetime格式
df = df[['DATE','transaction_total']]#df表中只留两列
#%%
df['DATE'] = [datetime.strftime(x,'%Y-%m-%d') for x in df['DATE']]#把形如‘2017-9-4 00:00:00’转化为‘2017-9-4 ’
#%%
df =df.pivot_table(index='DATE',aggfunc=np.sum)#按天数统计总合
#%%
import  matplotlib.pyplot as plt
#%%
plt.hist(df['transaction_total'], bins=200, normed=True)#做出直方图
df['transaction_total'].describe(include = 'all')
#%%
df.to_csv('sales_history_dataset.csv',index=True)
#%%
from sklearn.model_selection import train_test_split
x = df['transaction_total']
y = df.index
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)