# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import pandas as pd
sale=pd.read_csv("F:\sales history dataset\sales.csv",encoding="utf-8")


import sqlite3
con = sqlite3.connect(':memory:')
sale.to_sql('sale',con)
newTable = pd.read_sql_query("select transaction_total, transaction_created_utc from sale", con)
newTable.head()


import  matplotlib.pyplot as plt
plt.hist(newTable['transaction_total'], bins=500, normed=True)
newTable['transaction_total'].describe(include = 'all')
#%%
from sklearn.model_selection import train_test_split
newTable.head()
x = newTable['transaction_total']
y = newTable['transaction_created_utc']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)#分为训练集和测试集

#%%
type(newTable.transaction_created_utc[0])
newTable['transaction_created_utc']= pd.to_datetime(newTable['transaction_created_utc'],format = '%d.%m.%y')#将数据类型转换为日期类型
type(newTable.transaction_created_utc[0])
print(newTable.head(2))
print(newTable.shape)
#%%
newTable = newTable.set_index('transaction_created_utc')# 将date设置为index
#%%
print(type(newTable))
print(newTable.index)
print(type(newTable.index))
#%%
s = pd.Series(newTable['transaction_total'],index = newTable.index)#构造Series类型数据
print(type(s))
s.head(2)
#%%
z=newTable.resample('M').sum().head()#按周统计数据
print(z)