#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:57:03 2020

@author: mac
"""

import pandas as pd
from datetime import datetime
import os
import numpy as np

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import scale

from matplotlib import pyplot as plt
#%%
print (os.path.abspath('.'))
df = pd.read_csv('sales1.csv',encoding="utf-8")
#%%
df['transaction_created_utc'] = pd.to_datetime(df['transaction_created_utc'])#将df表中日期转为datetime格式
df = df[['transaction_created_utc','transaction_total']]
df['transaction_created_utc'] = [datetime.strftime(x,'%Y/%m/%d') for x in df['transaction_created_utc']]#把形如‘2017-9-4 00:00:00’转化为‘2017-9-4 ’
#%%
grouped = df.groupby(['transaction_created_utc']).count()
grouped = grouped.rename(columns = {'transaction_total':'Number of transactions'})
#%%
df = pd.read_csv('sales_history_dataset.csv',encoding='utf-8')
df['transaction_created_utc'] = pd.to_datetime(df['transaction_created_utc'])#将df表中日期转为datetime格式
df = df[['transaction_created_utc','transaction_total']]
df['transaction_created_utc'] = [datetime.strftime(x,'%Y/%m/%d') for x in df['transaction_created_utc']]#把形如‘2017-9-4 00:00:00’转化为‘2017-9-4 ’
df = df.set_index(['transaction_created_utc'])
#%%
result = pd.concat([df, grouped], axis=1)#生成带有交易数的数据
result = result.reset_index(['transaction_created_utc'])#将transaction_created_utc还原为列
#%%
#将result中的一列数据存入array中。
result.columns = ['transaction_created_utc','transaction_total','Number of transactions']
#将transaction_created_utc存入array中
date = result[['transaction_created_utc']]
date = np.array(date['transaction_created_utc'])
print date
#将transaction_total存入array中
transaction = result[['transaction_total']]
transaction = np.array(transaction['transaction_total'])
print transaction
#将Number of transactions存入array中
numbers = result[['Number of transactions']]
numbers = np.array(numbers['Number of transactions'])
print numbers
#%%
#diff: out[n] = a[n+1] - a[n] 得到价格变化
diff = np.diff(transaction)
print(diff)
#date = date[1:]
#transaction = transaction[1:]
#numbers = numbers[1:]
#%%
feature1 = 100*np.diff(transaction)/transaction[:-1]
print(transaction[:10])
print(feature1[:10]) 

feature2 = numbers[1:]
dataset = np.c_[feature1,feature2]
print(dataset[:5])
#%%
# scale归一化处理：均值为0和方差为1
# 将价格和交易数组成输入数据
# A = np.column_stack([scale(diff),scale(numbers)])
#%%
feature1 = transaction[:-1]
dataset = np.c_[feature1,feature2]
#%%
# 训练高斯HMM模型
model = GaussianHMM(n_components=8,covariance_type="diag",n_iter=1000)
model.fit(transaction)


#%%
#打印出每个隐含状态
for i in range (model.n_components):
    mean = model.means_[i][0]
    variance = np.diag(model.covars_[i])[0]
    print('Hidden state: {}, Mean={:.3f}, Variance={:.3f}'.format((i+1),mean,variance))

#%%
#使用HMM模型生成数据
N = 385
samples,_=model.sample(N)
plt.plot(samples[:,0])
#%%
plt.plot(np.arange(N), samples[:,0])
plt.title('Number of components = ' + str(N))

plt.show()

#%%
plt.plot(feature1[:N],c='red',label='Rise') # 将实际涨幅和预测的涨幅绘制到一幅图中方便比较
plt.plot(samples[:,0],c='blue',label='Predicted')
plt.legend()

#%%
plt.plot(feature2[:N],c='red',label='numbers')
plt.plot(samples[:,1],c='blue',label='Predicted')
plt.legend()
#%%
#模型的提升，修改n_components
for i in [8,12,16,18,20]:
    model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000)
    model.fit(dataset)
    samples,_=model.sample(500)
    plt.plot(samples[:,0])
    plt.title('hidden state N={}'.format(i))
    plt.show()










