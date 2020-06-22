#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 22:26:15 2020

@author: mac
"""

#%%
df = pd.read_csv('trainDemo.csv',encoding="utf-8")
df.iloc[:,1].plot()
dataset_X=df.iloc[:,1].values.reshape(1,-1).T

print(dataset_X.shape)
#%%
from hmmlearn.hmm import GaussianHMM
model = GaussianHMM(n_components=8, covariance_type="diag", n_iter=1000)
model.fit(dataset_X)
#%%
hidden_states=model.predict(dataset_X)
#%%
for i in range(model.n_components): # 打印出每个隐含状态
    mean=model.means_[i][0]
    variance=np.diag(model.covars_[i])[0]
    print('Hidden state: {}, Mean={:.3f}, Variance={:.3f}'
          .format((i+1),mean,variance))
#%%
# 使用HMM模型生成数据
N=2348
samples,_=model.sample(N)
plt.plot(samples[:,0])

#%%
print(samples)
import numpy
numpy.savetxt("Hours_HMMpred.csv", samples, delimiter=',')
#%%
plt.plot(dataset_X[:N],c='red',label='train') # 将实际涨幅和预测的涨幅绘制到一幅图中方便比较
plt.plot(samples[:,0],c='blue',label='Predicted')
plt.legend()
#%%
for i in [8,12,16,18,20]:
    model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000)
    model.fit(dataset_X)
    samples,_=model.sample(400)
    plt.plot(samples[:,0])
    plt.title('hidden state N={}'.format(i))
    plt.show()

#%%
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from datetime import datetime
df = pd.read_csv('Hours_HMMpred.csv',encoding="utf-8")
#%%
df['transaction_created_utc'] = pd.to_datetime(df['transaction_created_utc'])#将df表中日期转为datetime格式
df = df[['transaction_created_utc','transaction_total']]
df.set_index("transaction_created_utc", inplace=True)
#%%
df = df.resample('30min').sum()
#%%
df=df.reset_index()
#%%
df.to_csv("sum_30min_HMMpred.csv")

