#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:13:52 2020

@author: mac
"""

import pandas as pd
from datetime import datetime
import os
import numpy as np

from HMM import GaussianHMM
from sklearn.preprocessing import scale

from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator,DayLocator
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
date = date[1:]
transaction = transaction[1:]
numbers = numbers[1:]

# scale归一化处理：均值为0和方差为1
# 将价格和交易数组成输入数据
A = np.column_stack([scale(diff),scale(numbers)])
#%%
# 训练高斯HMM模型
model = GaussianHMM(8,2,20)
model.train(A)

#%%
# 预测隐状态
hidden_states = model.decode(A)

# 打印参数
print "Transition matrix: ", model.transmat_prob
print("Means and vars of each hidden state")
for i in range(model.n_state):
    print("{0}th hidden state".format(i))
    print("mean = ", model.emit_means[i])
    print("var = ", model.emit_covars[i])
    print()

# 画图描述
fig, axs = plt.subplots(model.n_state, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_state))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(date[mask], transaction[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_locator(DayLocator())

    ax.grid(True)

plt.show()


#%%
#=================================================================================
#hidden_states = model.predict(A)
#print hidden_states
#=================================================================================
#plt.figure(figsize=(25,18)) 
#for i in range(model.n_components):
#    pos = (hidden_states==i)
#    plt.plot_date(date[pos],transaction[pos],'o',label='hidden state %d'%i,lw=2)
#    plt.legend(loc="left")
#=================================================================================  

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
df.to_csv("mean_30min_HMMpred.csv")



















