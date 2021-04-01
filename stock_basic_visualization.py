#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:02:28 2021
@author: ZepeiZhao
"""

import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl


### visulization includes
# 1. Moving average chart
# 2. Rate of return chart
# 3. Income distribution scatter chart
# 4. Correlation heat map
# 5. Fast scatter plot of stock risk and return

### get data using Pandas web data reader, which is used to communicate with latest financial data

start = datetime.datetime(2020, 7, 20)
end = datetime.datetime(2021, 1, 6)
df = web.DataReader("GOOG", 'yahoo', start, end)
df.tail()
#print(df.tail())

### 1. moving average: Smoothing the price data by constantly updating the average price helps to reduce the "noise" in the price list
### shows the up and down trend of stock prices
close_px = df['Adj Close']
mavg = close_px.rolling(window=10).mean()

###### draw the moving average chart


mpl.rc('figure', figsize=(8, 7))
mpl.__version__
style.use('seaborn-poster')

close_px.plot(label='GOOG')
mavg.plot(label='mavg')
plt.legend()
plt.show()

### 2. Rate of return chart
rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')
plt.show()

### 3. Income distribution scatter chart
dfcomp = web.DataReader(['AAPL', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
retscomp = dfcomp.pct_change()

corr = retscomp.corr()

plt.scatter(retscomp.GOOG, retscomp.AAPL)
plt.xlabel('Returns GOOG')
plt.ylabel('Returns AAPL')
plt.show()

### 4. Correlation heat map
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()

plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)
plt.show()

### 5. Fast scatter plot of stock risk and return
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')

plt.ylabel('Risk')

for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(label,xy = (x, y), xytext = (20, -20),textcoords = 'offset points', ha = 'right', va = 'bottom',
                 bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                 arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.show()
