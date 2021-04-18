import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web

start = datetime.datetime(2020, 4, 18)
end = datetime.datetime(2021, 4, 18)
dfcomp = web.DataReader(['AAPL', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
dfcomp.tail()
print(dfcomp.tail())

dfcomp.to_pickle("close_price_data.pkl")


plt.style.use('seaborn-dark')

dfcomp.plot(cmap= "viridis",figsize=(20,15))
plt.grid()
plt.show()