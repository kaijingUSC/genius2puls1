import pandas as pd
import matplotlib.pyplot as plt

df_text = pd.read_pickle("/Users/pz/Desktop/560/560-proj/sentiments_microsoft_news.pkl")
#df_text.head()
plt.style.use('seaborn-dark')
df_text[["sentimental_analysis_average","sentimental_analysis_score"]].plot(cmap = "viridis",linestyle= '-', figsize = (15, 10))
plt.grid()
#plt.show()
# print(df_text)

df_values = pd.read_pickle("/Users/pz/Desktop/560/560-proj/close_price_data.pkl")
#df_values.head()


microsoft_df = df_values[["MSFT"]]
microsoft_df.plot(cmap = "jet",linestyle='-',figsize = (15,10),marker='.')
plt.grid()
#plt.show()
# print(microsoft_df)



X = df_text["sentimental_analysis_average"].copy()
#X = df_text["sentimental_analysis_average"].copy()
X = X.reset_index(drop= False)
X["time"] = pd.to_datetime(X["time"],errors = 'coerce', format = '%Y-%m-%dT%H:%M',infer_datetime_format = True, cache = True,utc=True)
X["time"] = pd.to_datetime(X["time"])
X.rename(columns={'sentimental_analysis_average':'sentimental_analysis_average1'}, inplace = True)
X = X.set_index(pd.DatetimeIndex(X["time"]))
X.set_index("time")
# print(X.shape)
# print(X)


df_text_gensim = pd.read_pickle("/Users/pz/Desktop/560/560-proj/sentiments_microsoft_news.pkl")
#df_text_gensim.head()
names = df_text_gensim.columns
df_gensim = df_text_gensim[names[5:]].copy()


df_gensim = df_gensim.set_index(X.index)
# print(df_gensim)

X = pd.concat([X,df_gensim],axis= 1)
#X = X.drop(["time"],axis= 1)
X = X.drop(["sentimental_analysis_average1"],axis= 1)
# X = X.resample('1d').first()
X = X.tz_convert(None)
y = microsoft_df.copy()

# print(X)
X.insert(X.shape[1], 't1', 0)
X['t1'] = X.index
y.insert(y.shape[1], 't1', 0)
y['t1'] = y.index

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

#result = pd.merge(X,y)
result = pd.merge(y, X, how='left', on=None, left_on='t1', right_on='t1',
      left_index=False, right_index=False, sort=True,
      suffixes=('_x', '_y'), copy=True, indicator=False)
result = result.fillna(0)
#print(result.sentimental_analysis_average)

result.set_index(["t1"],inplace=True)

# print(result.sentimental_analysis_average)
# print(result)
result.to_pickle("merge.pkl")