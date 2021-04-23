import requests
import urllib
import sys
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from pandas_datareader.data import DataReader
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-3):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back:(i+look_back+3), 0])
    return np.array(X), np.array(Y)

def extract_link_of_news(k_word, n_of_page):
    news_df = pd.DataFrame()

    link = "https://financialpost.com/search/?search_text=" + k_word + "&search_text="+k_word+"&date_range=-365d&sort=score&from="+str(n_of_page*10)

    print(link)
    info = get_info(link)
    news_df["internals_text"] = info[0]
    news_df["internals_dates"] = info[1]
    news_df["internal_urls"] = info[2]
    news_df["principal_url"] = link
    news_df["n_of_page"] = n_of_page

    return news_df



def get_info(ur):
    # news_df = pd.DataFrame()
    info = BeautifulSoup(requests.get(ur, allow_redirects=True).content, 'html.parser').find_all("div", {
        "class": "article-card__details"})
    links = ["https://financialpost.com/"+a['href'] for each_link in info for a in each_link.find_all('a', {"class":"article-card__link"},href=True)]
    text = [p.contents[0].strip() for each_link in info for p in each_link.find_all('p', {"class":"article-card__excerpt"})]
    date = [span.contents[0].strip() for each_link in info for span in each_link.find_all('span', {"class":"article-card__time"})]
    new_date = []

    for each in date:
        if 'ago' in each:
            if 'hour' in each:
                true_date = datetime.now() - timedelta(hours=int(each[0]))
            else:
                true_date = datetime.now() - timedelta(days=int(each[0]))
            each = true_date.strftime("%B %d, %Y")
        new_date.append(each)

    return [text, new_date, links]




### sentiment analysis


def sentimental_analysis(y):
    return (analyser.polarity_scores(y)["compound"])


n_of_pages = 20
COMPANY = "facebook"
SYMBOL = "fb"
df = pd.concat([extract_link_of_news(COMPANY,i) for i in range(1,n_of_pages+1)],ignore_index = True)


analyser = SentimentIntensityAnalyzer()

text = df['internals_text']
df["time"] = pd.to_datetime(df["internals_dates"])
df = df.sort_values("time")
df = df.set_index("time")

df["sentimental_analysis_score"] = df["internals_text"].apply(sentimental_analysis)

print(df.head())



sentiment_df = df[['internals_dates', 'sentimental_analysis_score']].groupby('internals_dates').mean().reset_index()
sentiment_df["time"] = pd.to_datetime(sentiment_df["internals_dates"])
sentiment_df = sentiment_df.sort_values("time").reset_index(drop=True)




def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-2):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back:(i+look_back+3), 0])
    return np.array(X), np.array(Y)

look_back = 7


end = max(sentiment_df['time'])
start = min(sentiment_df['time']) #- timedelta(days=6)
stock_df = DataReader(SYMBOL, data_source='yahoo', start=start, end=end)
# stock_df["ra"] = stock_df.Close.rolling(window=5).mean()
# stock_df = stock_df[stock_df['ra'].notna()]

stock_df = stock_df.filter(['Close'])
stock_df["t1"] = pd.to_datetime(stock_df.index)

result = pd.merge(stock_df, sentiment_df, how='left', on=None, left_on='t1', right_on='time', 
                  left_index=False, right_index=False, sort=True, 
                  suffixes=('_x', '_y'), copy=True, indicator=False)
result = result.fillna(0)
result.set_index(["t1"],inplace=True)


# original time series (Y)
y = result.Close.values
y = y.astype('float32')
y = np.reshape(y, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(y)


# add rolling average
# y_r = result.ra.values
# y_r = y_r.astype('float32')
# y_r = np.reshape(y_r, (-1, 1))
# scaler = MinMaxScaler(feature_range=(0, 1))
# y_r = scaler.fit_transform(y_r)


# extra information: features of the sentiment analysis
X = result.sentimental_analysis_score.values
X = X.astype('float32')
X = np.reshape(X, (-1, 1))

# training and testing settings (size)
percent_of_training = 0.8
# train_size = len(y) - 3
# train_size = len(y)
train_size = int(len(y) * percent_of_training) 

test_size = len(y) - train_size - 2 
#
# train_y_r, test_y_r = y_r[0:train_size+2,:], y_r[train_size-look_back:,:]
train_y, test_y = y[0:train_size+2,:], y[train_size-look_back:,:]
train_x, test_x = X[0:train_size+2,:], X[train_size-look_back:,:]


# features of the original time serie (y)
X_train_features_1, y_train = create_dataset(train_y, look_back)
X_test_features_1, y_test = create_dataset(test_y, look_back)

# X_train_features_1r, y_train_r = create_dataset(train_y_r, look_back)
# X_test_features_1r, y_test_r = create_dataset(test_y_r, look_back)

# calculate extra features in (X)
X_train_features_2, auxiliar_1 = create_dataset(train_x, look_back)
X_test_features_2, auxiliar_2 = create_dataset(test_x, look_back)


# join the all the features in one
## reshape arrays
X_train_features_1 = np.reshape(X_train_features_1, (X_train_features_1.shape[0], 1, X_train_features_1.shape[1]))
X_test_features_1  = np.reshape(X_test_features_1, (X_test_features_1.shape[0], 1, X_test_features_1.shape[1]))

# X_train_features_1r = np.reshape(X_train_features_1r, (X_train_features_1r.shape[0], 1, X_train_features_1r.shape[1]))
# X_test_features_1r  = np.reshape(X_test_features_1r, (X_test_features_1r.shape[0], 1, X_test_features_1r.shape[1]))

X_train_features_2 = np.reshape(X_train_features_2, (X_train_features_2.shape[0], 1, X_train_features_2.shape[1]))
X_test_features_2  = np.reshape(X_test_features_2, (X_test_features_2.shape[0], 1, X_test_features_2.shape[1]))

## put all together
X_train_all_features = np.append(X_train_features_1, X_train_features_2,axis=1)
# X_train_all_features = np.append(X_train_features_1r, X_train_all_features,axis=1)

X_test_all_features = np.append(X_test_features_1, X_test_features_2,axis=1)
# X_test_all_features = np.append(X_test_features_1r, X_test_all_features,axis=1)


import time 

start = time.time()
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train_all_features.shape[1], X_train_all_features.shape[2])))
model.add(RepeatVector(30))
model.add(LSTM(16, activation='relu', return_sequences=(True)))
model.add(TimeDistributed(Dense(3)))

model.compile(loss='mean_squared_error', optimizer='adam')

# history = model.fit(X_train_all_features,y_train, epochs=100, batch_size=1, #validation_data=(X_test_all_features, y_test),
#                     callbacks=[EarlyStopping(monitor='loss', patience=10)], verbose=0, shuffle=True)

history = model.fit(X_train_all_features,y_train, epochs=100, batch_size=1, validation_data=(X_test_all_features, y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0, shuffle=True)

model.summary()
train_predict = model.predict(X_train_all_features)

train_predict = np.array([x[0] for x in train_predict])
train_predict = scaler.inverse_transform(train_predict)


# plt.figure(figsize=(8,4))
# plt.style.use('seaborn-dark')


# plt.plot(history.history['loss'], label='Train Loss',color="green")
# plt.plot(history.history['val_loss'], label='Test Loss',color = "yellow")
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.grid()

# plt.show();

# p = np.array([[[i[0] for i in y[-4:]], [i[0] for i in X[-4:]]]])
# test_predict = model.predict(p)
# test_predict  = scaler.inverse_transform(np.array([x[0] for x in test_predict]))[0] 

# time_y_train = pd.DataFrame(data = result['Close'], index = result.index)
# time_y_test  = pd.DataFrame(data = test_y[:], index = result[train_size-look_back:].index,columns= [""])

# time_y_train_prediction = pd.DataFrame(data = train_predict[:,0], index = time_y_train[look_back:-2].index,columns= [""])
# time_y_test_prediction  = pd.DataFrame(data = test_predict[:,0], index = time_y_test[look_back+2:].index,columns= [""])



test_predict  = model.predict(X_test_all_features)
test_predict  = np.array([x[0] for x in test_predict])
test_predict = scaler.inverse_transform(test_predict)


time_y_train = pd.DataFrame(data = result['Close'][look_back:look_back + len(y_train)], index = result.index[look_back:look_back + len(y_train)])
time_y_test  = pd.DataFrame(data = result['Close'][train_size-look_back:-2], index = result['Close'][train_size-look_back:-2].index)

time_y_train_prediction = pd.DataFrame(data = train_predict[:,0], index = result.index[look_back:look_back + len(y_train)],columns= [""])
time_y_test_prediction  = pd.DataFrame(data = test_predict[:,0], index = result.index[train_size:-2],columns= [""])


plt.style.use('seaborn-dark')
plt.figure(figsize=(15,10))
# plt_predict = pd.DataFrame(data=test_predict, index=[result.index[-1] + timedelta(days=1), result.index[-1] + timedelta(days=2), result.index[-1] + timedelta(days=3)])
# plt.plot(plt_predict, label="Predict", marker=".")
# plt.annotate("{:.2f}".format(plt_predict[0][0]), (plt_predict.index[0],plt_predict[0][0]), ha='right')
# plt.annotate("{:.2f}".format(plt_predict[0][1]), (plt_predict.index[1],plt_predict[0][1]), ha='center')
# plt.annotate("{:.2f}".format(plt_predict[0][2]), (plt_predict.index[2],plt_predict[0][2]), ha='left')

plt.plot(time_y_train,label = "training",color ="green",marker='.')
plt.plot(time_y_test,label = "test",marker='.')
plt.plot(time_y_train_prediction,color="red",label = "prediction")
plt.plot(time_y_test_prediction,color="red")
# plt.title("LSTM fit of Stock Market Prices Including Sentiment Signal",size = 20)
plt.tight_layout()
sns.despine(top=True)
plt.ylabel('price', size=15)
plt.xlabel('date', size=15)
plt.legend(fontsize=15)
plt.grid()

plt.show()


# print('Train Mean Absolute Error:', mean_absolute_error(np.reshape(y_train,(y_train.shape[0],1)), train_predict[:,0]))
# print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(np.reshape(y_train,(y_train.shape[0],1)), train_predict[:,0])))
# print('Test Mean Absolute Error:', mean_absolute_error(np.reshape(y_test,(y_test.shape[0],1)), test_predict[:,0]))
# print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(np.reshape(y_test,(y_test.shape[0],1)), test_predict[:,0])))


# test_predict  = model.predict(X_test_all_features)
# test_predict  = np.array([x[0] for x in test_predict])

# predictions = scaler.inverse_transform(test_predict)
predictions = test_predict
y_true = []
for i in range(len(predictions)):
    i_start = i + len(result.Close.values) - len(predictions) - 2
    i_end = i + len(result.Close.values) - len(predictions) + 1
    y_true.append(result.Close.values[i_start:i_end])
y_true = np.array(y_true)
# predictions = [x[0] for x in predictions]

y_true_inv = scaler.transform(y_true)
test_predict_inv = scaler.transform(test_predict)


print('Test Mean Squared Error:', mean_squared_error(y_true[:,0],test_predict[:,0]))

print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_true[:,0],predictions[:,0])))
print('Test MAPE:',np.mean(abs(predictions[:,0] - y_true[:,0]) / y_true[:,0]))

print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_true[:,1],predictions[:,1])))
print('Test MAPE:',np.mean(abs(predictions[:,1] - y_true[:,1]) / y_true[:,1]))

print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_true[:,2],predictions[:,2])))
print('Test MAPE:',np.mean(abs(predictions[:,2] - y_true[:,2]) / y_true[:,2]))

end = time.time()
print("runtime: ", end-start)