#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
import seaborn as sns
from pandas_datareader.data import DataReader
import urllib
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

pd.set_option('display.float_format', lambda x: '%.4f' % x)
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib as mpl
from matplotlib import style
import pandas_datareader.data as web
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

matplotlib.use('Agg')
# plt.style.use("fivethirtyeight")

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
            true_date = datetime.now() - timedelta(days=int(each[0]))
            each = true_date.strftime("%B %d, %Y")
        new_date.append(each)
    
    return [text, new_date, links]

def split_by_dot(x):
    return x.split(".")

def sentimental_analysis_by_phrase(y):
    analyser = SentimentIntensityAnalyzer()
    y = list(map(lambda x: analyser.polarity_scores(x)["compound"], y))
    y = np.array(y)
    y = y[y != 0]
    return (y)

def sentimental_analysis(y):
    analyser = SentimentIntensityAnalyzer()
    return (analyser.polarity_scores(y)["compound"])

def search_stock(key):

    news = []
    key = key.strip().replace(" ", "%20")
    start_page = 'https://investing.com/search/?q=' + key
    page = requests.get(start_page, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(page.text, 'html.parser')
    
    search_link = soup.find(class_='js-inner-all-results-quote-item')
    symbol = search_link.find("span", class_="second").contents[0]
    company = search_link.find("span", class_="third").contents[0]

    return symbol, company

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-2):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back:(i+look_back+3), 0])
    return np.array(X), np.array(Y)

def stock_predict(SYMBOL, COMPANY):
    n_of_pages = 5
    look_back = 4

    # COMPANY = "FACEBOOK"
    # SYMBOL = "FB"
    df = pd.concat([extract_link_of_news(COMPANY,i) for i in range(1,n_of_pages+1)],ignore_index = True)
    analyser = SentimentIntensityAnalyzer()

    text = df['internals_text']
    df["text_split"] = df["internals_text"].apply(split_by_dot)
    df["time"] = pd.to_datetime(df["internals_dates"])
    df = df.sort_values("time")
    df = df.set_index("time")

    text_2 = df["text_split"][0]
    sen_res = sentimental_analysis_by_phrase(text_2)
    df["sentimental_analysis_phrase"] = df["text_split"].apply(sentimental_analysis_by_phrase)
    df["sentimental_analysis_average"] = df["sentimental_analysis_phrase"].apply(np.mean)

    df["sentimental_analysis_score"] = df["internals_text"].apply(sentimental_analysis)

    sentiment_df = df[['internals_dates', 'sentimental_analysis_score']].groupby('internals_dates').mean().reset_index()
    sentiment_df["time"] = pd.to_datetime(sentiment_df["internals_dates"])
    sentiment_df = sentiment_df.sort_values("time").reset_index(drop=True)

    end = max(sentiment_df['time'])
    start = min(sentiment_df['time'])
    stock_df = DataReader(SYMBOL, data_source='yahoo', start=start, end=end)
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

    # extra information: features of the sentiment analysis
    X = result.sentimental_analysis_score.values
    X = X.astype('float32')
    X = np.reshape(X, (-1, 1))

    train_size = len(y) 
    test_size = len(y) - train_size - 2 

    train_y, test_y = y[0:train_size+2,:], y[train_size-look_back:,:]
    train_x, test_x = X[0:train_size+2,:], X[train_size-look_back:,:]

    X_train_features_1, y_train = create_dataset(train_y, look_back)
    X_train_features_2, auxiliar_1 = create_dataset(train_x, look_back)

    X_train_features_1 = np.reshape(X_train_features_1, (X_train_features_1.shape[0], 1, X_train_features_1.shape[1]))
    X_train_features_2 = np.reshape(X_train_features_2, (X_train_features_2.shape[0], 1, X_train_features_2.shape[1]))
    X_train_all_features = np.append(X_train_features_1,X_train_features_2,axis=1)

    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(X_train_all_features.shape[1], X_train_all_features.shape[2])))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train_all_features,y_train, epochs=10, batch_size=1, 
                        callbacks=[EarlyStopping(monitor='loss', patience=10)], verbose=0, shuffle=False)

    p = np.array([[[i[0] for i in y[-4:]], [i[0] for i in X[-4:]]]])
    test_predict = model.predict(p)
    test_predict  = scaler.inverse_transform(np.array([x[0] for x in test_predict]))[0] 

    img = io.BytesIO()

    plt.style.use('seaborn-dark')
    plt.figure(figsize=(16,8))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)

    plt_predict = pd.DataFrame(data=test_predict, index=[result.index[-1] + timedelta(days=1), result.index[-1] + timedelta(days=2), result.index[-1] + timedelta(days=3)])
    plt_historical = pd.DataFrame(data=result['Close'], index=result.index)
    plt.plot(plt_historical[-30:], label="History", marker=".")
    plt.plot(plt_predict, label="Predict", marker=".")
    plt.annotate("{:.2f}".format(plt_predict[0][0]), (plt_predict.index[0],plt_predict[0][0]), ha='right')
    plt.annotate("{:.2f}".format(plt_predict[0][1]), (plt_predict.index[1],plt_predict[0][1]), ha='center')
    plt.annotate("{:.2f}".format(plt_predict[0][2]), (plt_predict.index[2],plt_predict[0][2]), ha='left')
    plt.legend(['History', 'Predictions'], loc='lower right')
    plt.title("LSTM fit of Stock Market Prices Including Sentiment Signal",size = 20)
    plt.tight_layout()
    sns.despine(top=True)
    plt.grid()

    plt.savefig(img, format='png')
    plot_url = base64.b64encode(img.getbuffer()).decode("ascii")

    news_return = []
    a = df[["internals_text", "sentimental_analysis_score"]].drop_duplicates()
    for idx, row in a[-10:].iterrows():
        line = {"text": row["internals_text"], "date": datetime.date(idx), "score": row["sentimental_analysis_score"]}
        news_return.insert(0, line)

    return plot_url, news_return, list(test_predict)


def stock_predict_plt(key):
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)  
    df = DataReader(key, data_source='yahoo', start=start, end=end)
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .8 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(30, len(train_data)):
        x_train.append(train_data[i-30:i, 0])
        y_train.append(train_data[i, 0])
            
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(70, return_sequences=False, input_shape= (x_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=20)

    test_data = scaled_data[training_data_len - 30: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(30, len(test_data)):
        x_test.append(test_data[i-30:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = data[-500:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    img = io.BytesIO()
    plt.figure(figsize=(16,8))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig(img, format='png')
    plot_url = base64.b64encode(img.getbuffer()).decode("ascii")

    return plot_url

def get_stock(key):
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)  
    df = DataReader(key, data_source='yahoo', start=start, end=end)
    # close_px = df['Adj Close']
    return df

def moving_average(key):
    df = get_stock(key)
    close_px = df['Adj Close']
    mavg = close_px.rolling(window=10).mean()
    mpl.rc('figure', figsize=(8, 7))
    mpl.__version__
    style.use('seaborn-poster')

    img = io.BytesIO()
    plt.figure()
    close_px.plot(label=str(key))
    mavg.plot(label='mavg')
    plt.legend()
    plt.show()
    plt.savefig(img, format='png')
    plot_url = base64.b64encode(img.getbuffer()).decode("ascii")
    return plot_url

def Rate_of_Return(key):
    df = get_stock(key)
    close_px = df['Adj Close']
    rets = close_px / close_px.shift(1) - 1

    img = io.BytesIO()
    plt.figure()
    # close_px.plot(label=str(key))
    rets.plot(label='return')
    plt.show()
    plt.savefig(img, format='png')
    plot_url = base64.b64encode(img.getbuffer()).decode("ascii")
    return plot_url

def Correlation(key):
    df = get_stock(key)
    # close_px = df['Adj Close']
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)  

    ### 3. Income distribution scatter chart
    dfcomp = web.DataReader(['AAPL', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
    retscomp = dfcomp.pct_change()

    corr = retscomp.corr()

    img = io.BytesIO()

    ### 4. Correlation heat map
    plt.imshow(corr, cmap='hot', interpolation='none')
    plt.colorbar()

    plt.xticks(range(len(corr)), corr.columns)
    plt.yticks(range(len(corr)), corr.columns)
    plt.show()
    plt.savefig(img, format='png')
    plot_url = base64.b64encode(img.getbuffer()).decode("ascii")
    return plot_url

def Risk_and_Return(key):
    df = get_stock(key)
    # close_px = df['Adj Close']
    
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day) 
    dfcomp = web.DataReader(['AAPL', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
    retscomp = dfcomp.pct_change()

    img = io.BytesIO()
    plt.figure()
    plt.scatter(retscomp.mean(), retscomp.std())
    plt.xlabel('Expected returns')

    plt.ylabel('Risk')

    for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
        plt.annotate(label,xy = (x, y), xytext = (20, -20),textcoords = 'offset points', ha = 'right', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.show()
    plt.savefig(img, format='png')
    plot_url = base64.b64encode(img.getbuffer()).decode("ascii")

    return plot_url
