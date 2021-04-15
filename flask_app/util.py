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
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

matplotlib.use('Agg')
plt.style.use("fivethirtyeight")


def search_stock(key):

    news = []
    
    start_page = 'https://investing.com/search/?q=' + key
    page = requests.get(start_page, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(page.text, 'html.parser')
    
    search_link = soup.find(class_='searchSectionMain').find('a').get('href')
    stock_link = 'https://investing.com' + search_link + '-news'
    stock_page = requests.get(stock_link, headers={"User-Agent": "Mozilla/5.0"})
    stock_soup = BeautifulSoup(stock_page.text, 'html.parser')
    
    news_article = stock_soup.find_all('article')
    for each in news_article:
        if(each.find(class_='date')):
            news.append({'time': each.find(class_='date').contents[0][3:], 'content': each.find(class_='title').contents[0]})
    return news


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

def moving_average(keys):
    return plot_url
