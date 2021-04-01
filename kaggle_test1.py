#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 01:02:28 2021

@author: mingjunliu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
from datetime import datetime


df = DataReader('AMZN', data_source='yahoo', start='2010-07-20', end='2021-01-06')

plt.figure(figsize=(10,5))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))


# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(248, len(train_data)):
    x_train.append(train_data[i-248:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 249:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#%%

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=10)


# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 248: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]


for i in range(248, len(test_data)):
    x_test.append(test_data[i-248:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# mean absolute percentage error
mape = np.mean(abs(predictions - y_test) / y_test)


# Plot the data
train = data[-500:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data

#%%
plt.style.use("fivethirtyeight")

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#%%
# moving average

ma_data = data['Close'][-500:]#.rolling(5).mean()[4:]
ma_train = ma_data[:len(ma_data) - len(y_test)]
ma_valid = ma_data[len(ma_data) - len(y_test):]
ma_preds = []
for i in range(0, len(ma_valid)):
    a = ma_train[len(ma_train)-200+i:].sum() + sum(ma_preds)
    b = a/200
    ma_preds.append(b)
ma_rmse = np.sqrt(np.mean((ma_preds-ma_valid) ** 2))
ma_mape = np.mean(abs(ma_preds-ma_valid) / ma_valid)

plt.figure(figsize=(16,8))
plt.plot(ma_train)
plt.plot(ma_valid)
plt.plot(ma_valid.index,ma_preds)
plt.show()

#%%
# rolling arima

from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math


arima_data = data['Close'][-500:]
arima_train = arima_data[:len(arima_data) - len(y_test)]
arima_valid = arima_data[len(arima_data) - len(y_test):]
arima_history = [x for x in arima_train]
arima_pred = []

# model = ARIMA(arima_train, order=(2,1,2))
# model_fit = model.fit(arima_train)
# # arima_pred = model.predict(n_periods=248)


for i in range(len(arima_valid)):
    arima_model = ARIMA(arima_history, order=(2,1,0))
    model_fit = arima_model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    arima_pred.append(yhat)
    obs = arima_valid[i]
    arima_history.append(obs)


arima_rmse = np.sqrt(np.mean((arima_valid-arima_pred) ** 2))
arima_mape = np.mean(abs(arima_pred-arima_valid) / arima_valid)

plt.plot(arima_train[-500:])
plt.plot(arima_valid)
plt.plot(arima_valid.index,arima_pred)
plt.show()

#%%
#svm, linear regression

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

y_test = scaled_data[training_data_len:, :]


svm_data = data['Close'][-500:]
svm_train = svm_data[:len(svm_data) - len(y_test)]
svm_valid = svm_data[len(svm_data) - len(y_test):]
svm_history_x = [[i for j in x for i in j] for x in x_train]
svm_history_y = [x for x in y_train]
svm_pred = []
# lr_pred = []

# model = ARIMA(arima_train, order=(2,1,2))
# model_fit = model.fit(arima_train)
# # arima_pred = model.predict(n_periods=248)

for i in tqdm(range(len(svm_valid))):
    svm_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svm_rbf.fit(svm_history_x, svm_history_y)
    
    # lr = LinearRegression()
    # lr.fit(svm_history_x, svm_history_y)
    
    xi = [i for j in x_test[i] for i in j]
    output = svm_rbf.predict([xi])
    
    # lr_pred.append(lr.predict([xi]))
    svm_pred.append(output[0])
    
    svm_history_x.append(xi)
    svm_history_y.append(y_test[i][0])
    
    # yhat = output[0]
    # svm_pred.append(yhat)
    # obs = svm_valid[i]
    # svm_history.append(obs)


svm_pred = [[i] for i in svm_pred]
svm_pred = scaler.inverse_transform(svm_pred)

plt.figure(figsize=(16,8))
plt.plot(svm_train[-500:])
plt.plot(svm_valid)
plt.plot(svm_valid.index,svm_pred)
plt.show()


svm_pred = [i for j in svm_pred for i in j]
svm_rmse = np.sqrt(np.mean((svm_valid-svm_pred) ** 2))
svm_mape = np.mean(abs(svm_pred-svm_valid) / svm_valid)
















