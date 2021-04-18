import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# %% md


result = pd.read_pickle("/Users/pz/Desktop/560/560-proj/merge.pkl")

# original time series (Y)
y = result.MSFT.values
y = y.astype('float32')
y = np.reshape(y, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(y)

# training and testing settings (size)
# percent_of_training = 0.7

train_size = int(len(y)-5)
test_size = len(y) - train_size
train_y, test_y = y[0:train_size, :], y[train_size:len(y), :]
print(len(train_y))
print(len(test_y))
# percent_of_training = 0.7
# train_size = int(len(y) * percent_of_training)
# test_size = len(y) - train_size
# train_y, test_y = y[0:train_size, :], y[train_size:len(y), :]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 1

# features of the original time serie (y)
X_train_features_1, y_train = create_dataset(train_y, look_back)
X_test_features_1, y_test = create_dataset(test_y, look_back)

# join the all the features in one
## reshape arrays
X_train_features = np.reshape(X_train_features_1, (X_train_features_1.shape[0], 1, X_train_features_1.shape[1]))
X_test_features = np.reshape(X_test_features_1, (X_test_features_1.shape[0], 1, X_test_features_1.shape[1]))


model = Sequential()
model.add(LSTM(128, input_shape=(X_train_features.shape[1], X_train_features.shape[2])))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train_features, y_train, epochs=300, batch_size=25, validation_data=(X_test_features, y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0, shuffle=False)

model.summary()
#
# # %%
#
train_predict = model.predict(X_train_features)
test_predict = model.predict(X_test_features)
print("train predict",len(train_predict))
print("test predict",test_predict)
#
#
print('Train Mean Absolute Error:',
      mean_absolute_error(np.reshape(y_train, (y_train.shape[0], 1)), train_predict[:, 0]))
print('Train Root Mean Squared Error:',
      np.sqrt(mean_squared_error(np.reshape(y_train, (y_train.shape[0], 1)), train_predict[:, 0])))
print('Test Mean Absolute Error:', mean_absolute_error(np.reshape(y_test, (y_test.shape[0], 1)), test_predict[:, 0]))
print('Test Root Mean Squared Error:',
      np.sqrt(mean_squared_error(np.reshape(y_test, (y_test.shape[0], 1)), test_predict[:, 0])))

# # %%
#
plt.figure(figsize=(8, 4))
plt.style.use('seaborn-dark')

plt.plot(history.history['loss'], label='Train Loss', color="green")
plt.plot(history.history['val_loss'], label='Test Loss', color="yellow")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.grid()

plt.show();
#
#
#
time_y_train = pd.DataFrame(data=train_y, index=result[0:train_size].index, columns=[""])
time_y_test = pd.DataFrame(data=test_y, index=result[train_size:].index, columns=[""])

time_y_train_prediction = pd.DataFrame(data=train_predict, index=time_y_train[2:].index, columns=[""])
time_y_test_prediction = pd.DataFrame(data=test_predict, index=time_y_test[2:].index, columns=[""])

# time_y_train_prediction = pd.DataFrame(data=train_predict, index=time_y_train[8:].index, columns=[""])
# time_y_test_prediction = pd.DataFrame(data=test_predict, index=time_y_test[8:].index, columns=[""])

plt.style.use('seaborn-dark')
plt.figure(figsize=(15, 10))

plt.plot(time_y_train, label="training", color="green", marker='.')
plt.plot(time_y_test, label="test", marker='.')
plt.plot(time_y_train_prediction, color="red", label="prediction")
plt.plot(time_y_test_prediction, color="red")
plt.title("LSTM fit of Microsoft Stock Market Prices", size=20)
plt.tight_layout()
sns.despine(top=True)
plt.ylabel('', size=15)
plt.xlabel('', size=15)
plt.legend(fontsize=15)
plt.grid()

plt.show();