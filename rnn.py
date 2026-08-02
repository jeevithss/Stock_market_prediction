import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import pickle


Train_dataset=pd.read_csv('Google_Stock_Price_Train.csv.xls')
train_dataset=Train_dataset.iloc[:, 1:2].values

sc=MinMaxScaler(feature_range=(0,1))
train_dataset=sc.fit_transform(train_dataset)

pickle.dump(sc, open('rnnscaler.pkl','wb'))

X_train=[]
Y_train=[]

for i in range(60,1258):
    X_train.append(train_dataset[i-60:i,0])
    Y_train.append(train_dataset[i,0])
X_train=np.array(X_train)
Y_train=np.array(Y_train)




X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


rnn=keras.models.Sequential()

rnn.add(keras.layers.LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
rnn.add(keras.layers.Dropout(0.2))

rnn.add(keras.layers.LSTM(units=50,return_sequences=True))
rnn.add(keras.layers.Dropout(0.2))

rnn.add(keras.layers.LSTM(units=50,return_sequences=True))
rnn.add(keras.layers.Dropout(0.2))

rnn.add(keras.layers.LSTM(units=50))
rnn.add(keras.layers.Dropout(0.2))

rnn.add(keras.layers.Dense(units=1))

rnn.compile(optimizer='adam',loss='mean_squared_error')
print(X_train.shape,Y_train.shape)
model = rnn.fit(X_train,Y_train , epochs=100,batch_size=64)

x = model.history['loss']
acc = 1- x[0]
print("Accuracy is ------ ", acc*100, "%")