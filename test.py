import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import model_from_json

sc = pickle.load(open('rnnscaler.pkl','rb'))
with open("rnnmodel.json", "r") as f:
    rnn = model_from_json(f.read())
rnn.load_weights("rnnmodel_weights.h5")

Train_dataset=pd.read_csv('Google_Stock_Price_Train.csv.xls')
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv.xls')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((Train_dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price=rnn.predict(X_test)
print(predicted_stock_price)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue',linestyle='dashed', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()