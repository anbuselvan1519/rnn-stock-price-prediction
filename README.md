# Stock Price Prediction

### NAME: ANBUSELVAN.S
### REFERENCE NO: 212223240008

## Aim:

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset:
Make a Recurrent Neural Network model for predicting stock price using a training and testing dataset. The model will learn from the training dataset sample and then the testing data is pushed for testing the model accuracy for checking its accuracy. The Dataset has features of market open price, closing price, high and low price for each day.
![image](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/139841744/f89a3123-36e4-4ffb-a712-4a6f4e8eb9eb)

## Design Steps:

### Step 1:
Write your own steps

### Step 2:
Loat the dataset

### Step 3:
Create the model and compile

### Step 4:
Fit the model. Evaluate the model.

## Program:
```
Developed by: ANBUSELVAN.S
Register No: 212223240008
```
### IMPORTING FILES:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
```
### TRAINING DATA:
```
dataset_train = pd.read_csv('trainset.csv')
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
```
### MODEL CREATION:
```
model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(60,1)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
```
### FITTING AND TESTING OF MODEL:
```
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
```
### PREDICTION:
```
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='black', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
## Output:

![image](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/139841744/f1cfeec1-5ba9-4b9e-a46b-f8029a3bcbc8)

### True Stock Price, Predicted Stock Price vs time:

![image](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/139841744/7bcbc23f-d42d-49b9-9ac6-5bf09e6216e1)

### Mean Square Error:

![image](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/139841744/697d1b36-9bf9-43e5-a1cb-2bcf9d09e8ee)

## Result:
A Recurrent Neural Network model for stock price prediction is developed.
