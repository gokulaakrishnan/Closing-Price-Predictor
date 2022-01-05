import math
from keras.engine import input_layer
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#getting required inputs
stocks= input("Enter name of stock: ")
startpoint= input("Starting date of dataset, eg. 2021-12-19: ")
endpoint= input("Ending date of dataset, eg. 2021-12-25: ")

#storing the dataset
df= web.DataReader (stocks , data_source= 'yahoo', start= startpoint, end= endpoint)

#filtering out the dataset to only closing price
data= df.filter(['Close'])
dataset= data.values

#identifying no. of rows to train model
datatrain= math.ceil(len(dataset) * .8)

#Scaling the data
scale= MinMaxScaler(feature_range=(0,1))
scale_data= scale.fit_transform(dataset)

#Creating training dataset and scale training dataset
train_dataset= scale_data[0:datatrain , :]

dataox= []
dataoy= []


for i in range(60, len(train_dataset)):
    dataox.append(train_dataset[i-60:i,0])
    dataoy.append(train_dataset[i,0])
    if i<=60:
        print(dataox)
        print(dataoy)
        print()


dataox= np.array(dataox)
dataoy= np.array(dataoy)


#Reshaping to 3d
xrows= dataox.shape[0]
xcolums= dataox.shape[1]
dataox= np.reshape(dataox, (xrows, xcolums, 1)) 


#building LSTM
model= Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (dataox.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


#Compiling the model
model.compile(optimizer='adam', loss= 'mean_squared_error')


#Training Model
model.fit(dataox, dataoy, batch_size=1, epochs=1)

#Testing dataset
datatest= scale_data[datatrain-60:,:]

x_testing=[]
y_testing= dataset[datatrain:,:]
for i in range (60, len(datatest)):
    x_testing.append(datatest[i-60:i,0])

x_testing= np.array(x_testing)

#reshaping
x_testing = np.reshape(x_testing, (x_testing.shape[0], x_testing.shape[1], 1))

#building model
prediction_values= model.predict(x_testing)
prediction_values= scale.inverse_transform(prediction_values)

#Get RMSE
rmse= np.sqrt(np.mean(prediction_values - y_testing)**2)


#Plotting the graph
train= data[:datatrain]
validate= data[datatrain:]
validate['Predicted Values']= prediction_values
plt.figure(figsize= (18,10))
plt.title('Stock Predictor')
plt.xlabel('Date', fontsize= 25)
plt.ylabel('Closing Price', fontsize= 25)
plt.plot(train['Close'])
plt.plot(validate[['Close', 'Predicted Values']])
