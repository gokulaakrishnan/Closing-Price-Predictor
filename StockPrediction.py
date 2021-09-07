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
name= input("Enter name of stock: ")
start_date= input("Starting date of dataset, eg. 2021-12-19: ")
end_date= input("Ending date of dataset, eg. 2021-12-25: ")

#storing the dataset
df= web.DataReader (name , data_source= 'yahoo', start= start_date, end= end_date)

#filtering out the dataset to only closing price
data= df.filter(['Close'])
dataset= data.values

#identifying no. of rows to train model
train_data= math.ceil(len(dataset) * .8)

#Scaling the data
scale= MinMaxScaler(feature_range=(0,1))
scale_data= scale.fit_transform(dataset)

#Creating training dataset and scale training dataset
train_dataset= scale_data[0:train_data , :]

x_data= []
y_data= []


for i in range(60, len(train_dataset)):
    x_data.append(train_dataset[i-60:i,0])
    y_data.append(train_dataset[i,0])
    if i<=60:
        print(x_data)
        print(y_data)
        print()


x_data= np.array(x_data)
y_data= np.array(y_data)


#Reshaping to 3d
rows_x= x_data.shape[0]
cols_x= x_data.shape[1]
x_data= np.reshape(x_data, (rows_x, cols_x, 1)) 


#building LSTM
model= Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_data.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


#Compiling the model
model.compile(optimizer='adam', loss= 'mean_squared_error')


#Training Model
model.fit(x_data, y_data, batch_size=1, epochs=1)

#Testing dataset
testing_data= scale_data[train_data-60:,:]

x_testing=[]
y_testing= dataset[train_data:,:]
for i in range (60, len(testing_data)):
    x_testing.append(testing_data[i-60:i,0])

x_testing= np.array(x_testing)

#reshaping
x_testing = np.reshape(x_testing, (x_testing.shape[0], x_testing.shape[1], 1))

#building model
prediction_values= model.predict(x_testing)
prediction_values= scale.inverse_transform(prediction_values)

#Get RMSE
rmse= np.sqrt(np.mean(prediction_values - y_testing)**2)


#Plotting the graph
train= data[:train_data]
validation_data= data[train_data:]
validation_data['Predicted Values']= prediction_values
plt.figure(figsize= (18,10))
plt.title('Stock Predictor')
plt.xlabel('Date', fontsize= 20)
plt.ylabel('Closing Price USD', fontsize= 20)
plt.plot(train['Close'])
plt.plot(validation_data[['Close', 'Predicted Values']])


