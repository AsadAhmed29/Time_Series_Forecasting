
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
from tensorflow.keras.models import load_model



df = pd.read_csv('Amazon Data.csv')

#DATA CLEANING#
copied_df = df.copy()

df.index = pd.to_datetime(df['Date']) #Setting Date coloumn as index and converting it into date time object
years = df.index.year
months = df.index.month
days = df.index.day
df = df.drop(['Date' , 'Adj Close','Company'] , axis =1 ) #Dropping iniial Date coloumn, adj close and company to avoid duplication and redundancy
df = df.sort_index() 

def custom_mean(series):
    return series.mean(skipna=True)

df['Open'] = df['Open'].fillna(df['Open'].rolling(window=5, center=True , min_periods=1).apply(custom_mean))
df['Close'] = df['Close'].fillna(df['Close'].rolling(window=5, center=True , min_periods=1).apply(custom_mean))
df['Close'] = df['Close'].fillna(df['Close'].rolling(window=5, center=True , min_periods=1).apply(custom_mean)) #Running twice incase there are 5 consecutive NaN values 
df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(window=5, center=True , min_periods=1).apply(custom_mean))
df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(window=5, center=True , min_periods=1).apply(custom_mean))

# #FEATURE UNDERSTANDING 

# plt.plot(df.index, df['Close'])
# plt.xlabel('Time')
# plt.ylabel('Closing Stock Price')
# plt.title('Stock Value Over the Years')
# plt.show() #Shows Closing Stock Prices over the Years

# plt.plot(df.index, df['Volume'])
# plt.xlabel('Time')
# plt.ylabel('Volume Traded')
# plt.title('Volume Traded Over the Years')
# plt.show()# Shows trends for volume traded over the years

# plt.plot(df.index, df['Open'])
# plt.xlabel('Time')
# plt.ylabel('Opening Stock Price')
# plt.title('Opening Stock Price Over the Years')
# plt.show()#Shows Opening Stock Prices over the Years

# plt.plot(df.index, df['High'])
# plt.xlabel('Time')
# plt.ylabel('Daily Highest trading Price ')
# plt.title('Trading Highs Over the Years')
# plt.show()# Shows trends for Daily highest trading price over the years

# plt.plot(df.index, df['Low'])
# plt.xlabel('Time')
# plt.ylabel('Daily Lowesr trading Price ')
# plt.title('Trading Lows Over the Years')
# plt.show()# Shows trends for Daily highest trading price over the years

#
# UNIVARIATE PREDICTION USING LSTM
#

# DATA PRE PROCESSING#

one_feature_df = df.filter(['Open']) 
one_feature_array = one_feature_df.values # converting into an array
training_array_len  = np.int64(len(one_feature_array)*0.8)


#Scaling
scaler = MinMaxScaler(feature_range=(0,1))                               
scaled_array = scaler.fit_transform(one_feature_array)

#Spliting the Data
training_data = scaled_array[0:training_array_len,0]
training_days = 45
x_train = []
y_train = []

for i in range(training_days,training_array_len):
  x_train.append(training_data[i-training_days:i])
  y_train.append(training_data[i])


#converting to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

#Reshaping into 3D for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], training_days, 1))


#BUILDING THE MODEL

from keras.models import Sequential
from keras.layers import Dense, LSTM


model = Sequential()
model.add(LSTM(50, return_sequences = True , input_shape = (training_days,1 ) ) )
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

# Load the saved model
open_model = load_model('Open_trained_model.h5')

#Creating test data
test_array = scaled_array[training_array_len-training_days: ,: ]

x_test = []
y_test = one_feature_array[training_array_len: ,:]
len(y_test)

for i in range(training_days, len(test_array)):
  x_test.append(test_array[i-training_days:i])

x_test = np.array(x_test) #Converting into array
x_test = np.reshape(x_test , (x_test.shape[0], x_test.shape[1], 1)) #Reshaping into 3D

# #Making Predictions
# predictions = open_model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)

# rmse = np.sqrt(np.mean(predictions-y_test)**2)
# accuracy = rmse

# #Plotting
# import matplotlib.pyplot as plt

# train_data_actual = one_feature_df[:training_array_len]
# test_data_actual = one_feature_df[training_array_len:]
# test_data_actual['Predictions'] = predictions


# plt.figure(figsize = (12,8))
# plt.xlabel('Date')
# plt.ylabel('Open Price USD$')
# plt.plot(train_data_actual , label = 'Training data')
# plt.plot(test_data_actual[['Open' , 'Predictions']] , label = ['Actual test values' , 'Predicted Test Values'] )
# plt.legend()
# plt.show()
print('5')