
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
print(tf.version)
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

#FEATURE UNDERSTANDING 

plt.plot(df.index, df['Close'])
plt.xlabel('Time')
plt.ylabel('Closing Stock Price')
plt.title('Stock Value Over the Years')
plt.show() #Shows Closing Stock Prices over the Years

plt.plot(df.index, df['Volume'])
plt.xlabel('Time')
plt.ylabel('Volume Traded')
plt.title('Volume Traded Over the Years')
plt.show()# Shows trends for volume traded over the years

plt.plot(df.index, df['Open'])
plt.xlabel('Time')
plt.ylabel('Opening Stock Price')
plt.title('Opening Stock Price Over the Years')
plt.show()#Shows Opening Stock Prices over the Years

plt.plot(df.index, df['High'])
plt.xlabel('Time')
plt.ylabel('Daily Highest trading Price ')
plt.title('Trading Highs Over the Years')
plt.show()# Shows trends for Daily highest trading price over the years

plt.plot(df.index, df['Low'])
plt.xlabel('Time')
plt.ylabel('Daily Lowesr trading Price ')
plt.title('Trading Lows Over the Years')
plt.show()# Shows trends for Daily highest trading price over the years

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

#Compiling
model.compile( optimizer = 'adam' , loss = 'mean_squared_error')

#Training 
#model.fit(x_train , y_train , batch_size = 1, epochs =10)       #trained on Google Colab

# Load the saved model
open_model = load_model('Open_trained_model.h5')
