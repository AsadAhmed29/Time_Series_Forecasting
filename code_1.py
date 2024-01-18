
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

# # DATA PRE PROCESSING#

def uni_variate_training(target):
  one_feature_df = df.filter([target]) #filtering
  one_feature_array = one_feature_df.values # converting into an array
  training_array_len  = np.int64(len(one_feature_array)*0.8)
  from sklearn.preprocessing import MinMaxScaler
  #Scaling the Data

  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_array = scaler.fit_transform(one_feature_array)

  #spliting the Data
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

  x_train = np.reshape(x_train, (x_train.shape[0], training_days, 1))


  #Building the model
  from keras.models import Sequential
  from keras.layers import Dense, LSTM


  model = Sequential()
  model.add(LSTM(50, return_sequences = True , input_shape = (training_days,1 ) ) )
  model.add(LSTM(50, return_sequences = False))
  model.add(Dense(25))
  model.add(Dense(1))

  #Compile the model
  model.compile( optimizer = 'adam' , loss = 'mean_squared_error')

  #Training the model
  model.fit(x_train , y_train , batch_size = 1, epochs =10)

  return model

def validate_and_plot(target ,model, ylabel):
  one_feature_df = df.filter([target]) #filtering
  one_feature_array = one_feature_df.values # converting into an array
  training_array_len  = np.int64(len(one_feature_array)*0.8)

  from sklearn.preprocessing import MinMaxScaler

  scaler = MinMaxScaler(feature_range=(0,1))

  scaled_array = scaler.fit_transform(one_feature_array)
  test_array = scaled_array[training_array_len-45: ,: ]

  x_test = []
  y_test = one_feature_array[training_array_len: ,:]

  for i in range(45, len(test_array)):
    x_test.append(test_array[i-45:i])

  x_test = np.array(x_test) #Converting into array
  x_test = np.reshape(x_test , (x_test.shape[0], x_test.shape[1], 1)) #Reshaping into 3D

  #Predictions
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  #Getting Rmse
  rmse = np.sqrt(np.mean(predictions-y_test)**2)


  train_data_actual = one_feature_df[:training_array_len]
  test_data_actual = one_feature_df[training_array_len:]
  test_data_actual['Predictions'] = predictions


  plt.figure(figsize = (12,8))
  plt.xlabel('Date')
  plt.ylabel(ylabel)
  plt.plot(train_data_actual , label = 'Training data')
  plt.plot(test_data_actual[[target , 'Predictions']] , label = ['Actual test values' , 'Predicted Test Values'] )
  plt.legend()
  return plt.show()

#open_model = uni_variate_training('Open')     # Trained and saved on colab#
#save_model(open_model, 'open_model.h5')       #                           #


open_model = load_model('open_model.h5')
validate_and_plot('Open' , open_model, 'Opening Stock Prices $')


#high_model = uni_variate_training('High')      # Trained and saved on colab#
#save_model(high_model, 'high_model.h5')        #                           #
high_model = load_model('high_model.h5')
validate_and_plot('High', high_model , 'Highest Price of the Day $')

#low_model = uni_variate_training('Low')        # Trained and saved on colab#
#save_model(low_model, 'low_model.h5')          #                           #
low_model = load_model('low_model.h5')
validate_and_plot('Low', low_model , 'Lowest Price of the Day $')

#volume_model = uni_variate_training('Volume')  # Trained and saved on colab#
#save_model(volume_model, 'volume_model.h5')    #                           #
Volume_model = load_model('volume_model.h5')
validate_and_plot('Volume', volume_model , 'Traded Volume')



