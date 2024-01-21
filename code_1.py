
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
from tensorflow.keras.models import load_model
from datetime import timedelta



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

cor_matrix = df.corr()

def line_plot( x, y, xlabel , ylabel , title):

  plt.plot(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  #plt.title(title)
  return plt.show() ,title

def scatter_plot(y1,y2 , xlabel , ylabel , title , c):
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.scatter(df[y1], df[y2] , c = df[y2], cmap =c)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)

  return plt.show(),title

def create_histogram(data, xlabel, ylabel, title, bins=10, color='blue', alpha=0.7):
  plt.hist(data, bins=bins, color=color, alpha=alpha)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.show()

  return plt.show() , title


##################################
# UNIVARIATE PREDICTION USING LSTM
##################################

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
#save_model(open_model, 'open_model.h5')       # ###########################


open_model = load_model('open_model.h5')
validate_and_plot('Open' , open_model, 'Opening Stock Prices $')


#high_model = uni_variate_training('High')      # Trained and saved on colab#
#save_model(high_model, 'high_model.h5')        # ###########################
high_model = load_model('high_model.h5')
validate_and_plot('High', high_model , 'Highest Price of the Day $')

#low_model = uni_variate_training('Low')        # Trained and saved on colab#
#save_model(low_model, 'low_model.h5')          #  ##########################
low_model = load_model('low_model.h5')
validate_and_plot('Low', low_model , 'Lowest Price of the Day $')

#volume_model = uni_variate_training('Volume')  # Trained and saved on colab#
#save_model(volume_model, 'volume_model.h5')    # ###########################
volume_model = load_model('volume_model.h5')
validate_and_plot('Volume', volume_model , 'Traded Volume')



##############################################
#Multivariate Analysis of Closing Stock Prices#
##############################################
df_for_training = df.copy()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_df = scaler.fit_transform(df_for_training)
#Input Reshaping
training_len  = np.int64(len(scaled_df)*0.8)
training_data = scaled_df[0:training_len]
training_days = 45


#Splitting Data into Train Data
X_train = []
Y_train = []

for i in range(training_days,training_len):
    X_train.append(training_data[i-training_days:i])
    Y_train.append(training_data[i,3])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], training_days, 5))

#Building the model
from keras.models import Sequential
from keras.layers import Dense, LSTM


final_model = Sequential()
final_model.add(LSTM(50, return_sequences = True , input_shape = (X_train.shape[1], X_train.shape[2]) ) )
final_model.add(LSTM(50, return_sequences = False))
final_model.add(Dense(25))
final_model.add(Dense(1))

#Compiling
final_model.compile( optimizer = 'adam' , loss = 'mean_squared_error')

#Training the model
final_model.fit(X_train , Y_train , batch_size = 1, epochs =10)


#####################
#TESTING#############
#####################

#Creating test Data
test_df = scaled_df[training_len-45: ,: ]

X_test = []
Y_test = df_for_training.values[training_len : ,3]

for i in range(45, len(test_df)):
  X_test.append(test_df[i-45:i])

X_test = np.array(X_test) #Converting into array
X_test = np.reshape(X_test , (X_test.shape[0], X_test.shape[1], 5)) #Reshaping into 3D
X_test.shape

#Predictions
predictions = final_model.predict(X_test)
prediction_copies = np.repeat(predictions , 5 , axis =-1)
predictions = scaler.inverse_transform(prediction_copies)
predictions = predictions[:,0]

########################
#PLOTTING###############
########################

train_data_actual = df_for_training[:training_len]
test_data_actual = df_for_training[training_len:]
test_data_actual['Predictions'] = predictions
print(test_data_actual.loc[:,['Close','Predictions']])

plt.figure(figsize = (12,8))
plt.xlabel('Date')
plt.ylabel('Closing Price USD$')
plt.plot(train_data_actual['Close'] , label = 'Training data')
plt.plot(test_data_actual[['Close' , 'Predictions']] , label = ['Actual test values' , 'Predicted Test Values'] )
plt.legend()
plt.show()



########################
#PREDICTING FUTURE######
########################
# Number of days for the sliding window
window_size = 45

scaler = MinMaxScaler(feature_range=(0,1))
# Function to predict future values
def predict_future(df, target, model, days_to_predict):
    df_target = df[target].copy()
    window = df_target.values.reshape(-1,1)
    scaler.fit(window)

    for i in range(0,days_to_predict):
        # Create a window of the last 'window_size' days
        window = df_target.values[-window_size:]
        window = window.reshape(-1,1)
        window = scaler.transform(window)
        window = np.array(window) #Converting into array
        window = np.reshape(window , (1,window.shape[0], 1))

        # Make a prediction for the next day
        prediction = model.predict(window)
        prediction = scaler.inverse_transform(prediction.reshape(-1,1))
        # Append the predicted value to the DataFrame
        next_date = df_target.index[-1] + timedelta(days=1)
        for value in prediction.flatten():
            df_target.loc[next_date] = value
            next_date += timedelta(days=1)

        # Concatenate the new DataFrame to the original DataFrame

    return df_target

# Get user input for the number of days to predict
days_to_predict = int(input("Enter the number of days to predict: "))


####################################
# Predict future values of features#
####################################
def predict_features(days_to_predict):
  df_open = predict_future(df, 'Open', open_model, days_to_predict)
  df_high = predict_future(df, 'High', high_model, days_to_predict)
  df_low = predict_future(df, 'Low' , low_model, days_to_predict)
  df_volume = predict_future(df, 'Volume' , volume_model , days_to_predict)
  df_close = df['Close']
  next_date = df_close.index[-1] + timedelta(days=1)
  for value in range(days_to_predict):
    df_close.loc[next_date] = 0
    next_date += timedelta(days=1)

  df_with_predicted_features = pd.concat([df_open , df_high,df_low, df_close , df_volume, ] ,axis= 1)
  return df_with_predicted_features

df_with_predicted_features = predict_features(days_to_predict)

final_model = load_model('final_model.h5')

# Number of days for the sliding window
window_size = 45

scaler = MinMaxScaler(feature_range=(0,1))

############################################
# Function to predict future Closing values#
############################################
def predict_future_final(df, model, days_to_predict):

  df_tar = df.copy()
  window = df_tar.values.reshape(-1,5)
  scaler.fit(window)

  for i in range(0,days_to_predict):
      # Create a window of the last 'window_size' days
      window = df_tar.values[-10+1-45:-10+1]
      window = window.reshape(-1,5)
      window = scaler.transform(window)
      window = np.array(window) #Converting into array
      window = np.reshape(window , (1,window.shape[0], 5))
      

      # Make a prediction for the next day
      prediction = model.predict(window)
      prediction_copies = np.repeat(prediction , 5 , axis =-1)
      prediction = scaler.inverse_transform(prediction_copies)
      prediction = prediction[:,0]
      # Append the predicted value to the DataFrame
      # next_date = df['date'].max() + timedelta(days=1)
      # df_target = df.append({ target: prediction}, ignore_index=True)
      #next_date = df_target.index[-1] + timedelta(days=1)
      for value in prediction.flatten():
          df_tar.loc[df_tar.index[-days_to_predict+i], 'Close'] = value
          #next_date += timedelta(days=1)
  return df_tar

final_df = predict_future_final(df_with_predicted_features,final_model,days_to_predict)
print(final_df.tail(15))

plt.figure(figsize = (12,8))
plt.xlabel('Date')
plt.ylabel('Close Price USD$')
plt.plot(final_df.loc[final_df.index[-500:-days_to_predict],'Close']  , label ='Actual Data'  )

plt.plot(final_df.loc[final_df.index[-days_to_predict:],'Close']  , label ='Prediction'  )
plt.legend()
plt.show()

