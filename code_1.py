
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
from tensorflow.keras.models import load_model
import seaborn as sns
from datetime import timedelta



df_uncleaned = pd.read_csv('Amazon Data.csv')

###############
#DATA CLEANING#
###############

copied_df = df_uncleaned.copy()

df_uncleaned.index = pd.to_datetime(df_uncleaned['Date']) #Setting Date coloumn as index and converting it into date time object
years = df_uncleaned.index.year
months = df_uncleaned.index.month
days = df_uncleaned.index.day
df = df_uncleaned.drop(['Date' , 'Adj Close','Company'] , axis =1 ) #Dropping iniial Date coloumn, adj close and company to avoid duplication and redundancy
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

########################
#INITIALIZING STREAMLIT#
########################

st.title("Stock Price Prediction Web App")
st.write("""
Welcome to the Stock Price Prediction App. This app explores stock data, visualizes relationships, and predicts future prices.

### Project Summary:
This project focuses on three key aspects:

1. **Exploratory Data Analysis (EDA):**
   - Explore the relationships among different stock features.
   - Select various plots to analyze the trends in stock data.

2. **Actual vs. Predicted Values:**
   - Visualize the actual closing stock prices alongside model predictions.
   - Compare how well the model captures the actual trends.

3. **Stock Price Forecasting:**
   - Enter the number of days you want predictions for.
   - Utilize the trained model to forecast future stock prices.

Explore each aspect using the sidebar menu on the left. Have an insightful journey into the world of stock price prediction!

*Note: For forecasting, you can use a trained model and provide the number of days for predictions.*

""")
#####################
#Adding Sidebar######
#####################
workflow_button = st.sidebar.button("Workflow")

if workflow_button:
    st.header('WORKFLOW')
    st.write(
        """
        ## Data Cleaning and Exploration:
        The data cleaning process involved handling missing values in the stock dataset. Null values in the 'Open,' 'Close,' 'High,' 'Low,' and 'Volume' columns were filled using a rolling mean approach. This ensured avoiding a significant change in the mean values of the coloumns which is evident from the Data Description of uncleaned and cleaned dataset

        ## Exploratory Data Analysis (EDA):
        The EDA section of the app allows users to explore various relationships within the stock dataset. Users can choose from a selection of plots, including a correlation matrix heatmap, closing stock prices over the years, volume traded over the years, scatter plots depicting relationships between different features, and distribution plots for closing prices and trading volumes.

        ## Univariate Prediction Models:
        The app provides insights into univariate prediction models for individual stock features (Open, High, Low, and Volume). LSTM (Long Short-Term Memory) neural networks were employed for training these models. After training, users can visualize the model's predictions against actual values for the training and test datasets.

        ## Multivariate Prediction Model:
        A multivariate prediction model was developed to forecast the closing stock prices. The model considers a window of the last 45 days to make predictions for the next day. LSTM layers are used in the neural network architecture for this purpose. The actual vs. predicted closing prices are then visualized.

        ## Forecasting
        The forecasting section enables users to predict future stock prices. Users can input the number of days they want to predict, and the app leverages the trained models to provide forecasts. The process involves predicting future values for open, high, low, close, and volume features using univariate trained models. These predicted features are taken as input to predict the closing stock Price using Multivariate trained Model. Finally, the app visualizes the predicted closing prices alongside the actual data.


        """
    )

st.sidebar.header('UNCLEANED DATA')

data_set_button = st.sidebar.button('Uncleaned Dataset')
if data_set_button:
  st.subheader('Dataset')
  st.write(df_uncleaned)

data_description_button = st.sidebar.button('Uncleaned Data Description')
if data_description_button:
  st.subheader('Data Description')
  st.write(df_uncleaned.describe().T)




st.sidebar.header('CLEANED DATA')

cl_data_set_button = st.sidebar.button('Cleaned Dataset')
if cl_data_set_button:
  st.subheader('Dataset')
  st.write(df)

cl_data_description_button = st.sidebar.button('Cleaned Data Description')
if cl_data_description_button:
  st.subheader('Data Description')
  st.write(df.describe().T)


##################################
#####ADDING EDA TO STREAMLIT######
##################################
  

# EDA Section
st.header("Exploratory Data Analysis (EDA)")

# Dropdown for selecting relationships
selected_relationship = st.selectbox("Select Relationship:", ['Corelation Matrix' ,"Closing Stock Price over the Years", "Volume Traded Over the Years", 'Closing Price Distribution',
'Open vs Close', 'Volume vs Close' , 'Volume Distribution' ])

# Button to trigger the plot
plot_button = st.button("Plot")

# Placeholder for the plot
plot_placeholder = st.empty()

# Check if the button is pressed
if plot_button:
  if selected_relationship == 'Corelation Matrix':
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plot_placeholder = st.pyplot()
    st.write("In the correlation matrix, a strong positive correlation of 1 is observed between Opening Prices, High Prices, Low Prices, and the target variable, indicating a linear relationship where as one variable increases, the others also increase proportionally. This high correlation implies a tight connection among these stock features. However, Volume Traded displays a weaker correlation with the rest of the variables, suggesting a less direct linear relationship. This distinction suggests that the trading volume may not follow the same patterns as the Opening, High, Low prices, and the target variable. The divergence in correlation strengths provides valuable insights into the unique behavior of the trading volume compared to other stock metrics.")

  elif selected_relationship == 'Closing Stock Price over the Years':
    plot_placeholder ,title = line_plot(df.index,df['Close'] , 'Years' , 'Closing Stock Price $' , 'Closing Stock Price over the Years' )
    st.subheader(title)
    st.pyplot()
    st.write("This graph shows the Closing price of the Amazon Stocks over the years depicting a steady growth after 2012 until the most significant peak after 2020 probably due to a new wave of online shopping post covid-19 ")

  elif selected_relationship == 'Volume Traded Over the Years':
    plot_placeholder ,title = line_plot(df.index, df['Volume'] , 'Years' , 'Volume Traded' , 'Volume of Shares Traded over the Years' )
    st.subheader(title)
    st.pyplot()
    st.write("The graph depicting the volume of shares traded over the years reveals a notable trend. In the initial years, up to 2005, there was a substantial surge in trading volume, coinciding with a period when the stock prices were relatively low. This suggests a heightened interest in trading during that period, possibly driven by increased market activity or specific events. However, as the years progressed, there was a consistent decline in trading volume despite a concurrent rise in stock prices. This divergence implies a shift in market dynamics, where the trading activity dwindled even as the stock values appreciated. Such a scenario could be indicative of a market transformation, possibly influenced by changes in investor behavior, market structure, or external factors impacting trading patterns.")

  elif selected_relationship == 'Open vs Close':
    plot_placeholder ,title = scatter_plot('Open' , 'Close' , 'Opening Prices' , 'Closing Prices' , 'Relationship between Opening and Closing Prices' , 'plasma')
    st.subheader(title)
    st.pyplot(plot_placeholder)
    st.write("This scatter plot shows the strong linear relationship between the two values as it could be seen in the Correlation Matrix. The colomns 'High' and 'Low' depicts the same trend and are therefore not plotted.")

  elif selected_relationship == 'Volume vs Close':
    plot_placeholder ,title = scatter_plot('Close' , 'Volume' , 'Closing Prices' , 'Volume of Shares Traded' , 'Relationship between Volume of Shares Traded and Closing Price', 'inferno')
    st.subheader(title)
    st.pyplot()
    st.write("This graph shows the weak negative correlation between the volume of shares traded and the closing prices same as it is shown in the correlation  matrix.The amount of shares traded was relatively high in the initial years when the prices were comparitively low. ")
  
  elif selected_relationship == 'Closing Price Distribution':
    plot_placeholder ,title = create_histogram(df['Close'] , 'Closing Prices' , 'Frequency' , 'Distribution of Stock Prices', color = 'red')
    st.subheader(title)
    st.pyplot()
    st.write("The right skewed distribution of the closing stock price shows that for the most amount of time, the stock prices stayed between 0-20 $ before having an constant upward growth (in 2012 as shown in the Line plot of Closing prices) ")

  elif selected_relationship == 'Volume Distribution':
    plot_placeholder ,title = create_histogram(df['Volume'] , 'Volume of Shares Traded' , 'Frequency' , 'Distribution of Stock Prices', color = 'red')
    st.subheader(title)
    st.pyplot()
    St.write("The right skewed distribution of the volumes traded depicts that mostly the amount of shares traded in a day lie between 0-300 million shares with occasional significant surge in trading activities")


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
#final_model.fit(X_train , Y_train , batch_size = 1, epochs =10)




########################
#ST TEST PLOTTING#######
########################
# Validation Section
st.header("Testing Trained Models")

# Dropdown for selecting relationships
selected_model = st.selectbox("Select Model :", ['Univariate Prediction Model', 'Multivariate Prediction Model'])

# Button to trigger the plot
test_button = st.button("Test and Plot")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

if test_button:


  if selected_model == 'Univariate Prediction Model':

    st.subheader('UniVariate Prediction Model')
    axes[0, 0].set_title('Open Stock Prices')
    validate_and_plot('Open', open_model, 'Opening Stock Prices $', axes[0,0])

    # Plot in the second subplot
    axes[0, 1].set_title('Highest Trading Prices of the Day')
    validate_and_plot('High', high_model, 'Highest Price of the Day $' , axes [0,1])

    # Plot in the third subplot
    axes[1, 0].set_title('Lowest Trading Prices of the Day')
    validate_and_plot('Low', low_model, 'Lowest Price of the Day $',  axes[1,0])

    # Plot in the fourth subplot
    axes[1, 1].set_title('Volume of Shares Traded')
    validate_and_plot('Volume', volume_model, 'Traded Volume $', axes[1,1] )

    # Adjust layout
    #plt.tight_layout()

    # Show the entire figure
    st.pyplot(fig)

  elif selected_model == 'Multivariate Prediction Model':
    st.subheader('Multivariate Prediction Model')
    # #####################
    # #TESTING#############
    # #####################

    final_model = load_model('final_model.h5')
    #Creating test Data
    test_df = scaled_df[training_len-45: ,: ]

    X_test = []
    Y_test = df_for_training.values[training_len : ,3]

    for i in range(45, len(test_df)):
      X_test.append(test_df[i-45:i])

    X_test = np.array(X_test) #Converting into array
    X_test = np.reshape(X_test , (X_test.shape[0], X_test.shape[1], 5)) #Reshaping into 3D

    #Predictions
    predictions = final_model.predict(X_test)
    prediction_copies = np.repeat(predictions , 5 , axis =-1)
    predictions = scaler.inverse_transform(prediction_copies)
    predictions = predictions[:,0]

    ########################
    #TEST PLOTTING##########
    ########################

    train_data_actual = df_for_training[:training_len]
    test_data_actual = df_for_training[training_len:]
    test_data_actual['Predictions'] = predictions
    print(test_data_actual.loc[:,['Close','Predictions']])

    plt.figure(figsize = (12,8))
    plt.xlabel('Date')
    plt.ylabel('Closing Stock Price (USD$)')
    plt.plot(train_data_actual['Close'] , label = 'Training data')
    plt.plot(test_data_actual[['Close' , 'Predictions']] , label = ['Actual test values' , 'Predicted Test Values'] )
    plt.legend()
    plt.show()
    st.pyplot()




########################
#PREDICTING FUTURE######
########################
st.header('Stock Price Forecasting')
days_to_predict = st.number_input("Enter the number of days you want to predict the stock price for:", min_value=1, max_value=500, value=50)

predict_button = st.button("Predict and Plot")

if predict_button:


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


      return df_target

  ###################################
  # Predict future values of features
  ###################################
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

  
  
  
  #####################################
  # Predict future Closing Stock Prices
  #####################################


  # Number of days for the sliding window
  window_size = 45

  scaler = MinMaxScaler(feature_range=(0,1))
  # Function to predict future values
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


        #Make a prediction for the next day
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
  st.write(final_df.tail(days_to_predict+5))

  plt.figure(figsize = (12,8))
  plt.xlabel('Date')
  plt.ylabel('Closing Price USD$')
  plt.plot(final_df.loc[final_df.index[-500:-days_to_predict],'Close']  , label ='Actual Data'  )

  plt.plot(final_df.loc[final_df.index[-days_to_predict:],'Close']  , label ='Prediction'  )
  plt.legend()
  st.pyplot()