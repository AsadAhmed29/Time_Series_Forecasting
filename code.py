
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
