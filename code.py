
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Amazon Data.csv')
copied_df = df.copy()

df.index = pd.to_datetime(df['Date']) #Setting Date coloumn as index and converting it into date time object
years = df.index.year
months = df.index.month
days = df.index.day
df = df.drop(['Date' , 'Company'] , axis =1 ) #Dropping iniial Date coloumn to avoid duplication
df = df.sort_index() 

def custom_mean(series):
    return series.mean(skipna=True)

df['Open'] = df['Open'].fillna(df['Open'].rolling(window=5, center=True , min_periods=1).apply(custom_mean))
df['Close'] = df['Close'].fillna(df['Close'].rolling(window=5, center=True , min_periods=1).apply(custom_mean))
df['Close'] = df['Close'].fillna(df['Close'].rolling(window=5, center=True , min_periods=1).apply(custom_mean)) #Running twice incase there are 5 consecutive NaN values 
df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(window=5, center=True , min_periods=1).apply(custom_mean))
df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(window=5, center=True , min_periods=1).apply(custom_mean))

