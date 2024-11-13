# Stock Price Prediction Using LSTM

This project demonstrates the application of **Long Short-Term Memory (LSTM)** networks for predicting stock prices, specifically for **Open**, **High**, **Low**, and **Volume** of stock data. It uses a univariate and multivariate approach to make predictions based on historical stock data.

## Project Overview

This project uses a dataset from **Amazon**'s stock price data (`Amazon Data.csv`). The dataset contains stock information including the **Open**, **High**, **Low**, **Close**, and **Volume** for each day. The goal is to train a model to predict future stock prices and volumes using LSTM networks, a type of Recurrent Neural Network (RNN) that is effective for time series forecasting.

## Data Cleaning

- The dataset is cleaned by setting the **Date** column as the index and converting it into a `datetime` object.
- Missing values are handled by applying a rolling mean with a window of 5 to fill gaps in data for the **Open**, **Close**, and **Volume** columns.
- After cleaning, the dataset is prepared for training and visualization.

## Univariate Stock Price Prediction

The project first uses a **univariate** approach, where each feature (Open, High, Low, and Volume) is predicted using only its own past values. The data is scaled using **MinMaxScaler**, then split into training and testing datasets. The model is built using **LSTM** layers, trained for 10 epochs with a batch size of 1.

### Functions:
1. `uni_variate_training(target)` - Trains a univariate LSTM model on the given target column (e.g., 'Open', 'High').
2. `validate_and_plot(target, model, ylabel)` - Validates the model by predicting on test data and visualizing the results.

## Multivariate Stock Price Prediction

The project also performs **multivariate** analysis, where the model takes into account multiple features (Open, High, Low, Close, Volume) to predict the **Close** price. The data is similarly scaled and split, but the input to the LSTM model is multidimensional (5 features over 45 days).

### Function:
- `multivariate_training()` - Trains a multivariate LSTM model using multiple features to predict stock prices.

## Interactive Visualizations

The project includes an interactive web application using **Streamlit** that allows users to:
- View the cleaned dataset.
- View statistical descriptions of the dataset.
- Select and test different models (univariate and multivariate).
- Visualize the predicted values against actual values for various stock features.

### Features:
- **Univariate Prediction Models**: Predicts stock features individually.
- **Multivariate Prediction Model**: Predicts the closing stock price using multiple features.
- **Visualization**: Displays training and testing data with predicted stock values.
