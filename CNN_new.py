#!/usr/bin/env python
# coding: utf-8

import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D

def scale_list(l, to_min, to_max):
    # Fix: handle lists of numpy arrays or numbers
    arr = np.array(l)
    if np.allclose(arr, arr[0]):
        return [np.floor((to_max + to_min) / 2)] * len(l)
    else:
        def scale_number(unscaled, to_min, to_max, from_min, from_max):
            return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min
        return [scale_number(i, to_min, to_max, np.min(arr), np.max(arr)) for i in arr]

def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.fillna(method='ffill')  # Handle missing data
    return data

def prepare_data(data, time_range):
    x, y = [], []
    for i in range(len(data) - time_range):
        x.append(data[i:i + time_range])  # Use a window of TIME_RANGE past returns
        y.append(data[i + time_range])  # Predict the return at the next time step
    return np.array(x), np.array(y)

def calculate_returns(prices):
    returns = (np.diff(prices) / prices[:-1]) * 100  # Percentage change
    return returns

def run_predictions():
    """
    Run predictions and return results in the format expected by collect_model_results.py
    Returns:
        dict: Dictionary containing:
            - dates: list of dates
            - actual_prices: list of actual prices
            - predicted_prices: list of predicted prices
    """
    # Define tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOG']
    start_date = '2021-01-01'  # Extended date range for more data
    end_date = '2022-06-01'

    # Fetch stock data
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    
    if stock_data.empty:
        raise ValueError("No data was downloaded. Please check your tickers and date range.")

    # Calculate returns for each stock
    returns_data = pd.DataFrame()
    for ticker in tickers:
        returns_data[ticker] = calculate_returns(stock_data[ticker].values)
    
    # Prepare data for model
    TIME_RANGE = 20
    x_data, y_data = prepare_data(returns_data.values, TIME_RANGE)
    
    # Reshape input for Conv1D
    x_data = x_data.reshape(x_data.shape[0], TIME_RANGE, len(tickers))
    
    # Split into train and validation
    train_size = int(len(x_data) * 0.8)
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    x_valid = x_data[train_size:]
    y_valid = y_data[train_size:]

    # Define the model
    model = Sequential()
    model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(TIME_RANGE, len(tickers))))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(len(tickers), activation='linear'))

    # Compile and train
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, batch_size=100, epochs=20, verbose=0)

    # Make predictions
    predictions = model.predict(x_valid)
    
    # Convert predictions back to prices
    predicted_prices = []
    actual_prices = []
    for i in range(len(predictions)):
        pred_price = stock_data.iloc[train_size + TIME_RANGE + i].values * (1 + predictions[i]/100)
        actual_price = stock_data.iloc[train_size + TIME_RANGE + i].values
        predicted_prices.append(pred_price.tolist())
        actual_prices.append(actual_price.tolist())

    # Format results
    dates = stock_data.index[train_size + TIME_RANGE:].strftime('%Y-%m-%d').tolist()

    return {
        'dates': dates,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices
    }

if __name__ == "__main__":
    import json
    try:
        result = run_predictions()
        try:
            with open("all_model_results.json", "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
        all_results["CNN_new"] = result
        with open("all_model_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("Saved CNN_new results to all_model_results.json")
    except Exception as e:
        print(f"CNN_new failed: {e}")

