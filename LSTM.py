#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'NFLX']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Download the data for each stock in tech_list
company_data = {}
for stock in tech_list:
    company_data[stock] = yf.download(stock, start=start, end=end)

# Combine the data into a single DataFrame
company_list = [company_data[stock] for stock in tech_list]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON", "TESLA" , "NETFLIX"]

# Add company_name to each DataFrame
for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
# Concatenate all the data into a single DataFrame
df = pd.concat(company_list, axis=0)

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
    start_date = '2020-01-01'  # Extended to 2020 for more data
    end_date = '2022-06-01'

    # Fetch stock data
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    if stock_data.empty:
        raise ValueError("No data was downloaded. Please check your tickers and date range.")
    
    # Create a new dataframe with only the 'Close' column 
    data = stock_data.copy()
    
    # Convert the dataframe to a numpy array
    dataset = data.values
    
    if dataset.size == 0:
        raise ValueError("Dataset is empty after conversion to numpy array")
    
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .95))
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create the training data set 
    train_data = scaled_data[0:int(training_data_len), :]
    
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, :])
        y_train.append(train_data[i, :])
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(len(tickers)))  # Output layer matches number of tickers
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]
    
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, :])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)
    
    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Format results
    dates = data.index[training_data_len:].strftime('%Y-%m-%d').tolist()
    actual_prices = y_test.tolist()
    predicted_prices = predictions.tolist()
    
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
        all_results["LSTM"] = result
        with open("all_model_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("Saved LSTM results to all_model_results.json")
    except Exception as e:
        print(f"LSTM failed: {e}")

