#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import tensorflow as tf
import yfinance as yf
from sklearn.cluster import AgglomerativeClustering
import json
import pandas as pd

# Function to fetch stock data from yfinance
def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data was downloaded. Check your internet connection or ticker symbols.")
    # Close column is the adjusted close column
    if 'Close' in data:
        data = data['Close']
    elif isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.get_level_values(0):
        data = data['Close']
    else:
        raise ValueError("'Close' column not found in downloaded data. Columns are: {}".format(data.columns))
    data = data.fillna(method='ffill')  # Fill missing data
    return data

# Function to compute correlation matrix and clusters
def create_hyperedges_from_correlation(stock_data, num_clusters):
    returns = stock_data.pct_change().dropna()  # Compute daily returns
    correlation_matrix = returns.corr()  # Correlation matrix of returns

    # Convert correlation matrix to distance matrix (1 - correlation)
    distance_matrix = 1 - correlation_matrix

    # Perform hierarchical clustering to group similar stocks into hyperedges
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(distance_matrix)

    # Create incidence matrix based on clustering
    num_tickers = len(stock_data.columns)
    H = np.zeros((num_tickers, num_clusters))

    for ticker_idx, cluster_label in enumerate(labels):
        H[ticker_idx, cluster_label] = 1  # Assign ticker to its cluster (hyperedge)

    return H, correlation_matrix

# Custom LSTM-Hypergraph Model
class LSTMHypergraphModel(tf.keras.Model):
    def __init__(self, lstm_units, num_tickers, num_clusters):
        super(LSTMHypergraphModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False, name='lstm_layer')
        self.dense = tf.keras.layers.Dense(num_tickers, name='dense_output')  # Output for each ticker

    def call(self, inputs, incidence_matrix):
        stock_data_seq = inputs  # Time series input data
        lstm_output = self.lstm(stock_data_seq)  # LSTM processes the sequential data
        stock_features = self.dense(lstm_output)  # Output layer, one for each ticker

        # Apply the incidence matrix to model relationships between stocks
        # Using matrix multiplication of stock features with the incidence matrix
        hypergraph_output = tf.matmul(stock_features, incidence_matrix)

        # Make sure the final output matches the number of tickers
        final_output = tf.matmul(hypergraph_output, tf.transpose(incidence_matrix))
        return final_output

def calculate_returns(stock_data):
    """Calculates daily returns."""
    returns = stock_data.pct_change().dropna()
    return returns

# Test function with RMSE calculation
def test_model_with_hypergraph(model, stock_data, incidence_matrix):
    time_steps = stock_data.shape[0] - 1
    current_features = stock_data[:-1].values.reshape(1, time_steps, stock_data.shape[1])  # Shape: [1, time_steps, num_tickers]
    next_day_returns = calculate_returns(stock_data).values.reshape(1, time_steps, stock_data.shape[1])  # Shape: [1, time_steps, num_tickers]

    # Make predictions using the trained model
    predictions = model(current_features, incidence_matrix)

    # Calculate the RMSE
    mse_loss = tf.keras.losses.MeanSquaredError()
    test_mse = mse_loss(next_day_returns, predictions)

    # Root Mean Squared Error
    test_rmse = tf.sqrt(test_mse)

    print(f"Test RMSE: {test_rmse.numpy()}")


# Main function to test the model
def main():
    # Define tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOG']
    start_date = '2022-01-01'
    end_date = '2022-06-01'

    # Fetch stock data from yfinance
    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Generate hypergraph incidence matrix based on correlation
    num_clusters = 3  # Define how many hyperedges you want to generate
    H, _ = create_hyperedges_from_correlation(stock_data, num_clusters)

    # Define the LSTM-Hypergraph model
    lstm_units = 64
    num_tickers = len(tickers)
    model = LSTMHypergraphModel(lstm_units=lstm_units, num_clusters=num_clusters, num_tickers=num_tickers)

    # Test the model
    test_model_with_hypergraph(model, stock_data, H)

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
    start_date = '2022-01-01'
    end_date = '2022-06-01'

    # Fetch stock data
    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Generate hypergraph incidence matrix
    num_clusters = 3
    H, _ = create_hyperedges_from_correlation(stock_data, num_clusters)

    # Define and initialize model
    lstm_units = 64
    num_tickers = len(tickers)
    model = LSTMHypergraphModel(lstm_units=lstm_units, num_clusters=num_clusters, num_tickers=num_tickers)

    # Get predictions for each time step
    predicted_prices = []
    for i in range(1, stock_data.shape[0]):
        current_features = stock_data.iloc[:i].values.reshape(1, i, stock_data.shape[1])
        pred = model(current_features, H)
        predicted_prices.append(pred.numpy().flatten().tolist())

    # Format results
    dates = stock_data.index[1:].strftime('%Y-%m-%d').tolist()  # Skip first day due to returns calculation
    actual_prices = stock_data.iloc[1:].values.tolist()  # Skip first day

    return {
        'dates': dates,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices
    }

if __name__ == "__main__":
    try:
        result = run_predictions()
        try:
            with open("all_model_results.json", "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
        all_results["hypergraph_lstm"] = result
        with open("all_model_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("Saved hypergraph_lstm results to all_model_results.json")
    except Exception as e:
        print(f"hypergraph_lstm failed: {e}")

