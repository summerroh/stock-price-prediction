#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import yfinance as yf

# Set seeds for reproducibility
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Function to fetch stock data from yfinance for multiple tickers
def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.fillna(method='ffill')  # Handle missing data
    return data

# Build a simplified model for multiple tickers
class SimpleModel(tf.keras.Model):
    def __init__(self, hidden_units=64, output_units=3):  # Adjust output_units to 3 for three tickers
        super(SimpleModel, self).__init__()
        self.hidden_units = hidden_units

        # Dense layers
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense_out = tf.keras.layers.Dense(output_units, activation='linear')  # Predicting future stock prices

    def call(self, features):
        # First dense layer
        x = self.dense1(features)  # Shape: [batch_size, hidden_units]

        # Second dense layer
        x = self.dense2(x)

        # Final output layer
        out = self.dense_out(x)  # Output shape: [batch_size, output_units]
        return out

# Training function
def train_model(model, stock_data, epochs=50, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    mse_loss = tf.keras.losses.MeanSquaredError()

    # Prepare input and target data
    current_features = stock_data[:-1].values  # Features: previous day prices
    next_day_prices = stock_data[1:].values  # Target: next day's prices
    
    # Convert to tensors
    current_features = tf.convert_to_tensor(current_features, dtype=tf.float32)
    next_day_prices = tf.convert_to_tensor(next_day_prices, dtype=tf.float32)

    # Training loop
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Forward pass through the model
            predictions = model(current_features)

            # Calculate the loss (MSE between predicted and actual future prices)
            loss = mse_loss(next_day_prices, predictions)

        # Compute gradients and apply updates
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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
    start_date = '2022-01-04' 
    end_date = '2022-06-01'

    # Fetch stock data
    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Define and compile the model
    model = SimpleModel(output_units=len(tickers))

    # Train the model
    train_model(model, stock_data, epochs=10, learning_rate=0.001)

    # Get predictions for each time step
    predicted_prices = []
    for i in range(1, len(stock_data)):
        current_features = stock_data.iloc[:i].values
        current_features = tf.convert_to_tensor(current_features, dtype=tf.float32)
        pred = model(current_features)
        predicted_prices.append(pred.numpy()[-1].tolist())  # Get the last prediction

    # Format results
    dates = stock_data.index[1:].strftime('%Y-%m-%d').tolist()  # Skip first day
    actual_prices = stock_data.iloc[1:].values.tolist()  # Skip first day

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
        all_results["simple_hypergraph_model"] = result
        with open("all_model_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("Saved simple_hypergraph_model results to all_model_results.json")
    except Exception as e:
        print(f"simple_hypergraph_model failed: {e}")

