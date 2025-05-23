import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Stock Model Dashboard",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Prediction Model Dashboard")

# Load results
def load_results(json_path):
    if not os.path.exists(json_path):
        st.error(f"File not found: {json_path}")
        return None
    with open(json_path, "r") as f:
        return json.load(f)

results = load_results("all_model_results.json")
if not results:
    st.stop()

model_names = list(results.keys())
model = st.sidebar.selectbox("Select Model", model_names)

model_data = results[model]

dates = model_data["dates"]
actual_prices = np.array(model_data["actual_prices"])
predicted_prices = np.array(model_data["predicted_prices"])

# Handle shape: (n_samples, n_tickers)
ticker_count = actual_prices.shape[1] if actual_prices.ndim > 1 else 1

ticker_names = [f"Ticker {i+1}" for i in range(ticker_count)]
if ticker_count == 3:
    # Try to infer tickers from your scripts
    ticker_names = ['AAPL', 'MSFT', 'GOOG']

ticker_idx = st.sidebar.selectbox("Select Ticker", range(ticker_count), format_func=lambda i: ticker_names[i])

# Prepare DataFrame for plotting
df = pd.DataFrame({
    "Date": pd.to_datetime(dates),
    "Actual": actual_prices[:, ticker_idx],
    "Predicted": predicted_prices[:, ticker_idx]
})

st.subheader(f"Model: {model} | Ticker: {ticker_names[ticker_idx]}")
st.line_chart(df.set_index("Date"), y=["Actual", "Predicted"])

# Show summary stats
rmse = np.sqrt(np.mean((df["Actual"] - df["Predicted"]) ** 2))
st.write(f"**RMSE:** {rmse:.2f}")

st.dataframe(df, use_container_width=True)

# Add a footer
st.markdown("---")
st.markdown("Dashboard created for stock price prediction model comparison") 