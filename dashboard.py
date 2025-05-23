import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
# Reorder model names for dropdown
preferred_order = ["CNN_new", "LSTM", "simple_hypergraph_model", "hypergraph_lstm"]
ordered_model_names = [m for m in preferred_order if m in model_names] + [m for m in model_names if m not in preferred_order]
model = st.sidebar.selectbox("Select Model", ordered_model_names)

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

# Show model structure image if available
image_path = os.path.join("model_structures", f"{model}_structure.png")
if os.path.exists(image_path):
    st.image(image_path, caption=f"{model} Structure", use_column_width=True)
else:
    st.info("Model structure image not available.")

# Calculate metrics for each model
comparison_data = []
for m in ordered_model_names:
    m_data = results[m]
    actual = np.array(m_data["actual_prices"])[:, ticker_idx]
    predicted = np.array(m_data["predicted_prices"])[:, ticker_idx]
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    comparison_data.append({
        "Model": m,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    })

comp_df = pd.DataFrame(comparison_data)

# Melt the DataFrame for grouped bar chart
comp_df_melted = comp_df.melt(id_vars=["Model"], value_vars=["MSE", "RMSE", "MAE"], var_name="Metric", value_name="Value")

fig = px.bar(comp_df_melted, x="Model", y="Value", color="Metric", barmode="group", title="Model Comparison Metrics (Selected Ticker)")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(comp_df, use_container_width=True)

# Add a footer
st.markdown("---")
st.markdown("Dashboard created for stock price prediction model comparison") 