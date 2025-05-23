import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction Model Comparison",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Price Prediction Model Comparison Dashboard")
st.markdown("""
This dashboard compares the performance of different deep learning models for stock price prediction:
- CNN
- Hypergraph GRU
- Hypergraph LSTM
- LSTM
- Simple Hypergraph Model
""")

# Sample data structure (replace with actual data loading)
def load_model_results():
    # This is a placeholder - replace with actual data loading logic
    models = ['CNN', 'Hypergraph GRU', 'Hypergraph LSTM', 'LSTM', 'Simple Hypergraph']
    metrics = {
        'MSE': [0.0023, 0.0019, 0.0018, 0.0021, 0.0024],
        'RMSE': [0.048, 0.044, 0.042, 0.046, 0.049],
        'MAE': [0.035, 0.032, 0.031, 0.034, 0.036]
    }
    return pd.DataFrame(metrics, index=models)

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Model Performance Metrics", "Prediction Comparison", "Error Analysis"])

with tab1:
    st.header("Model Performance Metrics")
    
    # Load and display metrics
    metrics_df = load_model_results()
    
    # Create bar charts for each metric
    for metric in metrics_df.columns:
        fig = px.bar(
            metrics_df,
            y=metric,
            title=f"{metric} Comparison Across Models",
            labels={'index': 'Model', 'value': metric},
            color=metrics_df[metric],
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Prediction Comparison")
    
    # Placeholder for actual vs predicted values
    # Replace with actual data loading
    dates = pd.date_range(start='2023-01-01', periods=100)
    actual_prices = np.random.normal(100, 5, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual_prices, name='Actual Prices', line=dict(color='black')))
    
    # Add predictions for each model (replace with actual data)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, model in enumerate(['CNN', 'Hypergraph GRU', 'Hypergraph LSTM', 'LSTM', 'Simple Hypergraph']):
        predicted_prices = actual_prices + np.random.normal(0, 2, 100)
        fig.add_trace(go.Scatter(x=dates, y=predicted_prices, name=f'{model} Predictions', line=dict(color=colors[i])))
    
    fig.update_layout(
        title='Actual vs Predicted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Error Analysis")
    
    # Create error distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Error distribution histogram
        errors = np.random.normal(0, 1, 1000)  # Replace with actual error data
        fig = px.histogram(
            errors,
            title='Error Distribution',
            labels={'value': 'Prediction Error', 'count': 'Frequency'},
            nbins=50
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Error box plot by model
        error_data = pd.DataFrame({
            'Model': np.repeat(['CNN', 'Hypergraph GRU', 'Hypergraph LSTM', 'LSTM', 'Simple Hypergraph'], 200),
            'Error': np.concatenate([np.random.normal(0, i, 200) for i in [1, 0.8, 0.7, 0.9, 1.1]])
        })
        fig = px.box(
            error_data,
            x='Model',
            y='Error',
            title='Error Distribution by Model'
        )
        st.plotly_chart(fig, use_container_width=True)

# Add a sidebar with model selection
st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect(
    "Select models to display",
    ['CNN', 'Hypergraph GRU', 'Hypergraph LSTM', 'LSTM', 'Simple Hypergraph'],
    default=['CNN', 'Hypergraph GRU', 'Hypergraph LSTM', 'LSTM', 'Simple Hypergraph']
)

# Add a sidebar with date range selection
st.sidebar.header("Date Range")
start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2023, 12, 31))

# Add a footer
st.markdown("---")
st.markdown("Dashboard created for stock price prediction model comparison") 