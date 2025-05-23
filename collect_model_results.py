import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import importlib.util
import sys
import nbformat
from nbconvert import PythonExporter

def load_model(model_name):
    """Dynamically load the model from Jupyter notebook."""
    # Map model names to notebook files
    notebook_map = {
        'CNN': 'CNN_new.ipynb',
        'Hypergraph GRU': 'hypergraph_ GRU (1).ipynb',
        'Hypergraph LSTM': 'hypergraph_lstm.ipynb',
        'LSTM': 'LSTM.ipynb',
        'Simple Hypergraph': 'simple hypergraph model.ipynb'
    }
    
    notebook_file = notebook_map.get(model_name)
    if not notebook_file or not os.path.exists(notebook_file):
        raise FileNotFoundError(f"Notebook file {notebook_file} not found for model {model_name}")
    
    # Read the notebook
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Convert notebook to Python code
    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(notebook)
    
    # Create a temporary module
    module_name = f"model_{model_name.lower().replace(' ', '_')}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    # Execute the Python code in the module's namespace
    exec(python_code, module.__dict__)
    
    return module

def run_model_predictions(model_name):
    """
    Run predictions using the actual model files.
    Each model handles its own data loading and predictions.
    """
    try:
        # Load the appropriate model module
        model_module = load_model(model_name)
        
        # Get predictions from the model
        # Each model should return a dictionary with dates, actual_prices, and predicted_prices
        results = model_module.run_predictions()
        
        return results
    except Exception as e:
        print(f"Error running {model_name}: {str(e)}")
        return None

def calculate_metrics(actual, predicted):
    """Calculate error metrics for model predictions."""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae)
    }

def collect_results():
    # Define models to test
    models = [
        'CNN',
        'Hypergraph GRU',
        'Hypergraph LSTM',
        'LSTM',
        'Simple Hypergraph'
    ]
    
    # Create results directory if it doesn't exist
    os.makedirs('model_results', exist_ok=True)
    
    # Collect results for each model
    all_results = {}
    
    for model in models:
        print(f"Running {model}...")
        # Get predictions from the model
        predictions = run_model_predictions(model)
        
        if predictions is None:
            print(f"Skipping {model} due to errors")
            continue
            
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(predictions['actual_prices']),
            np.array(predictions['predicted_prices'])
        )
        
        # Store results
        all_results[model] = {
            'predictions': predictions,
            'metrics': metrics
        }
    
    # Save results to JSON file
    with open('model_results/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create a summary CSV file
    summary_data = []
    for model, results in all_results.items():
        metrics = results['metrics']
        summary_data.append({
            'Model': model,
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('model_results/summary.csv', index=False)
    
    print("Results collected and saved successfully!")
    return all_results

if __name__ == "__main__":
    collect_results()