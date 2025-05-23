import json
import importlib
import sys
import traceback
import os

model_files = [
    ('CNN_new', 'CNN_new.py'),
    ('hypergraph_lstm', 'hypergraph_lstm.py'),
    ('LSTM', 'LSTM.py'),
    ('simple_hypergraph_model', 'simple hypergraph model.py'),
]

results = {}

# Load existing results if the file exists
if os.path.exists('all_model_results.json'):
    with open('all_model_results.json', 'r') as f:
        try:
            results = json.load(f)
        except Exception:
            results = {}

for module_name, file_name in model_files:
    if module_name in results:
        print(f"Skipping {module_name}, already has results.")
        continue
    try:
        # Try to import as a module
        mod = importlib.import_module(module_name)
        if hasattr(mod, 'run_predictions'):
            print(f"Running {module_name}.run_predictions()...")
            results[module_name] = mod.run_predictions()
        else:
            print(f"No run_predictions() in {module_name}, skipping.")
    except Exception as e:
        print(f"Could not import {module_name}, trying subprocess fallback. Error: {e}")
        import subprocess
        try:
            output = subprocess.check_output([sys.executable, file_name], stderr=subprocess.STDOUT, text=True)
            # Optionally, parse output if run_predictions prints JSON
            results[module_name] = {'output': output}
        except Exception as sub_e:
            print(f"Subprocess failed for {file_name}: {sub_e}")
            traceback.print_exc()
            results[module_name] = {'error': str(sub_e)}
    # Save after each model
    with open('all_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results for {module_name} to all_model_results.json")

print("All model results saved to all_model_results.json") 