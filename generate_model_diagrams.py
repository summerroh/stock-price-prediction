# Generate model structures images for all models

import os
import re
from graphviz import Digraph

models = [
    ("CNN_new", "CNN_new.py"),
    ("LSTM", "LSTM.py"),
    ("simple_hypergraph_model", "simple_hypergraph_model.py"),
    ("hypergraph_lstm", "hypergraph_lstm.py")
]

output_dir = "model_structures"
os.makedirs(output_dir, exist_ok=True)

layer_regex = re.compile(r"(Dense|Conv2D|LSTM|GRU|Flatten|Dropout|MaxPooling2D|InputLayer|Activation|BatchNormalization|Embedding|Add|Concatenate|AveragePooling2D|GlobalAveragePooling2D|Reshape|RepeatVector|TimeDistributed|Bidirectional)\s*\(")

for model_name, file_name in models:
    if not os.path.exists(file_name):
        print(f"File {file_name} not found.")
        continue
    with open(file_name, 'r') as f:
        code = f.readlines()
    layers = []
    for line in code:
        match = layer_regex.search(line)
        if match:
            layer_type = match.group(1)
            # Optionally, extract parameters
            param_str = line.strip().split('(', 1)[-1].rsplit(')', 1)[0]
            layers.append(f"{layer_type}({param_str})")
    if not layers:
        print(f"No layers found in {file_name}")
        continue
    dot = Digraph(comment=f"{model_name} Structure")
    for i, layer in enumerate(layers):
        dot.node(str(i), layer)
        if i > 0:
            dot.edge(str(i-1), str(i))
    out_path = os.path.join(output_dir, f"{model_name}_structure")
    dot.render(out_path, format="png", cleanup=True)
    print(f"Saved {out_path}.png") 