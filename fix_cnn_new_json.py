import json

with open("all_model_results.json", "r") as f:
    data = json.load(f)

cnn = data["CNN_new"]
dates = cnn["dates"]
actual = cnn["actual_prices"]
pred = cnn["predicted_prices"]

# Find valid rows
valid_rows = [
    i for i, (a, p) in enumerate(zip(actual, pred))
    if len(a) == 3 and len(p) == 3
]

# Filter all arrays to only valid rows
cnn["dates"] = [dates[i] for i in valid_rows]
cnn["actual_prices"] = [actual[i] for i in valid_rows]
cnn["predicted_prices"] = [pred[i] for i in valid_rows]

# Save the fixed file
with open("all_model_results_FIXED.json", "w") as f:
    json.dump(data, f, indent=2) 