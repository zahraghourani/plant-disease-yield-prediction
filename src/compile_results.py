import os
import re
import pandas as pd

results_dir = "results"
summary_files = [f for f in os.listdir(results_dir) if f.endswith('_summary.txt')]

data = []
for fname in summary_files:
    model_name = fname.replace('_summary.txt', '')
    path = os.path.join(results_dir, fname)
    with open(path, 'r') as f:
        content = f.read()
    # Look for "Accuracy: 0.xxxx" (from the final evaluation block)
    match = re.search(r'Accuracy: (\d+\.\d+)', content)
    if match:
        accuracy = float(match.group(1))
    else:
        accuracy = None
    # Optionally, also extract other metrics if needed
    # E.g., precision, recall, f1, training time, etc. (might be printed as well)
    data.append({'model_name': model_name, 'accuracy': accuracy})

df = pd.DataFrame(data)
df = df.sort_values('accuracy', ascending=False)
df.to_csv('all_models_results.csv', index=False)
print("Saved to all_models_results.csv")
print(df)