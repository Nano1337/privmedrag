import pandas as pd
import json
import os

# Path to the JSON file
json_file_path = "./data/medical_mcqs.json"

# Path for the output Parquet file
parquet_file_path = "./data/medical_mcqs.parquet"

# Read the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Extract the MCQs
mcqs = data.get('mcqs', [])

# Create a list to store the selected data
selected_data = []

# Extract only the required columns
for mcq in mcqs:
    selected_data.append({
        'question': mcq.get('question', ''),
        'options': mcq.get('options', []),
        'correct_index': mcq.get('correct_index', 0),
        'patient_id': mcq.get('patient_id', ''),
        'question_type': mcq.get('question_type', '')
    })

# Convert to DataFrame
df = pd.DataFrame(selected_data)

# Save to Parquet format
df.to_parquet(parquet_file_path, index=False)

print(f"Conversion completed. Parquet file saved to: {parquet_file_path}")
print(f"DataFrame shape: {df.shape}")
print("Column names:", df.columns.tolist())