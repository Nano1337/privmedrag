from datasets import Dataset
from huggingface_hub import login

# Path to the Parquet file
parquet_file_path = "./data/medical_mcqs.parquet"

# Load the dataset from the Parquet file
dataset = Dataset.from_parquet(parquet_file_path, columns=['question', 'options', 'correct_index', 'patient_id', 'question_type'])

# Print dataset info
print(f"Dataset loaded with {len(dataset)} examples")
print(f"Features: {dataset.features}")

# Set your Hugging Face repository name
repo_id = "Nano1337/medical-mcqs"

# Login to Hugging Face Hub
login()

# Create the repository and push the dataset
dataset.push_to_hub(
    repo_id,
    private=False,  # Set to True if you want a private repository
    commit_message="Upload medical MCQs dataset"
)

print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}")