import torch
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def generate_biobert_embeddings(texts, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", batch_size=16):
    """Generate embeddings using BioBERT or similar biomedical models"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            # Use CLS token as the sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        all_embeddings.append(embeddings)
    
    # Concatenate all batches
    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

def main():
    # Load PrimeKG nodes
    nodes_path = "dataset/primekg/raw/nodes.csv"
    
    print(f"Loading nodes from {nodes_path}")
    nodes_df = pd.read_csv(nodes_path)
    
    # Create output directory if it doesn't exist
    os.makedirs("dataset/primekg/processed", exist_ok=True)
    
    # Prepare text data
    print("Preparing text data for embedding generation...")
    texts = []
    for _, row in nodes_df.iterrows():
        name = row['node_name'] if not pd.isna(row['node_name']) else ""
        category = row['node_type'] if not pd.isna(row['node_type']) else ""
        source = row['node_source'] if not pd.isna(row['node_source']) else ""
        # Combine information for richer embeddings
        text = f"{name} is a {category} from {source}"
        texts.append(text)
    
    print(f"Generating embeddings for {len(texts)} nodes...")
    embeddings = generate_biobert_embeddings(texts)
    
    # Save embeddings
    output_path = "dataset/primekg/processed/biobert_embeddings.npy"
    np.save(output_path, embeddings)
    print(f"Saved embeddings of shape {embeddings.shape} to {output_path}")

if __name__ == "__main__":
    main()
