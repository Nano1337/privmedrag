from rgl.data.dataset import DownloadableRGLDataset
import dgl
import torch
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.functional import normalize

class PrimeKGDataset(DownloadableRGLDataset):
    def __init__(self, dataset_root_path=None):
        """
        PrimeKG Dataset for GraphRAG
        
        Parameters:
        -----------
        dataset_root_path : str
            Path to the dataset root directory
        """
        # If you're manually downloading the dataset, you can set these to empty
        download_urls = []
        download_file_names = []
        
        super().__init__(
            dataset_name="primekg",
            download_urls=download_urls,
            download_file_names=download_file_names,
            cache_name="cache.p",
            dataset_root_path=dataset_root_path,
        )
    
    def download_graph(self, dataset_name, graph_root_path):
        """Load PrimeKG into a DGL graph"""
        # Define paths to PrimeKG data files
        nodes_path = os.path.join(self.raw_root_path, "nodes.csv")
        edges_path = os.path.join(self.raw_root_path, "edges.csv")
        
        print(f"Loading nodes from {nodes_path}")
        print(f"Loading edges from {edges_path}")
        
        # Load node data
        nodes_df = pd.read_csv(nodes_path)
        
        # From the data inspection, we know the columns are:
        # node_index, node_id, node_type, node_name, node_source
        print(f"Loaded {len(nodes_df)} nodes")
        
        # Map node_type to category for consistency with our framework
        nodes_df['category'] = nodes_df['node_type']
        
        # Add description column with empty strings for now
        # In a real application, you could fetch descriptions from external sources
        nodes_df['description'] = [f"A {node_type} entity" for node_type in nodes_df['node_type']]
        
        # Map node_name to name for consistency
        nodes_df['name'] = nodes_df['node_name']
        
        # Load edge data
        edges_df = pd.read_csv(edges_path)
        print(f"Loaded {len(edges_df)} edges")
        
        # From the data inspection, we know the columns are:
        # relation, display_relation, x_index, y_index
        
        # In this file, edges are defined by node indices, not IDs
        # x_index and y_index are source and target node indices
        src = edges_df['x_index'].values
        dst = edges_df['y_index'].values
        
        # Also save the relation types
        relation_types = edges_df['relation'].values
        
        # Create DGL graph
        self.graph = dgl.graph((src, dst))
        
        # Store edge types as edge data
        # Create a numerical encoding for the relation types
        unique_relations = list(set(relation_types))
        relation_to_id = {rel: i for i, rel in enumerate(unique_relations)}
        edge_type_ids = torch.tensor([relation_to_id[rel] for rel in relation_types])
        self.graph.edata['type'] = edge_type_ids
        print(f"Graph has {len(unique_relations)} distinct relationship types")
        
        # Create initial node features
        self._generate_node_features(nodes_df)
        
        # Store node attributes as raw_ndata
        self.raw_ndata = {
            'name': nodes_df['name'].values,
            'node_id': nodes_df['node_id'].values,
            'category': nodes_df['category'].values,
            'description': nodes_df['description'].fillna('').values,
            'source': nodes_df['node_source'].values
        }
        
        # Store relationship information in raw_edata
        self.raw_edata = {
            'relation_type': relation_types,
            'display_relation': edges_df['display_relation'].values
        }
        
        # Train/val/test splits (80/10/10)
        n_nodes = self.graph.num_nodes()
        indices = np.random.permutation(n_nodes)
        train_size = int(0.8 * n_nodes)
        val_size = int(0.1 * n_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        self.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True
    
    def _generate_node_features(self, nodes_df):
        """Generate initial node features from text descriptions"""
        # Combine name and description for better features
        texts = []
        for i, row in nodes_df.iterrows():
            name = row['name'] if not pd.isna(row['name']) else ""
            desc = row['description'] if not pd.isna(row['description']) else ""
            category = row['category'] if not pd.isna(row['category']) else ""
            text = f"{name} {category} {desc}"
            texts.append(text)
        
        # Use TF-IDF for initial features
        vectorizer = TfidfVectorizer(max_features=128)
        try:
            features = vectorizer.fit_transform(texts).toarray()
        except ValueError:
            # Fallback if TF-IDF fails due to all empty strings
            print("Warning: Using random features as fallback")
            features = np.random.randn(len(texts), 128)
        
        # Convert to PyTorch tensor and normalize
        features = torch.FloatTensor(features)
        self.feat = normalize(features, p=2, dim=1)
        self.graph.ndata['feat'] = self.feat
    
    def process(self):
        """Additional processing (optional)"""
        # Store edge types
        edges_path = os.path.join(self.raw_root_path, "PrimeKG", "relationships.csv")
        if os.path.exists(edges_path):
            edges_df = pd.read_csv(edges_path)
            self.raw_edata = {
                'relation_type': edges_df['relation_type'].values,
                'source': edges_df['source'].values,
                'pmids': edges_df['pmids'].fillna('').values
            }
        
        # Try to load precomputed embeddings if available
        self._load_precomputed_embeddings()
    
    def _load_precomputed_embeddings(self):
        """Load precomputed embeddings if available"""
        embedding_path = os.path.join(self.processed_root_path, "biobert_embeddings.npy")
        if os.path.exists(embedding_path):
            embeddings = np.load(embedding_path)
            self.feat = torch.FloatTensor(embeddings)
            self.graph.ndata['feat'] = self.feat
            print(f"Loaded precomputed embeddings of shape {self.feat.shape}")
            return True
        return False