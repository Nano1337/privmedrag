#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Medical Entity Summarization using PrimeKG Knowledge Graph
=========================================================

This script demonstrates how to use the PrimeKG knowledge graph to generate comprehensive 
medical summaries for entities such as diseases, drugs, genes, and more. It uses graph traversal 
to find related entities and their relationships, and then generates a summary using a language model.

Key features:

1. Knowledge Graph Traversal: Finds related entities by traversing the graph structure rather than 
   using vector similarity, ensuring relationships are captured accurately.

2. Relationship Prioritization: Emphasizes medically relevant relationships like drug indications 
   and contraindications, enhancing the medical context in the summary.

3. Category Filtering and Balancing: Can filter for specific entity types (e.g., drugs, genes) and 
   balance the selection across different categories for comprehensive context.

4. LLM-powered Summarization: Uses GPT models to generate coherent medical summaries based on the 
   graph-derived relationships.

Usage Examples:
--------------

# Basic usage - summarize a random disease entity
python demo_primekg_entity_summary.py --dataset_path ../dataset/primekg --entity_type disease

# Find relationships between a disease and drugs, prioritizing medical relationships
python demo_primekg_entity_summary.py --dataset_path ../dataset/primekg --entity_type disease \
    --filter_category drug --use_priority --n_hops 2

# Explore a specific entity by ID with balanced category selection
python demo_primekg_entity_summary.py --dataset_path ../dataset/primekg --entity_id 27158 \
    --balance_categories --max_neighbors 20

# Generate in-depth summary of a rare disease with extensive traversal
python demo_primekg_entity_summary.py --dataset_path ../dataset/primekg --entity_type disease \
    --random_entity --n_hops 3 --max_neighbors 30 --verbose
"""

import sys
import os
import random
import argparse
import torch
import pandas as pd
import numpy as np
import openai
import csv
from typing import List, Dict, Any, Tuple, Set
from rgl.datasets.primekg import PrimeKGDataset
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import steiner_batch_retrieve
from rgl.utils import llm_utils
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main(args):
    # Initialize OpenAI API key from environment
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai.api_key and not args.no_summary:
        print("Warning: No OpenAI API key found in environment. Make sure OPENAI_API_KEY is set in your .env file.")
    model = args.model
    
    print(f"Loading PrimeKG dataset from {args.dataset_path}...")
    dataset = PrimeKGDataset(args.dataset_path)
    names = dataset.raw_ndata["name"]
    descriptions = dataset.raw_ndata["description"]
    categories = dataset.raw_ndata["category"]
    src = dataset.graph.edges()[0].numpy().tolist()
    dst = dataset.graph.edges()[1].numpy().tolist()
    
    # Get all available categories
    unique_categories = list(set(categories))
    print(f"Available entity categories: {sorted(set([cat.split('/')[0] for cat in unique_categories]))}")
    
    # Function to select a query node
    def select_query_node(category=args.entity_type, entity_id=args.entity_id, name_filter=args.entity_name_contains):
        # If a specific entity ID is provided, use it
        if entity_id >= 0:
            if entity_id < len(names):
                print(f"Using specified entity ID: {entity_id}")
                return entity_id
            else:
                print(f"Warning: Specified entity ID {entity_id} is out of range. Falling back to category selection.")
        
        # Match category using a prefix match since PrimeKG has composite categories (e.g., gene/protein)
        mask = [cat.startswith(category) for cat in categories]
        valid_indices = [i for i, m in enumerate(mask) if m]
        
        # Apply name filter if specified
        if name_filter and valid_indices:
            name_filter = name_filter.lower()
            valid_indices = [i for i in valid_indices if name_filter in names[i].lower()]
            if valid_indices:
                print(f"Found {len(valid_indices)} entities matching name filter '{name_filter}'")
            else:
                print(f"Warning: No entities found matching name filter '{name_filter}'. Ignoring filter.")
                # Reset to all category matches if no name matches found
                mask = [cat.startswith(category) for cat in categories]
                valid_indices = [i for i, m in enumerate(mask) if m]
        if not valid_indices:
            print(f"No entities found for category: {category}")
            print(f"Available categories: {sorted(set([cat.split('/')[0] for cat in set(categories)]))}")
            return None
        # Use either a random entity or the first one
        if args.random_entity:
            return random.choice(valid_indices)
        return valid_indices[0]
    
    # Select a query node
    query_node_idx = select_query_node()
    if query_node_idx is None:
        return
        
    entity_type = categories[query_node_idx].split('/')[0] if '/' in categories[query_node_idx] else categories[query_node_idx]
    query_name = names[query_node_idx]  # Define query_name here for use throughout the function
    print(f"Selected entity: {query_name} (ID: {query_node_idx}, Type: {entity_type})")
    
    print("Traversing knowledge graph to find related entities...")
    
    # Find n-hop neighbors of the query node
    def get_n_hop_neighbors(graph, node_idx, n_hops=1, max_neighbors=20):
        """Get nodes within n hops of the query node"""
        visited = set([node_idx])
        frontier = set([node_idx])
        neighbors = set()
        
        # Store edges and their types for each discovered node
        edge_info = {}
        
        # Get relationship information from edges file directly instead of using graph edge types
        # This is safer for PrimeKG which has many edges
        relation_df = None
        edges_path = os.path.join(args.dataset_path, "raw", "edges.csv")
        if os.path.exists(edges_path):
            try:
                relation_df = pd.read_csv(edges_path)
                print(f"Loaded relations from {edges_path}")
            except Exception as e:
                print(f"Warning: Could not load relation data: {e}")
        
        for hop in range(n_hops):
            new_frontier = set()
            for node in frontier:
                try:
                    # Get all outgoing edges
                    out_edges = graph.out_edges(node)
                    dst_nodes = out_edges[1].tolist() if isinstance(out_edges, tuple) else []
                    
                    # Get all incoming edges
                    in_edges = graph.in_edges(node)
                    src_nodes = in_edges[0].tolist() if isinstance(in_edges, tuple) else []
                    
                    # Process outgoing neighbors
                    for dst in dst_nodes:
                        if dst not in edge_info:
                            edge_info[dst] = []
                        
                        # Get relation type from dataframe if available
                        rel_type = "related_to"
                        display_rel = "related to"
                        
                        if relation_df is not None:
                            try:
                                # Find the relation where x_index is node and y_index is dst
                                relation_row = relation_df[(relation_df['x_index'] == node) & 
                                                           (relation_df['y_index'] == dst)]
                                if not relation_row.empty:
                                    rel_type = relation_row['relation'].values[0]
                                    display_rel = relation_row['display_relation'].values[0]
                            except Exception:
                                pass  # Use defaults if lookup fails
                        
                        edge_info[dst].append((node, rel_type, display_rel))
                    
                    # Process incoming neighbors
                    for src in src_nodes:
                        if src not in edge_info:
                            edge_info[src] = []
                        
                        # Get relation type from dataframe if available
                        rel_type = "related_to"
                        display_rel = "related to"
                        
                        if relation_df is not None:
                            try:
                                # Find the relation where x_index is src and y_index is node
                                relation_row = relation_df[(relation_df['x_index'] == src) & 
                                                           (relation_df['y_index'] == node)]
                                if not relation_row.empty:
                                    rel_type = relation_row['relation'].values[0]
                                    display_rel = relation_row['display_relation'].values[0]
                            except Exception:
                                pass  # Use defaults if lookup fails
                        
                        edge_info[src].append((node, rel_type, display_rel))
                    
                    # Add all neighbors to the new frontier
                    new_nodes = set(dst_nodes + src_nodes) - visited
                    new_frontier.update(new_nodes)
                    neighbors.update(new_nodes)
                    visited.update(new_nodes)
                
                except Exception as e:
                    print(f"Warning: Error processing node {node}: {e}")
                
                # Limit the number of neighbors
                if len(neighbors) >= max_neighbors:
                    return list(neighbors), edge_info
            
            frontier = new_frontier
            if not frontier:
                break
                
        return list(neighbors), edge_info
    
    # Define important medical relationship types to prioritize when exploring
    PRIORITY_RELATIONS = {
        'disease': ['indication', 'contraindication', 'off-label use', 'disease_disease', 'disease_protein'],
        'drug': ['indication', 'contraindication', 'off-label use', 'drug_effect', 'drug_protein'],
        'gene': ['disease_protein', 'drug_protein', 'protein_protein']
    }
    
    # Get priority relationships based on entity type
    priority_rels = PRIORITY_RELATIONS.get(entity_type.lower(), [])
    print(f"Using priority relationships for {entity_type}: {priority_rels}")
    
    # Get directly connected entities (n-hop neighbors)
    if args.verbose:
        print(f"Traversing {args.n_hops} hops from entity {query_name} (ID: {query_node_idx})...")
    
    direct_neighbors, edge_info = get_n_hop_neighbors(
        dataset.graph, query_node_idx, n_hops=args.n_hops, max_neighbors=100  # Get more neighbors than we need for filtering
    )
    print(f"Found {len(direct_neighbors)} connected entities within {args.n_hops} hops")
    
    # Filter neighbors by category and relationship priority if specified
    if args.filter_category or args.use_priority:
        try:
            filtered_neighbors = []
            scored_neighbors = []
            
            # Track categories for balance
            category_counts = {}
            
            for n in direct_neighbors:
                # Skip if this is the query node
                if n == query_node_idx:
                    continue
                    
                # Get category for filtering
                cat = categories[n] if n < len(categories) else ""
                main_cat = cat.split('/')[0] if '/' in cat else cat
                
                # Store for category balance
                if main_cat not in category_counts:
                    category_counts[main_cat] = 0
                
                # Score based on relationship and category match
                score = 0
                rel_types_found = set()
                
                # Category match (if filter specified)
                category_match = False
                if not args.filter_category or args.filter_category.lower() in cat.lower():
                    category_match = True
                    score += 5  # Base score for category match
                
                # Relationship priority
                if n in edge_info and args.use_priority:
                    for _, rel_type, _ in edge_info[n]:
                        rel_types_found.add(rel_type)
                        if rel_type in priority_rels:
                            # Highly prioritize therapeutic relationships
                            if rel_type in ['indication', 'contraindication', 'off-label use']:
                                score += 10
                            # Disease-gene relationships also important
                            elif rel_type == 'disease_protein':
                                score += 8
                            else:
                                score += 3
                                
                # Bonus for specific entity-to-entity relationships
                node_source = dataset.raw_ndata["source"][n] if "source" in dataset.raw_ndata else ""
                if main_cat == 'drug' and 'DRUGBANK' in node_source:
                    score += 4  # Bonus for drugs from DRUGBANK
                if main_cat == 'gene' and node_source in ["NCBI", "UniProt"]:
                    score += 2  # Bonus for genes from reliable sources
                
                # Ensure diversity by slightly penalizing over-represented categories
                if category_counts[main_cat] > args.max_per_category:
                    score -= category_counts[main_cat] - args.max_per_category
                
                # Apply minimum score threshold if specified
                if score < args.min_relationship_score:
                    if args.verbose:
                        print(f"Skipping {names[n]} ({cat}) - Score {score} below threshold {args.min_relationship_score}")
                
                # Keep if it matches category filter (or no filter)
                # or if it has a priority relationship (when use_priority is enabled)
                if category_match or (args.use_priority and score > 0):
                    scored_neighbors.append((n, score, cat, list(rel_types_found)))
                    category_counts[main_cat] += 1
            
            # Sort by score (highest first)
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Group by category for display
            neighbors_by_category = {}
            for n, score, cat, rel_types in scored_neighbors:
                main_cat = cat.split('/')[0] if '/' in cat else cat
                if main_cat not in neighbors_by_category:
                    neighbors_by_category[main_cat] = []
                neighbors_by_category[main_cat].append((n, score, cat, rel_types))
            
            if args.verbose:
                print(f"\nFound {len(scored_neighbors)} entities with scores:")
                for n, score, cat, rel_types in scored_neighbors[:10]:
                    print(f"  - {names[n]} ({cat}): Score {score}, Relations: {', '.join(rel_types[:3])}")
                if len(scored_neighbors) > 10:
                    print(f"  ... and {len(scored_neighbors)-10} more entities")
            
            # Get top K from each category for balanced selection
            balanced_selection = []
            for cat, items in neighbors_by_category.items():
                # Sort each category internally by score
                items.sort(key=lambda x: x[1], reverse=True)
                # Take top K (or fewer if not enough) from each category
                balanced_selection.extend(items[:args.max_per_category])
            
            # Resort the balanced selection by score
            balanced_selection.sort(key=lambda x: x[1], reverse=True)
            
            # Take final selection (either balanced or top-scored based on flag)
            if args.balance_categories and balanced_selection:
                final_selection = balanced_selection[:args.max_neighbors]
            else:
                final_selection = scored_neighbors[:args.max_neighbors]
                
            # Extract node IDs for final selection
            filtered_neighbors = [n for n, _, _, _ in final_selection]
            
            if filtered_neighbors:
                if args.filter_category:
                    print(f"Found entities matching filter '{args.filter_category}'")
                if args.use_priority:
                    print(f"Prioritized by medical relationships: {priority_rels}")
                    
                # Print categories found with counts
                found_categories = {}
                for _, _, cat, _ in final_selection:
                    main_cat = cat.split('/')[0] if '/' in cat else cat
                    if main_cat not in found_categories:
                        found_categories[main_cat] = 0
                    found_categories[main_cat] += 1
                    
                print("Entity types selected:")
                for cat, count in found_categories.items():
                    print(f"  - {cat}: {count}")
                    
                # Print top entities from each category
                print("\nTop entities by category:")
                for cat_name in sorted(neighbors_by_category.keys()):
                    cat_entities = neighbors_by_category[cat_name][:2]  # Top 2 per category
                    if cat_entities:
                        print(f"  {cat_name.upper()}:")
                        for i, (n, score, _, rel_types) in enumerate(cat_entities):
                            rel_str = ", ".join(rel_types[:2])
                            if len(rel_types) > 2:
                                rel_str += f" +{len(rel_types)-2} more"
                            print(f"    {i+1}. {names[n]} - Score: {score} - Relations: {rel_str}")
                
                direct_neighbors = filtered_neighbors
            else:
                print(f"Warning: No entities found matching criteria. Showing all related entities.")
        except Exception as e:
            print(f"Warning: Error filtering entities: {e}. Showing all related entities.")
    
    # Make sure we have neighbors, otherwise print a warning
    if not direct_neighbors:
        print("Warning: No related entities found for this entity. The knowledge graph may not have connections for it.")
        direct_neighbors = []  # Ensure it's empty but defined
    
    # Combine the query node with its neighbors
    subg_nodes = [query_node_idx] + direct_neighbors[:args.max_neighbors]
    
    print(f"Retrieved {len(subg_nodes)} nodes in the subgraph")
    
    # Prepare prompt
    query_description = descriptions[query_node_idx]
    query_category = categories[query_node_idx]
    
    # GROUP ENTITIES BY CATEGORY FOR BETTER PROMPT ORGANIZATION
    entities_by_category = {}
    for node in subg_nodes:
        if node == query_node_idx:
            continue
        cat = categories[node]
        main_cat = cat.split('/')[0] if '/' in cat else cat
        if main_cat not in entities_by_category:
            entities_by_category[main_cat] = []
        entities_by_category[main_cat].append(node)

    # CREATE THE PROMPT WITH ORGANIZED CATEGORIES
    prompt = f"""
    You are a medical expert. Given information about a primary medical entity and related entities from a knowledge graph, generate a comprehensive medical summary.

    PRIMARY ENTITY:
    Name: {query_name}
    Category: {query_category}
    Description: {query_description}

    RELATED ENTITIES FROM KNOWLEDGE GRAPH (GROUPED BY TYPE):
    """
        
    # ADD ENTITIES BY CATEGORY
    for category_name, category_nodes in entities_by_category.items():
        prompt += f"\n\n{category_name.upper()} ENTITIES:"
        
        for node in category_nodes:
            related_name = names[node]
            related_description = descriptions[node]
            related_category = categories[node]
            
            # Add relationship information
            relationships = []
            if node in edge_info:
                for src_dst, rel_type, display_rel in edge_info[node]:
                    if src_dst == query_node_idx:
                        direction = "Primary entity → This entity"
                    else:
                        direction = "This entity → Primary entity"
                    relationships.append(f"{rel_type} ({display_rel}): {direction}")
            
            relationship_text = "\n".join(relationships) if relationships else "Unknown relationship"
            
            prompt += f"\n\nName: {related_name}\nCategory: {related_category}\nDescription: {related_description}\nRelationships: {relationship_text}"
    
    prompt += f"""

Based on the information above, please generate a comprehensive medical summary about {query_name}. Your summary should:

1. Explain what {query_name} is, including its classification and general characteristics
2. Describe its etiology, pathophysiology, and molecular mechanisms (if genes/proteins are mentioned)
3. Discuss any treatments, including approved drugs and off-label medications
4. Explain relationships with other diseases or conditions
5. Include practical implications for patients

Organize your response with appropriate headings. Synthesize information from the related entities to create a coherent explanation. Only include scientifically accurate information that can be inferred from the provided data.
"""
    
    # Generate summary using LLM if not disabled
    if not args.no_summary:
        print("\nGenerating summary using LLM...")
        
        try:
            # Generate summary using OpenAI through llm_utils
            sys_msg = "You are a medical expert providing accurate information."
            generated_summary = llm_utils.chat_openai(
                prompt, 
                model=model, 
                sys_prompt=sys_msg
            )
            
            # Save to file if specified
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(f"Summary for {query_name} (ID: {query_node_idx})\n\n")
                    f.write(generated_summary)
                print(f"Summary saved to {args.output_file}")
                
        except Exception as e:
            print(f"Error generating summary: {e}")
            generated_summary = "Error generating summary. Please check that your OPENAI_API_KEY is set in the .env file."
    else:
        generated_summary = "[Summary generation skipped]"
    
    # Print results
    print("\n" + "="*80)
    print("PRIMARY ENTITY")
    print("="*80)
    print(f"Name: {query_name}")
    print(f"Category: {query_category}")
    print(f"Description: {query_description}")
    
    print("\n" + "="*80)
    print("RELATED ENTITIES")
    print("="*80)
    # Print entities grouped by category
    for category, nodes in entities_by_category.items():
        print(f"- {category.upper()} ({len(nodes)})")
        for node in nodes[:3]:  # Show first 3 entities per category
            # Show relationship information if available
            relationship_info = ""
            if node in edge_info:
                relationships = []
                for src_dst, rel_type, display_rel in edge_info[node]:
                    relationships.append(f"{rel_type}")
                if relationships:
                    relationship_info = f" - Related via: {', '.join(relationships[:2])}"
                    if len(relationships) > 2:
                        relationship_info += f" and {len(relationships)-2} more"
            
            print(f"  • {names[node]} ({categories[node]}){relationship_info}")
        if len(nodes) > 3:
            print(f"    ...and {len(nodes)-3} more {category} entities")
    
    if len(subg_nodes) > 6:
        print(f"...and {len(subg_nodes)-6} more entities")
        
    print("\n" + "="*80)
    print("GENERATED SUMMARY")
    print("="*80)
    print(generated_summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Entity Summarization using PrimeKG Knowledge Graph")
    
    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument("--dataset_path", type=str, default="../dataset/primekg", 
                          help="Path to the PrimeKG dataset directory containing raw/ and processed/ subdirectories")
    
    # Entity selection options
    entity_group = parser.add_argument_group('Entity Selection')
    entity_group.add_argument("--entity_type", type=str, default="disease", 
                            help="Type of entity to summarize (e.g., 'disease', 'drug', 'gene')")
    entity_group.add_argument("--entity_id", type=int, default=-1, 
                            help="Specific entity ID to use (overrides --entity_type if specified)")
    entity_group.add_argument("--random_entity", action="store_true", 
                            help="Select a random entity of the specified --entity_type")
    entity_group.add_argument("--entity_name_contains", type=str, default="", 
                            help="Filter entity selection to those containing this substring (case-insensitive)")
    
    # Graph traversal options
    traversal_group = parser.add_argument_group('Graph Traversal')
    traversal_group.add_argument("--n_hops", type=int, default=1, 
                               help="Number of hops for graph traversal (1-3 recommended, higher values may be slow)")
    traversal_group.add_argument("--max_neighbors", type=int, default=10, 
                               help="Maximum number of neighbors to retrieve in total")
    traversal_group.add_argument("--max_per_category", type=int, default=5, 
                               help="Maximum number of entities per category when using --balance_categories")
    traversal_group.add_argument("--filter_category", type=str, default="", 
                               help="Filter neighbors by category (e.g., 'drug', 'gene', 'disease')")
    traversal_group.add_argument("--use_priority", action="store_true", 
                               help="Use priority relationships for medical context (highly recommended)")
    traversal_group.add_argument("--balance_categories", action="store_true", 
                               help="Balance entity selection across categories for more diverse context")
    traversal_group.add_argument("--min_relationship_score", type=int, default=0, 
                               help="Minimum relationship score threshold for including an entity (0-25)")
    
    # Output and LLM options
    output_group = parser.add_argument_group('Output and LLM Configuration')
    output_group.add_argument("--model", type=str, default="gpt-4o", 
                            help="OpenAI model to use for summary generation")
    output_group.add_argument("--verbose", action="store_true", 
                            help="Print additional debugging information and traversal details")
    output_group.add_argument("--output_file", type=str, default="", 
                            help="Optional file to save the generated summary")
    output_group.add_argument("--no_summary", action="store_true", 
                            help="Skip summary generation, only show graph traversal results")
    # Remove the OpenAI API key argument - using dotenv only
    
    args = parser.parse_args()
    main(args)
