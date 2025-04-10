#!/usr/bin/env python3
"""
Script to evaluate LLM performance on medical MCQs with and without RAG
Uses Hugging Face dataset and PrimeKG graph traversal for enhanced context
"""

import argparse
import json
import os
import sys
import random
import pandas as pd
import numpy as np
import time
from datasets import load_dataset
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Set, Optional

# Import OpenAI for LLM calls
import openai

# Import PrimeKG dataset class for graph traversal
from rgl.datasets.primekg import PrimeKGDataset
from rgl.utils import llm_utils

# Import Gemini API if available
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables for API keys
load_dotenv()


def load_patient_data(patient_id: str, synthea_path: str) -> Dict:
    """
    Load patient data from Synthea parquet file for a specific patient ID
    
    Args:
        patient_id: The patient ID to retrieve data for
        synthea_path: Path to the Synthea parquet file
        
    Returns:
        Dictionary containing patient data or empty dict if not found
    """
    try:
        # Load the entire dataframe - in production, you'd want to optimize this
        # to only load the specific patient data needed
        df = pd.read_parquet(synthea_path)
        
        # Filter for the specific patient
        patient_df = df[df['patient_id'] == patient_id]
        
        if len(patient_df) == 0:
            print(f"Warning: Patient ID {patient_id} not found in Synthea data")
            return {}
            
        # Convert to dictionary
        return patient_df.iloc[0].to_dict()
    except Exception as e:
        print(f"Error loading patient data: {e}")
        return {}

def format_patient_data(patient_data: Dict) -> str:
    """
    Format patient data into a readable string for context
    
    Args:
        patient_data: Dictionary containing patient data
        
    Returns:
        Formatted string with patient information
    """
    if not patient_data or not isinstance(patient_data, dict):
        return "No patient data available."
    
    # Basic demographic information - simplify formatting
    patient_info = [
        "PATIENT ELECTRONIC HEALTH RECORD (EHR):",
        f"Patient ID: {patient_data.get('patient_id', 'Unknown')}",
        f"Gender: {patient_data.get('demographic_GENDER', 'Unknown')}",
        f"Birth Date: {patient_data.get('demographic_BIRTHDATE', 'Unknown')}",
        f"Race: {patient_data.get('demographic_RACE', 'Unknown')}",
        f"Ethnicity: {patient_data.get('demographic_ETHNICITY', 'Unknown')}"
    ]
    
    # Conditions (medical problems)
    conditions = patient_data.get('conditions', [])
    if isinstance(conditions, np.ndarray) and len(conditions) > 0:
        patient_info.append("\nMEDICAL CONDITIONS:")
        for i, condition in enumerate(conditions[:15]):  # Limit to first 15 conditions
            patient_info.append(f"- {condition}")
        if len(conditions) > 15:
            patient_info.append(f"...and {len(conditions) - 15} more conditions")
    
    # Medications
    medications = patient_data.get('medications', [])
    if isinstance(medications, np.ndarray) and len(medications) > 0:
        patient_info.append("\nCURRENT MEDICATIONS:")
        for medication in medications:
            patient_info.append(f"- {medication}")
    
    # Key observations (vital signs, lab results)
    observations = patient_data.get('observations', [])
    if isinstance(observations, np.ndarray) and len(observations) > 0:
        patient_info.append("\nKEY OBSERVATIONS:")
        vital_signs = []
        lab_results = []
        other_observations = []
        
        # Process the first 30 observations (for brevity)
        for obs in observations[:30]:
            if isinstance(obs, dict):
                desc = obs.get('description', '')
                value = obs.get('value', '')
                units = obs.get('units', '')
                
                # Categorize observations
                if any(term in desc.lower() for term in ['blood pressure', 'heart rate', 'respiratory', 'height', 'weight', 'bmi']):
                    vital_signs.append(f"- {desc}: {value} {units if units else ''}")
                elif any(term in desc.lower() for term in ['cholesterol', 'triglycerides', 'hemoglobin', 'glucose', 'a1c']):
                    lab_results.append(f"- {desc}: {value} {units if units else ''}")
                else:
                    other_observations.append(f"- {desc}: {value} {units if units else ''}")
        
        # Add categorized observations
        if vital_signs:
            patient_info.append("Vital Signs:")
            patient_info.extend(vital_signs)
        if lab_results:
            patient_info.append("Lab Results:")
            patient_info.extend(lab_results)
        if other_observations:
            patient_info.append("Other Clinical Observations:")
            patient_info.extend(other_observations[:5])  # Limit other observations
    
    # Procedures
    procedures = patient_data.get('procedures', [])
    if isinstance(procedures, np.ndarray) and len(procedures) > 0:
        patient_info.append("\nRECENT PROCEDURES:")
        for i, procedure in enumerate(procedures[:10]):  # Limit to first 10 procedures
            patient_info.append(f"- {procedure}")
        if len(procedures) > 10:
            patient_info.append(f"...and {len(procedures) - 10} more procedures")
    
    return "\n".join(patient_info)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on medical MCQs with PrimeKG RAG")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Nano1337/medical-mcqs",
        help="Hugging Face dataset name containing medical MCQs"
    )
    parser.add_argument(
        "--synthea_path",
        type=str,
        default="/home/ubuntu/haoli/privmedrag/dataset/synthea-unified.parquet",
        help="Path to the Synthea unified parquet file"
    )
    parser.add_argument(
        "--primekg_path",
        type=str,
        default="/home/ubuntu/haoli/privmedrag/dataset/primekg",
        help="Path to the PrimeKG dataset directory"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output/evaluation_results.json",
        help="Path to save the evaluation results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for evaluation (or 'gemini' for Gemini API)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.0-flash",
        help="Specific Gemini model to use when --model is 'gemini' or --use_gemini is set"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Number of questions to evaluate (set to -1 for all)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for LLM calls"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_gemini",
        action='store_true',
        help="Use Gemini API instead of OpenAI"
    )
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load MCQs from Hugging Face dataset
    print(f"Loading MCQs from Hugging Face dataset: {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name)
        mcqs = dataset['train']
        print(f"Loaded {len(mcqs)} MCQs from dataset")
        
        # Limit number of questions if specified
        if args.num_questions > 0 and args.num_questions < len(mcqs):
            # Get a deterministic sample for evaluation based on seed
            # This ensures the same questions are selected each time with the same seed
            indices = list(range(len(mcqs)))
            random.shuffle(indices)  # Shuffle is deterministic with the set seed
            indices = indices[:args.num_questions]  # Take the first n questions after shuffling
            indices.sort()  # Sort to ensure consistent processing order
            mcqs = [mcqs[i] for i in indices]
            print(f"Selected {len(mcqs)} deterministic MCQs for evaluation with seed {args.seed}")
    except Exception as e:
        print(f"Error loading MCQs from Hugging Face: {e}")
        return 1
        
    # Load PrimeKG dataset for graph traversal
    print(f"Loading PrimeKG dataset from {args.primekg_path}...")
    try:
        primekg_dataset = PrimeKGDataset(args.primekg_path)
        print(f"Successfully loaded PrimeKG with {len(primekg_dataset.raw_ndata['name'])} nodes")
    except Exception as e:
        print(f"Error loading PrimeKG dataset: {e}")
        print("Continuing without graph traversal capabilities")
        primekg_dataset = None
    
    # Evaluate MCQs with and without RAG
    llm_results = []
    rag_results = []
    
    print(f"Evaluating {len(mcqs)} questions with model {args.model}...")
    
    use_gemini = args.use_gemini or args.model.lower() == "gemini"
    if use_gemini:
        if not GEMINI_AVAILABLE:
            print("Error: Gemini API not available. Install with 'pip install google-generativeai'")
            print("Falling back to OpenAI API...")
            use_gemini = False
        elif not os.environ.get("GEMINI_API_KEY"):
            print("Error: GEMINI_API_KEY not found in environment variables")
            print("Please add it to your .env file or set it in your environment")
            return 1
        else:
            print(f"Using Gemini API for evaluation with model: {args.gemini_model}")
            args.model = args.gemini_model  # Set the model name for reporting
    
    for i, mcq in enumerate(mcqs):
        print(f"\nProcessing question {i+1}/{len(mcqs)} - Type: {mcq['question_type']}")
        
        # Load patient data for this MCQ
        patient_id = mcq['patient_id']
        print(f"Loading patient data for ID: {patient_id}")
        patient_data = load_patient_data(patient_id, args.synthea_path)
        
        # Evaluate without RAG
        print("Evaluating without RAG...")
        llm_result = evaluate_mcq(
            mcq=mcq,
            patient_data=patient_data,
            primekg_dataset=None,  # No RAG
            model=args.model,
            use_rag=False,
            max_retries=args.max_retries,
            use_gemini=use_gemini
        )
        llm_results.append(llm_result)
        
        # Evaluate with RAG
        print("Evaluating with RAG using PrimeKG...")
        rag_result = evaluate_mcq(
            mcq=mcq,
            patient_data=patient_data,
            primekg_dataset=primekg_dataset,  # Use RAG
            model=args.model,
            use_rag=True,
            max_retries=args.max_retries,
            use_gemini=use_gemini
        )
        rag_results.append(rag_result)
        
        # Print immediate results
        print(f"Question: {mcq['question'][:100]}...")
        # Ensure correct_index is within bounds
        option_labels = ['a', 'b', 'c', 'd']
        correct_idx = mcq['correct_index']
        if 0 <= correct_idx < len(option_labels):
            correct_answer = option_labels[correct_idx]
        else:
            correct_answer = f"unknown (index {correct_idx})"
        print(f"Correct answer: {correct_answer}")
        print(f"Without RAG: {llm_result['model_answer']} - {'✓' if llm_result['correct'] else '✗'}")
        print(f"With RAG: {rag_result['model_answer']} - {'✓' if rag_result['correct'] else '✗'}")
    
    # Analyze results
    llm_accuracy = calculate_accuracy(llm_results)
    rag_accuracy = calculate_accuracy(rag_results)
    
    # Calculate accuracy by question type
    llm_by_type = calculate_accuracy_by_type(llm_results)
    rag_by_type = calculate_accuracy_by_type(rag_results)
    
    # Prepare detailed results
    results = {
        "overall": {
            "base_accuracy": llm_accuracy,
            "rag_accuracy": rag_accuracy,
            "improvement": rag_accuracy - llm_accuracy
        },
        "by_question_type": {
            q_type: {
                "base_accuracy": llm_by_type.get(q_type, 0),
                "rag_accuracy": rag_by_type.get(q_type, 0),
                "improvement": rag_by_type.get(q_type, 0) - llm_by_type.get(q_type, 0),
                "num_questions": sum(1 for r in llm_results if r.get("question_type") == q_type)
            }
            for q_type in set(list(llm_by_type.keys()) + list(rag_by_type.keys()))
        },
        "question_results": [
            {
                "question_id": i,
                "patient_id": mcqs[i]['patient_id'],
                "question_type": mcqs[i]['question_type'],
                "question": mcqs[i]['question'],
                "options": mcqs[i]['options'].tolist() if hasattr(mcqs[i]['options'], 'tolist') else mcqs[i]['options'],
                "correct_answer": ['a', 'b', 'c', 'd'][mcqs[i]['correct_index']] if 0 <= mcqs[i]['correct_index'] < 4 else f"unknown (index {mcqs[i]['correct_index']})",
                "base_answer": llm_results[i]['model_answer'],
                "rag_answer": rag_results[i]['model_answer'],
                "base_correct": llm_results[i]['correct'],
                "rag_correct": rag_results[i]['correct']
            } for i in range(len(mcqs))
        ],
        "metadata": {
            "model": args.model,
            "num_questions": len(mcqs),
            "dataset": args.dataset_name,
            "primekg_path": args.primekg_path,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Save results to file if specified
    if args.output_file:
        # Add improved and degraded question details to results
        improved_questions = [i for i in range(len(mcqs)) 
                             if rag_results[i]['correct'] and not llm_results[i]['correct']]
        degraded_questions = [i for i in range(len(mcqs)) 
                             if not rag_results[i]['correct'] and llm_results[i]['correct']]
        
        results["improved_questions"] = [
            {
                "question_id": idx,
                "patient_id": mcqs[idx]['patient_id'],
                "question_type": mcqs[idx]['question_type'],
                "question": mcqs[idx]['question'],
                "options": mcqs[idx]['options'].tolist() if hasattr(mcqs[idx]['options'], 'tolist') else mcqs[idx]['options'],
                "correct_answer": ['a', 'b', 'c', 'd'][mcqs[idx]['correct_index']] if 0 <= mcqs[idx]['correct_index'] < 4 else "unknown",
                "base_answer": llm_results[idx]['model_answer'],
                "rag_answer": rag_results[idx]['model_answer']
            } for idx in improved_questions
        ]
        
        results["degraded_questions"] = [
            {
                "question_id": idx,
                "patient_id": mcqs[idx]['patient_id'],
                "question_type": mcqs[idx]['question_type'],
                "question": mcqs[idx]['question'],
                "options": mcqs[idx]['options'].tolist() if hasattr(mcqs[idx]['options'], 'tolist') else mcqs[idx]['options'],
                "correct_answer": ['a', 'b', 'c', 'd'][mcqs[idx]['correct_index']] if 0 <= mcqs[idx]['correct_index'] < 4 else "unknown",
                "base_answer": llm_results[idx]['model_answer'],
                "rag_answer": rag_results[idx]['model_answer']
            } for idx in degraded_questions
        ]
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total Questions: {len(mcqs)}")
    print(f"Base Model Accuracy: {llm_accuracy:.2%}")
    print(f"RAG-enhanced Accuracy: {rag_accuracy:.2%}")
    print(f"Improvement with RAG: {rag_accuracy - llm_accuracy:.2%}")
    
    print("\nResults by Question Type:")
    for q_type in sorted(results["by_question_type"].keys()):
        type_results = results["by_question_type"][q_type]
        num_questions = type_results['num_questions']
        print(f"  {q_type} ({num_questions} questions):")
        print(f"    Base: {type_results['base_accuracy']:.2%}")
        print(f"    RAG: {type_results['rag_accuracy']:.2%}")
        print(f"    Improvement: {type_results['improvement']:.2%}")
        
    # Find questions where RAG made a difference
    improved_questions = [i for i in range(len(mcqs)) 
                         if rag_results[i]['correct'] and not llm_results[i]['correct']]
    degraded_questions = [i for i in range(len(mcqs)) 
                         if not rag_results[i]['correct'] and llm_results[i]['correct']]
    
    print(f"\nQuestions improved by RAG: {len(improved_questions)}")
    print(f"Questions degraded by RAG: {len(degraded_questions)}")
    
    # Log detailed information about improved questions
    if improved_questions:
        print("\n=== QUESTIONS IMPROVED BY RAG ===\n")
        for idx in improved_questions:
            print(f"Question {idx+1} - Type: {mcqs[idx]['question_type']}")
            print(f"Patient ID: {mcqs[idx]['patient_id']}")
            print(f"Q: {mcqs[idx]['question']}")
            
            # Print options with correct one marked
            option_labels = ['a', 'b', 'c', 'd']
            correct_idx = mcqs[idx]['correct_index']
            for i, option in enumerate(mcqs[idx]['options']):
                if i < len(option_labels):
                    marker = "✓" if i == correct_idx else " "
                    print(f"  {option_labels[i]}. {option} {marker}")
            
            print(f"Base model answer: {llm_results[idx]['model_answer']} (incorrect)")
            print(f"RAG answer: {rag_results[idx]['model_answer']} (correct)")
            print("\n" + "-"*50 + "\n")
    
    # Log detailed information about degraded questions
    if degraded_questions:
        print("\n=== QUESTIONS DEGRADED BY RAG ===\n")
        for idx in degraded_questions:
            print(f"Question {idx+1} - Type: {mcqs[idx]['question_type']}")
            print(f"Patient ID: {mcqs[idx]['patient_id']}")
            print(f"Q: {mcqs[idx]['question']}")
            
            # Print options with correct one marked
            option_labels = ['a', 'b', 'c', 'd']
            correct_idx = mcqs[idx]['correct_index']
            for i, option in enumerate(mcqs[idx]['options']):
                if i < len(option_labels):
                    marker = "✓" if i == correct_idx else " "
                    print(f"  {option_labels[i]}. {option} {marker}")
            
            print(f"Base model answer: {llm_results[idx]['model_answer']} (correct)")
            print(f"RAG answer: {rag_results[idx]['model_answer']} (incorrect)")
            print("\n" + "-"*50 + "\n")
    
    return 0

def get_graph_traversal_context(query_terms: List[str], primekg_dataset, max_neighbors=20, n_hops=1):
    """
    Retrieve relevant medical knowledge from PrimeKG by traversing the graph
    starting from entities that match the query terms
    
    Args:
        query_terms: List of medical terms to search for in the graph
        primekg_dataset: Loaded PrimeKG dataset
        max_neighbors: Maximum number of neighbors to retrieve
        n_hops: Number of hops for graph traversal
        
    Returns:
        String containing relevant medical knowledge from the graph
    """
    if not primekg_dataset:
        return "No knowledge graph data available."
    
    # Extract node data from PrimeKG
    names = primekg_dataset.raw_ndata["name"]
    descriptions = primekg_dataset.raw_ndata["description"]
    categories = primekg_dataset.raw_ndata["category"]
    graph = primekg_dataset.graph
    
    # Filter out non-healthcare related terms
    non_medical_terms = [
        "higher education", "education", "employment", "full-time", "part-time", 
        "transportation", "access", "finding", "received", "lack of"
    ]
    
    # Only keep healthcare-related entities
    filtered_terms = []
    for term in query_terms:
        # Skip entities that match non-medical terms
        if any(non_med in term.lower() for non_med in non_medical_terms):
            continue
            
        # Remove common non-medical qualifiers
        clean_term = term
        for suffix in [" (finding)", " (disorder)", " (procedure)"]:
            if clean_term.endswith(suffix):
                clean_term = clean_term[:-len(suffix)]
        
        filtered_terms.append(clean_term)
        
    if not filtered_terms:
        return "No healthcare-related entities found for KG search."
    
    # Find matching entities using multiple strategies
    print("Searching for matching entities in PrimeKG...")
    
    # Get relevant medical categories
    medical_categories = ['drug', 'disease', 'symptom', 'chemical', 'gene', 'protein', 'anatomy']
    
    # Track matched nodes and their mapping to query terms
    matched_nodes = []
    query_entity_map = {query: [] for query in filtered_terms}
    found_node_indices = set()
    
    # For each query term, try to find matches in the knowledge graph
    for query_term in filtered_terms:
        clean_query = query_term.lower().strip()
        found_match = False
        
        # 1. Try exact name match first (highest priority)
        exact_matches = []
        for i, name in enumerate(names):
            if name.lower() == clean_query:
                exact_matches.append(i)
        
        if exact_matches:
            idx = exact_matches[0]  # Take the first exact match
            matched_nodes.append((idx, names[idx], 'exact'))
            query_entity_map[query_term].append(idx)
            found_node_indices.add(idx)
            print(f"KG Exact Match: '{query_term}' -> '{names[idx]}'")
            found_match = True
            continue
        
        # 2. Try medical term matching with category filtering
        category_matches = []
        
        # First, identify potential matches by significant word overlap
        query_words = set(clean_query.split())
        if len(query_words) >= 1:
            for i, name in enumerate(names):
                # Skip if node category doesn't match medical categories
                node_category = categories[i].lower()
                if not any(cat in node_category for cat in medical_categories):
                    continue
                    
                name_lower = name.lower()
                name_words = set(name_lower.split())
                
                # Calculate word overlap for multi-word terms
                if len(query_words) > 1 and len(name_words) > 1:
                    common_words = query_words.intersection(name_words)
                    if common_words:
                        # Calculate Jaccard similarity
                        similarity = len(common_words) / len(query_words.union(name_words))
                        if similarity > 0.3:  # Reasonable threshold
                            category_matches.append((i, similarity, 'word_overlap'))
                
                # For single words or when word overlap fails, try more specific matching
                elif len(query_words) <= 2 or len(name_words) <= 2:
                    # Check for significant substring match
                    if len(clean_query) > 3 and len(name_lower) > 3:
                        if clean_query in name_lower:
                            # Prefer matches where query is a larger portion of the name
                            similarity = len(clean_query) / len(name_lower)
                            if similarity > 0.5:  # Significant portion
                                category_matches.append((i, similarity, 'substring'))
                        # Also check if any significant word in the query matches
                        elif any(word in name_lower for word in query_words if len(word) > 3):
                            # Less confident match
                            similarity = 0.4
                            category_matches.append((i, similarity, 'keyword'))
        
        # Sort matches by similarity score
        category_matches.sort(key=lambda x: x[1], reverse=True)
        
        if category_matches:
            # Take the best match
            idx, score, match_type = category_matches[0]
            matched_nodes.append((idx, names[idx], match_type))
            query_entity_map[query_term].append(idx)
            found_node_indices.add(idx)
            print(f"KG {match_type.title()} Match: '{query_term}' -> '{names[idx]}' (Score: {score:.2f})")
            found_match = True
        
        if not found_match:
            print(f"KG No Match: '{query_term}' not found in PrimeKG node names.")
    
    # If no matches found, return empty context
    if not matched_nodes:
        return "No relevant entities found in knowledge graph."
    
    # Sort matches (exact matches first, then by match type)
    matched_nodes.sort(key=lambda x: (0 if x[2] == 'exact' else 
                                     1 if x[2] == 'word_overlap' else 
                                     2 if x[2] == 'substring' else 3))
    
    # Take top 3 matches as starting points
    start_nodes = matched_nodes[:3]
    
    # Define important medical relationship types to prioritize with weights
    PRIORITY_RELATIONS = {
        'drug': {
            'indication': 1.5, 
            'contraindication': 1.3, 
            'off-label use': 1.2, 
            'drug_effect': 1.4, 
            'drug_protein': 1.0,
            'treats': 1.8
        },
        'disease': {
            'indication': 1.5, 
            'contraindication': 1.3, 
            'symptom': 1.6,
            'risk_factor': 1.5,
            'disease_disease': 1.4, 
            'disease_protein': 1.0,
            'has_symptom': 1.7
        },
        'symptom': {
            'symptom_of': 1.8,
            'co_occurs_with': 1.4,
            'diagnoses': 1.7
        },
        'gene': {
            'disease_protein': 1.0, 
            'drug_protein': 1.0, 
            'protein_protein': 0.8
        }
    }
    
    all_neighbors = set()
    edge_info = {}
    
    # For each starting node, traverse the graph using DGL's efficient neighbor sampling
    for node_idx, node_name, _ in start_nodes:
        # Get node category
        node_category = categories[node_idx].split('/')[0] if '/' in categories[node_idx] else categories[node_idx]
        node_category_lower = node_category.lower()
        
        # Get priority relationships for this node type
        priority_rels = PRIORITY_RELATIONS.get(node_category_lower, {})
        
        # Collect relations using DGL's efficient neighbor sampling
        collected_relations = {node_idx: []}
        
        # Get outgoing edges
        out_edges = graph.out_edges(node_idx)
        if isinstance(out_edges, tuple) and len(out_edges) == 2:
            src_nodes, dst_nodes = out_edges
            
            # Process all destination nodes and score them
            relation_scores = []
            for i in range(len(dst_nodes)):
                try:
                    dst = dst_nodes[i].item() if hasattr(dst_nodes[i], 'item') else int(dst_nodes[i])
                    # Find the relation type
                    edge_id = graph.edge_ids(node_idx, dst)
                    
                    # Get relation type from edge data
                    if hasattr(edge_id, 'numel') and edge_id.numel() > 0:
                        rel_type = graph.edata['relation_type'][edge_id].item() if hasattr(edge_id, 'item') else str(edge_id)
                        
                        # Calculate priority score based on relation type
                        rel_type_str = str(rel_type)
                        # Check if this is a priority relation for this node type
                        score = 0.5  # Base score for all relations
                        
                        # Boost score if it's a priority relation
                        for priority_key, weight in priority_rels.items():
                            if priority_key in rel_type_str.lower():
                                score = weight
                                break
                        
                        # Add target node category bonus
                        dst_category = categories[dst].split('/')[0].lower() if '/' in categories[dst] else categories[dst].lower()
                        # Boost specific category connections
                        if dst_category in ['disease', 'symptom', 'drug', 'clinical_finding']:
                            score += 0.3
                        
                        relation_scores.append((rel_type, dst, '->', score))
                except Exception as e:
                    continue
            
            # Sort by score and take top max_neighbors/2
            relation_scores.sort(key=lambda x: x[3], reverse=True)
            top_relations = relation_scores[:max_neighbors // 2]
            
            # Add top relations to collected_relations
            for rel_type, dst, direction, _ in top_relations:
                collected_relations[node_idx].append((rel_type, dst, direction))
                all_neighbors.add(dst)
        
        # Get incoming edges
        in_edges = graph.in_edges(node_idx)
        if isinstance(in_edges, tuple) and len(in_edges) == 2:
            src_nodes, dst_nodes = in_edges
            
            # Process all source nodes and score them
            relation_scores = []
            for i in range(len(src_nodes)):
                try:
                    src = src_nodes[i].item() if hasattr(src_nodes[i], 'item') else int(src_nodes[i])
                    # Find the relation type
                    edge_id = graph.edge_ids(src, node_idx)
                    
                    # Get relation type from edge data
                    if hasattr(edge_id, 'numel') and edge_id.numel() > 0:
                        rel_type = graph.edata['relation_type'][edge_id].item() if hasattr(edge_id, 'item') else str(edge_id)
                        
                        # Calculate priority score based on relation type
                        rel_type_str = str(rel_type)
                        # Check if this is a priority relation for this node type
                        score = 0.5  # Base score for all relations
                        
                        # Boost score if it's a priority relation
                        for priority_key, weight in priority_rels.items():
                            if priority_key in rel_type_str.lower():
                                score = weight
                                break
                        
                        # Add source node category bonus
                        src_category = categories[src].split('/')[0].lower() if '/' in categories[src] else categories[src].lower()
                        # Boost specific category connections
                        if src_category in ['disease', 'symptom', 'drug', 'clinical_finding']:
                            score += 0.3
                        
                        relation_scores.append((rel_type, src, '<-', score))
                except Exception as e:
                    continue
            
            # Sort by score and take top max_neighbors/2
            relation_scores.sort(key=lambda x: x[3], reverse=True)
            top_relations = relation_scores[:max_neighbors // 2]
            
            # Add top relations to collected_relations
            for rel_type, src, direction, _ in top_relations:
                collected_relations[node_idx].append((rel_type, src, direction))
                all_neighbors.add(src)
        
        edge_info.update(collected_relations)
    
    # Format the retrieved knowledge - group by category for better organization
    knowledge_context = ["KNOWLEDGE GRAPH INFORMATION:"]
    
    # Add starting point entities
    knowledge_context.append("\n* RELEVANT ENTITIES:")
    for node_idx, node_name, match_type in start_nodes:
        node_category = categories[node_idx]
        node_desc = descriptions[node_idx]
        knowledge_context.append(f"- {node_name} ({node_category}): {node_desc}")
    
    # Group related entities by category
    entities_by_category = {}
    for node in all_neighbors:
        if node in [idx for idx, _, _ in start_nodes]:  # Skip query nodes
            continue
        cat = categories[node]
        main_cat = cat.split('/')[0] if '/' in cat else cat
        if main_cat not in entities_by_category:
            entities_by_category[main_cat] = []
        entities_by_category[main_cat].append(node)
    
    # Only include the most clinically relevant entity categories
    relevant_categories = ['disease', 'symptom', 'drug', 'clinical_finding', 'procedure']
    for category, nodes in entities_by_category.items():
        if category.lower() in relevant_categories:
            knowledge_context.append(f"\n{category.upper()} ENTITIES:")
            for node in nodes[:5]:  # Limit to 5 entities per category
                node_name = names[node]
                node_desc = descriptions[node]
                knowledge_context.append(f"- {node_name}: {node_desc}")
    
    # Add relationships
    knowledge_context.append("\n* KEY RELATIONSHIPS:")
    for node_idx, node_name, _ in start_nodes:
        if node_idx in edge_info:
            knowledge_context.append(f"\nRelationships for {node_name}:")
            for rel, target_idx, direction in edge_info[node_idx][:10]:  # Limit to 10 relationships
                target_name = names[target_idx]
                # Format relation for better readability
                formatted_rel = rel.replace('_', ' ').capitalize()
                if direction == '->':
                    knowledge_context.append(f"- {node_name} {formatted_rel} {target_name}")
                else:
                    knowledge_context.append(f"- {target_name} {formatted_rel} {node_name}")
    
    return "\n".join(knowledge_context)


def chat_with_gemini(prompt, sys_prompt=None, model_name="gemini-2.0-flash"):
    """
    Call the Gemini API with the given prompt using the updated API format
    
    Args:
        prompt: The prompt to send to Gemini
        sys_prompt: System prompt (used in the prompt for Gemini)
        model_name: The specific Gemini model to use
        
    Returns:
        The response from Gemini
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini API not available. Install with 'pip install google-generativeai'")
    
    # Prepare the full prompt with system message if provided
    full_prompt = f"{sys_prompt}\n\n{prompt}" if sys_prompt else prompt
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Prepare content in the correct format
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=full_prompt),
            ],
        ),
    ]
    
    # Configure response format
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    
    # Call Gemini API and collect the response
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )
        return response.text
    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")


def evaluate_mcq(mcq, patient_data, primekg_dataset, model, use_rag=False, max_retries=3, use_gemini=False):
    """
    Evaluate a single MCQ using the LLM with or without RAG
    
    Args:
        mcq: Dictionary containing the MCQ data
        patient_data: Dictionary containing patient data
        primekg_dataset: PrimeKG dataset for graph traversal (or None if not using RAG)
        model: LLM model to use (or "gemini" for Gemini API)
        use_rag: Whether to use RAG for enhanced context
        max_retries: Maximum number of retries for LLM calls
        use_gemini: Whether to use Gemini API instead of OpenAI
        
    Returns:
        Dictionary with evaluation results
    """
    # Extract MCQ data - IMPORTANT: correct_index should never be included in the prompt
    question = mcq['question']
    options = mcq['options']
    correct_index = mcq['correct_index']
    patient_id = mcq['patient_id']
    question_type = mcq['question_type']
    
    # Format patient data as context
    patient_context = format_patient_data(patient_data)
    
    # Get additional context from knowledge graph if using RAG
    kg_context = ""
    if use_rag and primekg_dataset:
        # Extract key medical terms from the question and options
        medical_terms = []
        
        # Extract medical terms using improved approach
        
        # 1. Add terms from question - extract medical entities
        # Look for capitalized terms and multi-word phrases
        question_words = question.split()
        i = 0
        while i < len(question_words):
            word = question_words[i].strip('.,?!()[]{}"\'\'\"\')')
            
            # Check if this is a potential medical term (capitalized or known prefix)
            if len(word) > 3 and (word[0].isupper() or 
                                 any(word.lower().startswith(prefix) for prefix in 
                                     ['anti', 'hyper', 'hypo', 'cardio', 'neuro', 'gastro', 'hepat'])):
                # Try to capture multi-word medical terms
                term = word
                j = i + 1
                while j < len(question_words) and not question_words[j].endswith(('.', '?', '!')):
                    if question_words[j][0].isupper() or question_words[j].lower() in ['syndrome', 'disease', 'disorder']:
                        term += " " + question_words[j].strip('.,?!()[]{}"\'\'\"\')')
                        j += 1
                    else:
                        break
                        
                medical_terms.append(term)
                i = j
            else:
                i += 1
        
        # 2. Add terms from options - similar approach
        for option in options:
            option_words = option.split()
            i = 0
            while i < len(option_words):
                word = option_words[i].strip('.,?!()[]{}"\'\'\"\')')
                if len(word) > 3 and (word[0].isupper() or 
                                     any(word.lower().startswith(prefix) for prefix in 
                                         ['anti', 'hyper', 'hypo', 'cardio', 'neuro', 'gastro', 'hepat'])):
                    term = word
                    j = i + 1
                    while j < len(option_words) and not option_words[j].endswith(('.', '?', '!')):
                        if option_words[j][0].isupper() or option_words[j].lower() in ['syndrome', 'disease', 'disorder']:
                            term += " " + option_words[j].strip('.,?!()[]{}"\'\'\"\')')
                            j += 1
                        else:
                            break
                    
                    medical_terms.append(term)
                    i = j
                else:
                    i += 1
        
        # 3. Add terms from patient conditions and medications
        conditions = patient_data.get('conditions', [])
        if isinstance(conditions, list) or isinstance(conditions, np.ndarray):
            medical_terms.extend(conditions[:5])  # Add top conditions
        
        medications = patient_data.get('medications', [])
        if isinstance(medications, list) or isinstance(medications, np.ndarray):
            medical_terms.extend(medications[:5])  # Add top medications
        
        # Remove duplicates and limit to top 15 terms
        medical_terms = list(set(medical_terms))[:15]
        
        # Get knowledge graph context
        kg_context = get_graph_traversal_context(medical_terms, primekg_dataset)
    
    # Format the prompt for the LLM - keeping it simple and focused
    prompt = f"""You are a medical expert taking a multiple-choice medical exam. Answer the following question based on your medical knowledge.

{patient_context}
"""
    
    # Add knowledge graph context if using RAG
    if use_rag and kg_context:
        prompt += f"\n{kg_context}\n"
    
    # Add the question and options
    prompt += f"\nQUESTION: {question}\n\nOPTIONS:\n"
    
    # Add options with letter labels - IMPORTANT: correct_index is NOT used here
    option_labels = ['a', 'b', 'c', 'd']
    for i, option in enumerate(options):
        # Ensure we don't go out of bounds if there are fewer than 4 options
        if i < len(options) and i < len(option_labels):
            prompt += f"{option_labels[i]}. {option}\n"
    
    # Keep instructions concise and direct
    prompt += "\nSelect the most appropriate answer from the options above. Provide your answer as a JSON object with the following format: {\"answer\": \"a\"} where the letter corresponds to your selected option."
    
    # Call the LLM with retry logic
    for attempt in range(max_retries):
        try:
            # Call the appropriate LLM API
            sys_msg = "You are a medical expert providing accurate answers to medical questions. Always respond with a valid JSON object containing only the answer letter."
            
            if use_gemini or model.lower() == "gemini":
                # Use Gemini API
                try:
                    # Pass the specific Gemini model name from args
                    model_name = model if model.startswith("gemini") else "gemini-2.0-flash"
                    response = chat_with_gemini(prompt, sys_prompt=sys_msg, model_name=model_name)
                except Exception as e:
                    print(f"Error with Gemini API: {e}")
                    raise e
            else:
                # Use OpenAI API
                response = llm_utils.chat_openai(prompt, model=model, sys_prompt=sys_msg)
            
            # Parse the response to extract the answer
            # Try to find JSON object in the response
            import re
            import json
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{\s*"answer"\s*:\s*"([a-d])"\s*\}', response)
            if json_match:
                answer_letter = json_match.group(1).lower()
                # Make sure the answer letter is valid
                if answer_letter in option_labels:
                    answer_index = option_labels.index(answer_letter)
                    # Ensure correct_index is within bounds
                    if 0 <= correct_index < len(option_labels):
                        is_correct = (answer_index == correct_index)
                    else:
                        is_correct = False
                        print(f"Warning: correct_index {correct_index} out of bounds")
                else:
                    answer_index = -1
                    is_correct = False
                
                return {
                    "mcq_id": f"{patient_id}_{question_type}",
                    "question_type": question_type,
                    "correct": is_correct,
                    "expected_answer": option_labels[correct_index] if 0 <= correct_index < len(option_labels) else "unknown",
                    "model_answer": answer_letter,
                    "rag_used": use_rag
                }
            else:
                # If no JSON found, try again with more explicit instructions
                print(f"Attempt {attempt+1}: Failed to parse JSON response. Retrying...")
                if attempt == max_retries - 1:
                    # On last attempt, default to incorrect
                    return {
                        "mcq_id": f"{patient_id}_{question_type}",
                        "question_type": question_type,
                        "correct": False,
                        "expected_answer": option_labels[correct_index],
                        "model_answer": "invalid",
                        "rag_used": use_rag,
                        "error": "Failed to parse response"
                    }
        except Exception as e:
            print(f"Attempt {attempt+1}: Error calling LLM: {e}")
            if attempt == max_retries - 1:
                # On last attempt, default to incorrect
                return {
                    "mcq_id": f"{patient_id}_{question_type}",
                    "question_type": question_type,
                    "correct": False,
                    "expected_answer": option_labels[correct_index],
                    "model_answer": "error",
                    "rag_used": use_rag,
                    "error": str(e)
                }
            time.sleep(2)  # Wait before retrying

def calculate_accuracy(results):
    """Calculate overall accuracy"""
    if not results:
        return 0
    
    correct = sum(1 for r in results if r.get("correct", False))
    return correct / len(results)

def calculate_accuracy_by_type(results):
    """Calculate accuracy by question type"""
    by_type = {}
    
    for q_type in set(r.get("question_type") for r in results):
        type_results = [r for r in results if r.get("question_type") == q_type]
        if type_results:
            correct = sum(1 for r in type_results if r.get("correct", False))
            by_type[q_type] = correct / len(type_results)
    
    return by_type

if __name__ == "__main__":
    sys.exit(main())
