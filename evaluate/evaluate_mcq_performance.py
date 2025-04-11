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
import random
import re
from datasets import load_dataset
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Set, Optional
from difflib import get_close_matches
import medspacy
from data_utils import anonymize_data

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


def load_patient_data(patient_id: str, synthea_path: str, privacy_level: int = 0) -> Dict:
    """
    Load patient data from Synthea parquet file for a specific patient ID
    
    Args:
        patient_id: The patient ID to retrieve data for
        synthea_path: Path to the Synthea parquet file
        privacy_level: Privacy level for anonymization (0: No anonymization, 1: Remove PII, 2: k-anonymity and l-diversity)
        
    Returns:
        Dictionary containing patient data or empty dict if not found
    """
    try:
        # Load the entire dataframe - in production, you'd want to optimize this
        # to only load the specific patient data needed
        df = pd.read_parquet(synthea_path)
        df = anonymize_data(df, level=privacy_level)

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
        default="./dataset/synthea-unified.parquet",
        help="Path to the Synthea unified parquet file"
    )
    parser.add_argument(
        "--primekg_path",
        type=str,
        default="./dataset/primekg",
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
    parser.add_argument(
        "--privacy_level",
        type=int,
        default=0,
        help="Privacy level for anonymization (0: No anonymization, 1: Remove PII, 2: k-anonymity and l-diversity)"
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
        print(f"Loading patient data for ID: {patient_id} with privacy level {args.privacy_level}")
        patient_data = load_patient_data(patient_id, args.synthea_path, privacy_level=args.privacy_level)
        
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

def extract_base_medication_name(medication_string):
    """
    Extract the base drug name from a medication string that includes dosage and form
    
    Args:
        medication_string: String like 'lisinopril 10 MG Oral Tablet'
        
    Returns:
        Base drug name (e.g., 'lisinopril')
    """
    # Common patterns in medication strings
    # 1. Extract content before dosage
    dosage_pattern = re.compile(r'(.+?)\s+\d+\s*(?:MG|MCG|ML|G)', re.IGNORECASE)
    match = dosage_pattern.search(medication_string)
    if match:
        return match.group(1).strip()
    
    # 2. Handle brand name with generic in brackets like "Tegretol [Carbamazepine]"
    bracket_pattern = re.compile(r'\[(.+?)\]', re.IGNORECASE)
    match = bracket_pattern.search(medication_string)
    if match:
        return match.group(1).strip()
    
    # 3. Handle generic name with brand in brackets like "Carbamazepine [Tegretol]"
    bracket_pattern = re.compile(r'(.+?)\s*\[', re.IGNORECASE)
    match = bracket_pattern.search(medication_string)
    if match:
        return match.group(1).strip()
    
    # 4. Just take the first word for simplicity if nothing else matched
    words = medication_string.split()
    if words:
        # Handle mixed case like amLODIPine -> amlodipine
        word = words[0]
        if any(c.isupper() for c in word[1:]):
            return word.lower()
        return word
    
    # If all else fails, return the original string
    return medication_string


def extract_medical_entities(text):
    """
    Use medspacy to extract medical entities from text
    
    Args:
        text: Text to extract entities from
        
    Returns:
        List of extracted medical entity texts
    """
    try:
        # Load the medspacy model
        nlp = medspacy.load()
        
        # Process the text
        doc = nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            # Only keep relevant clinical entities
            if ent.label_ in ["PROBLEM", "TREATMENT", "TEST", "MEDICATION", "PROCEDURE"]:
                entities.append(ent.text)
        
        # Add any CYP enzymes which might not be captured
        for token in doc:
            if "CYP" in token.text and token.text not in entities:
                entities.append(token.text)
        
        return entities
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        # Fallback to simple extraction
        words = text.split()
        entities = []
        i = 0
        while i < len(words):
            # Look for capitalized words or medical abbreviations
            if (words[i][0].isupper() and len(words[i]) > 2) or "CYP" in words[i]:
                entities.append(words[i])
            i += 1
        return entities


def get_graph_traversal_context(query_terms: List[str], primekg_dataset, max_neighbors=10, n_hops=2, question_type=None):
    """
    Retrieve relevant medical knowledge from PrimeKG by traversing the graph
    starting from entities that match the query terms
    
    Args:
        query_terms: List of medical terms to search for in the graph
        primekg_dataset: Loaded PrimeKG dataset
        max_neighbors: Maximum number of neighbors to retrieve per node
        n_hops: Number of hops for graph traversal (default: 3)
        
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
    
    print("Searching for matching entities in PrimeKG...")
    
    # Track matched nodes
    matched_nodes = []
    
    # Keep track of matches we've already found to avoid duplication
    matched_entity_names = set()
    
    # Find matches for each medical term in the knowledge graph
    for query_term in filtered_terms:
        clean_query = query_term.lower().strip()
        if len(clean_query) < 3:  # Skip very short terms
            continue
            
        # Skip if we've already matched this term (case-insensitive)
        if clean_query in matched_entity_names:
            continue
            
        # Try to extract base medication name for better matching
        base_medication = extract_base_medication_name(query_term).lower()
        base_medication_length = len(base_medication)
        
        # For fuzzy matching, collect all node names for efficient search
        found_match = False
        all_names_lower = [str(name).lower() for name in names if name]
        
        # 1. Try exact match first (highest priority)
        for i, name_lower in enumerate(all_names_lower):
            if name_lower == clean_query:
                matched_nodes.append((i, 1.0, 'exact'))
                found_match = True
                matched_entity_names.add(clean_query)
                print(f"KG Exact Match: '{query_term}' -> '{names[i]}'")
                break
        
        if found_match:
            continue
            
        # 2. Try exact match with extracted base medication name
        if base_medication_length >= 4 and base_medication != clean_query:
            for i, name_lower in enumerate(all_names_lower):
                if name_lower == base_medication:
                    matched_nodes.append((i, 0.95, 'base_med_exact'))
                    found_match = True
                    matched_entity_names.add(base_medication)
                    print(f"KG Base Medication Match: '{query_term}' -> '{names[i]}'")
                    break
        
        if found_match:
            continue
            
        # 3. Try fuzzy matching with difflib for medications
        if base_medication_length >= 4:
            close_matches = get_close_matches(base_medication, all_names_lower, n=1, cutoff=0.8)
            if close_matches:
                best_match = close_matches[0]
                match_idx = all_names_lower.index(best_match)
                similarity = 1.0 - (1 - 0.8)  # Convert cutoff to similarity score
                matched_nodes.append((match_idx, similarity, 'fuzzy'))
                found_match = True
                matched_entity_names.add(base_medication)
                print(f"KG Fuzzy Match: '{query_term}' -> '{names[match_idx]}' (Score: {similarity:.2f})")
                continue
        
        # 4. Word overlap (for multi-word terms)
        if len(clean_query.split()) > 1:
            best_overlap = None
            best_score = 0.4  # Minimum threshold
            best_idx = -1
            
            query_words = set(clean_query.split())
            for i, name_lower in enumerate(all_names_lower):
                if len(name_lower.split()) > 1:
                    name_words = set(name_lower.split())
                    common_words = query_words.intersection(name_words)
                    if common_words and len(common_words) >= len(query_words) * 0.5:
                        # At least half of query words found in name
                        similarity = len(common_words) / max(len(query_words), len(name_words))
                        if similarity > best_score:
                            best_score = similarity
                            best_overlap = name_lower
                            best_idx = i
            
            if best_overlap:
                matched_nodes.append((best_idx, best_score, 'overlap'))
                found_match = True
                matched_entity_names.add(clean_query)
                print(f"KG Overlap Match: '{query_term}' -> '{names[best_idx]}' (Score: {best_score:.2f})")
                continue
        
        # 5. Substring match (as a last resort)
        if not found_match and len(clean_query) > 3:
            best_substring = None
            best_score = 0.6  # Minimum threshold
            best_idx = -1
            
            for i, name_lower in enumerate(all_names_lower):
                if clean_query in name_lower:
                    similarity = len(clean_query) / len(name_lower)
                    if similarity > best_score:
                        best_score = similarity
                        best_substring = name_lower
                        best_idx = i
            
            if best_substring:
                matched_nodes.append((best_idx, best_score, 'substring'))
                found_match = True
                matched_entity_names.add(clean_query)
                print(f"KG Substring Match: '{query_term}' -> '{names[best_idx]}' (Score: {best_score:.2f})")
        
        if not found_match:
            print(f"KG No Match: '{query_term}' not found in PrimeKG node names.")
    
    # If no matches found, return empty context
    if not matched_nodes:
        return "No relevant entities found in knowledge graph."
    
    # Sort matches by score (higher score first)
    matched_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # Take only the top 2 matches as seed nodes for a more focused search
    seed_nodes = [node_idx for node_idx, _, _ in matched_nodes[:2]]
    print(f"Selected {len(seed_nodes)} seed nodes for traversal")
    
    # Set a hard cap on total nodes to avoid explosion
    MAX_TOTAL_NODES = 75
    
    # Keep track of all visited nodes
    all_important_nodes = set(seed_nodes)
    
    # Perform focused traversal from each seed node separately
    for seed_idx, seed_node in enumerate(seed_nodes):
        print(f"Traversing from seed {seed_idx+1}/{len(seed_nodes)}: {names[seed_node] if seed_node < len(names) else 'Unknown'}")
        
        # Track nodes for this seed
        current_level = {seed_node}
        visited_from_this_seed = {seed_node}
        
        # Traverse n hops from this seed
        for hop in range(n_hops):
            next_level = set()
            
            # Process each node in current level
            for node in current_level:
                if len(all_important_nodes) >= MAX_TOTAL_NODES:
                    break
                    
                try:
                    # Limit outgoing edges - take only the most relevant
                    # Try to get successors - these are outgoing edges
                    successors = graph.successors(node).tolist()
                    # Shuffle to avoid bias
                    random.shuffle(successors)
                    
                    # Only take a limited number of neighbors
                    neighbor_limit = max(3, max_neighbors // (hop+1))  # Reduce limit with distance
                    for i, neighbor in enumerate(successors):
                        if i >= neighbor_limit:  # Hard limit per node
                            break
                            
                        if neighbor not in visited_from_this_seed:
                            next_level.add(neighbor)
                            visited_from_this_seed.add(neighbor)
                            all_important_nodes.add(neighbor)
                            
                    # Get incoming edges, but with lower priority
                    predecessors = graph.predecessors(node).tolist()
                    random.shuffle(predecessors)
                    for i, neighbor in enumerate(predecessors):
                        if i >= neighbor_limit // 2:  # Even stricter limit for incoming edges
                            break
                            
                        if neighbor not in visited_from_this_seed:
                            next_level.add(neighbor)
                            visited_from_this_seed.add(neighbor)
                            all_important_nodes.add(neighbor)
                            
                except Exception as e:
                    print(f"Error traversing from node {node}: {e}")
                    continue
            
            # Move to next level
            current_level = next_level
            print(f"  Hop {hop+1} from seed {seed_idx+1}: Added {len(next_level)} nodes, total {len(visited_from_this_seed)}")
            
            # Stop if we've reached our limit or no more nodes to explore
            if len(all_important_nodes) >= MAX_TOTAL_NODES or not next_level:
                break
        
        # Check if we've hit the overall limit
        if len(all_important_nodes) >= MAX_TOTAL_NODES:
            print(f"Reached node limit of {MAX_TOTAL_NODES}, stopping traversal")
            break
    
    # Collect edges between important nodes
    edge_info = {}
    
    # Use the graph to find connections between our important nodes
    for node_idx in all_important_nodes:
        
        try:
            # Skip if node is out of range
            if node_idx >= len(names):
                continue
                
            # Skip processing if we've already done this node
            if node_idx in edge_info:
                continue
                
            node_name = names[node_idx]
            edge_info[node_idx] = []
            
            # Track neighbors we find
            node_neighbors = []
            
            # Get outgoing edges (node_idx -> other)
            try:
                successors = graph.successors(node_idx).tolist()
                for dst in successors[:max_neighbors]:
                    if dst >= len(names):
                        continue
                        
                    # Get the relationship type
                    try:
                        edge_id = graph.edge_ids(node_idx, dst)
                        if hasattr(edge_id, 'numel') and edge_id.numel() > 0:
                            rel_type = graph.edata['relation_type'][edge_id].item() if hasattr(edge_id, 'item') else edge_id
                            rel_type_str = str(rel_type)
                            
                            # Add this relationship
                            if dst in all_important_nodes:
                                edge_info[node_idx].append((rel_type_str, dst, '→'))
                                node_neighbors.append(dst)
                    except Exception as e:
                        print(f"Error getting edge type: {e}")
            except Exception as e:
                print(f"Error getting successors for node {node_idx}: {e}")
                
            # Get incoming edges (other -> node_idx)
            try:
                predecessors = graph.predecessors(node_idx).tolist()
                for src in predecessors[:max_neighbors]:
                    if src >= len(names):
                        continue
                        
                    # Get the relationship type
                    try:
                        edge_id = graph.edge_ids(src, node_idx)
                        if hasattr(edge_id, 'numel') and edge_id.numel() > 0:
                            rel_type = graph.edata['relation_type'][edge_id].item() if hasattr(edge_id, 'item') else edge_id
                            rel_type_str = str(rel_type)
                            
                            # Add this relationship
                            if src in all_important_nodes:
                                edge_info[node_idx].append((rel_type_str, src, '←'))
                                node_neighbors.append(src)
                    except Exception as e:
                        print(f"Error getting edge type: {e}")
            except Exception as e:
                print(f"Error getting predecessors for node {node_idx}: {e}")
        except Exception as e:
            print(f"Error processing node {node_idx}: {e}")
    
    # Format the retrieved knowledge - group by category for better organization
    knowledge_context = ["KNOWLEDGE GRAPH INFORMATION:"]
    
    # First highlight the seed nodes
    knowledge_context.append("\nSEED ENTITIES:")
    for node_idx in seed_nodes:
        if node_idx < len(names) and node_idx < len(descriptions):
            name = names[node_idx]
            description = descriptions[node_idx]
            knowledge_context.append(f"- {name}: {description}")
    
    # Collect entities with their information
    entity_info = []
    for node in all_important_nodes:
        if node in seed_nodes:  # Skip seed nodes as we already included them
            continue
            
        if node >= len(names) or node >= len(categories) or node >= len(descriptions):
            continue
            
        name = names[node]
        category = categories[node] if node < len(categories) else ""
        description = descriptions[node] if node < len(descriptions) else ""
        
        if not name or not isinstance(name, str):
            continue
            
        # Skip overly generic terms
        if name.lower() in ["disease", "disorder", "syndrome", "drug", "medication"]:
            continue
            
        entity_info.append((node, name, category, description))
    
    # Group entities by category
    entities_by_category = {}
    for node, name, category, description in entity_info:
        # Get the main category
        if not category or not isinstance(category, str):
            continue
            
        # Handle common variations of categories
        main_cat = category.lower()
        
        # Standardize categories to a smaller set
        if 'disease' in main_cat or 'disorder' in main_cat or 'condition' in main_cat:
            main_cat = 'Disease'
        elif 'medication' in main_cat or 'drug' in main_cat or 'compound' in main_cat:
            main_cat = 'Drug'
        elif 'gene' in main_cat or 'protein' in main_cat or 'enzyme' in main_cat:
            main_cat = 'Gene/Protein'
        elif 'symptom' in main_cat or 'finding' in main_cat or 'sign' in main_cat:
            main_cat = 'Symptom/Finding'
        else:
            main_cat = 'Other'  # Group less common categories
            
        if main_cat not in entities_by_category:
            entities_by_category[main_cat] = []
        entities_by_category[main_cat].append((node, name, description))
    
    # Only include the most relevant categories, with type-specific filtering
    for category, nodes in entities_by_category.items():
        if nodes:  # Only add if we have entities
            # Add category header
            knowledge_context.append(f"\n{category.upper()} ENTITIES:")
            
            # Apply category-specific limits
            limit = 3  # Default limit
            # Give more space to key medical categories
            if category in ['Disease', 'Drug', 'Symptom/Finding']:
                limit = 5
                
            for node, name, description in nodes[:limit]:  # Apply the appropriate limit
                knowledge_context.append(f"- {name}: {description}")
    
    # Add relationships - but be more selective
    knowledge_context.append("\n* KEY RELATIONSHIPS:")
    for node_idx in seed_nodes:
        if node_idx in edge_info and node_idx < len(names):
            node_name = names[node_idx]
            knowledge_context.append(f"\nRelationships for {node_name}:")
            # Only take top 5 relationships to reduce noise
            for rel_type, other_node, direction in edge_info[node_idx][:5]:
                other_name = names[other_node] if other_node < len(names) else "Unknown"
                if direction == '→':
                    knowledge_context.append(f"- {node_name} --[{rel_type}]--> {other_name}")
                else:  # direction == '←'
                    knowledge_context.append(f"- {other_name} --[{rel_type}]--> {node_name}")
    
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
        # Use medspacy to extract medical entities from the question and options
        all_text = question + "\n" + "\n".join(options)
        question_entities = extract_medical_entities(all_text)
        
        # Also extract entities from patient information
        patient_text = ""
        if "age" in patient_data:
            patient_text += f"Patient age: {patient_data['age']}. "
        if "gender" in patient_data:
            patient_text += f"Gender: {patient_data['gender']}. "
        if "conditions" in patient_data and isinstance(patient_data['conditions'], list):
            patient_text += "Conditions: " + ", ".join(patient_data['conditions'][:5]) + ". "
        if "medications" in patient_data and isinstance(patient_data['medications'], list):
            patient_text += "Medications: " + ", ".join(patient_data['medications'][:5]) + ". "
            
        patient_entities = []
        if patient_text:
            patient_entities = extract_medical_entities(patient_text)
        
        # Collect all conditions and medications
        conditions = patient_data.get('conditions', [])
        medications = patient_data.get('medications', [])
        
        # Combine all terms and deduplicate - ensure everything is a list
        conditions_list = list(conditions[:5]) if isinstance(conditions, (list, np.ndarray)) else []
        medications_list = list(medications[:5]) if isinstance(medications, (list, np.ndarray)) else []
        
        # Extract base medication names
        base_medication_names = []
        for med in medications_list:
            base_name = extract_base_medication_name(med)
            if base_name and len(base_name) > 3:
                base_medication_names.append(base_name)
                
        # Add both full medication strings and extracted base names
        all_medical_terms = question_entities + patient_entities + conditions_list + medications_list + base_medication_names
        
        # Deduplicate terms (case-insensitive) while preserving original case
        term_map = {}
        for term in all_medical_terms:
            term_lower = term.lower()
            # Keep the longer version if we have duplicates with different casing
            if term_lower not in term_map or len(term) > len(term_map[term_lower]):
                term_map[term_lower] = term
                
        # Create the final deduplicated list
        medical_terms = list(term_map.values())
        
        print(f"Found {len(medical_terms)} unique medical entities")
        
        # This section is now handled above in the deduplication logic
        # No need to add conditions/medications again since we already included them
        
        # We already deduplicated above, just limit to top 15 terms to improve precision
        medical_terms = medical_terms[:15]
        
        # Get knowledge graph context with question type
        kg_context = get_graph_traversal_context(medical_terms, primekg_dataset, question_type=question_type)
    
    # Format the prompt for the LLM - adjusted by question type
    base_prompt = "You are a medical expert taking a multiple-choice medical exam. "
    
    # Add type-specific instructions
    if question_type == 'RISK_ASSESSMENT':
        prompt = f"""{base_prompt}Focus on statistical patterns, patient risk factors, and evidence-based guidelines for risk prediction.

{patient_context}
"""
    elif question_type == 'MECHANISM_INTEGRATION':
        prompt = f"""{base_prompt}Focus on biological mechanisms, physiological processes, and drug interactions.

{patient_context}
"""
    elif question_type == 'CLINICAL_INTERPRETATION':
        prompt = f"""{base_prompt}Focus on interpreting clinical findings, lab results, and diagnostic reasoning.

{patient_context}
"""
    else:  # Default prompt for other question types
        prompt = f"""{base_prompt}Answer the following question based on your medical knowledge.

{patient_context}
"""
    
    # Add knowledge graph context if using RAG
    if use_rag and kg_context:
        if question_type == 'RISK_ASSESSMENT':
            prompt += f"\nRELEVANT RISK FACTORS AND EVIDENCE:\n{kg_context}\n"
        else:
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
