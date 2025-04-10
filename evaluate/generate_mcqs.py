#!/usr/bin/env python3
"""
Main script for generating Medical MCQs from EHR data
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Any

from config import QUESTION_TYPES, NUM_QUESTIONS
from data_utils import (
    load_patient_data,
    select_patients_with_rich_data,
    identify_applicable_question_types
)
from question_generator import QuestionGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate Medical MCQs from EHR data")
    parser.add_argument(
        "--input_file",
        type=str,
        default="../dataset/synthea-info.json",
        help="Path to the input processed patients data file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output/medical_mcqs.json",
        help="Path to save the generated MCQs"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=NUM_QUESTIONS,
        help="Number of MCQs to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--primekg_path",
        type=str,
        default=None,
        help="Path to the PrimeKG dataset directory"
    )
    parser.add_argument(
        "--questions_per_type",
        type=int,
        default=1,
        help="Number of questions to generate per question type for each patient"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="./output/mcq_checkpoint.json",
        help="Path to save checkpoint data for resumable generation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if it exists"
    )
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load and process patient data
    print(f"Loading patient data from {args.input_file}...")
    try:
        patients = load_patient_data(args.input_file)
    except Exception as e:
        print(f"Error loading patient data: {e}")
        return 1
        
    print(f"Found {len(patients)} patients")
    
    # Select patients with rich data
    rich_patients = select_patients_with_rich_data(patients)
    print(f"Selected {len(rich_patients)} patients with rich data")
    
    if not rich_patients:
        print("Error: No patients with sufficient data for MCQ generation")
        return 1
        
    # Initialize question generator
    generator = QuestionGenerator(question_types=QUESTION_TYPES, primekg_path=args.primekg_path)
    
    # Generate MCQs
    print("Generating MCQs...")
    all_mcqs = []
    
    # Load checkpoint if resuming
    if args.resume and os.path.exists(args.checkpoint_file):
        try:
            with open(args.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_mcqs = checkpoint_data.get('mcqs', [])
                processed_patients = set(checkpoint_data.get('processed_patients', []))
                print(f"Loaded checkpoint with {len(all_mcqs)} MCQs and {len(processed_patients)} processed patients")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            processed_patients = set()
    else:
        processed_patients = set()
    
    # Initialize output file with empty structure if it doesn't exist
    if not os.path.exists(args.output_file):
        with open(args.output_file, 'w') as f:
            json.dump({"count": 0, "mcqs": []}, f, indent=2)
    
    for patient in rich_patients:
        patient_id = patient.get('patient_id', 'unknown')
        
        # Skip already processed patients if resuming
        if patient_id in processed_patients:
            print(f"Skipping already processed patient {patient_id}")
            continue
            
        # Identify applicable question types for this patient
        applicable_types = identify_applicable_question_types(patient, QUESTION_TYPES)
        
        if not applicable_types:
            continue
            
        # Generate MCQs for this patient
        patient_mcqs = generator.generate_mcqs_for_patient(patient, applicable_types, args.questions_per_type)
        
        if patient_mcqs:
            all_mcqs.extend(patient_mcqs)
            processed_patients.add(patient_id)
            
            # Save checkpoint after each patient
            checkpoint_data = {
                "mcqs": all_mcqs,
                "processed_patients": list(processed_patients)
            }
            
            try:
                with open(args.checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                    
                # Update output file incrementally
                with open(args.output_file, 'w') as f:
                    json.dump({"count": len(all_mcqs), "mcqs": all_mcqs}, f, indent=2)
                    
                print(f"Updated checkpoint and output with {len(all_mcqs)} MCQs")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
        
    print(f"Generated {len(all_mcqs)} MCQs from {len(rich_patients)} patients")
    
    # Ensure diversity and limit to requested number
    if len(all_mcqs) > args.num_questions:
        # Sort by question type to ensure diversity
        question_type_weights = {q_type: cfg.get('weight', 0.2) for q_type, cfg in QUESTION_TYPES.items()}
        
        # Weighted sampling based on question type
        selected_mcqs = []
        remaining = args.num_questions
        
        # First ensure at least some of each type
        question_types = list(QUESTION_TYPES.keys())
        for q_type in question_types:
            type_mcqs = [mcq for mcq in all_mcqs if mcq['question_type'] == q_type]
            if type_mcqs:
                # Take at least 5 of each type if available, or all if fewer
                num_to_take = min(5, len(type_mcqs), remaining)
                selected_mcqs.extend(random.sample(type_mcqs, num_to_take))
                remaining -= num_to_take
                
        # Then fill the rest with weighted random sampling
        if remaining > 0 and all_mcqs:
            remaining_mcqs = [mcq for mcq in all_mcqs if mcq not in selected_mcqs]
            
            # Calculate weights for each MCQ
            weights = [question_type_weights.get(mcq['question_type'], 0.2) for mcq in remaining_mcqs]
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
                
            # Weighted random sampling without replacement
            remaining_indices = random.choices(
                range(len(remaining_mcqs)),
                weights=weights,
                k=min(remaining, len(remaining_mcqs))
            )
            for idx in remaining_indices:
                if idx < len(remaining_mcqs):
                    selected_mcqs.append(remaining_mcqs[idx])
                    
        all_mcqs = selected_mcqs
    
    # Randomize the order of MCQs
    random.shuffle(all_mcqs)
    
    # Save MCQs to file
    try:
        mcq_data = {
            "count": len(all_mcqs),
            "mcqs": all_mcqs
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(mcq_data, f, indent=2)
            
        print(f"Saved {len(all_mcqs)} MCQs to {args.output_file}")
    except Exception as e:
        print(f"Error saving MCQs: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
