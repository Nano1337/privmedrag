"""
Utilities for processing EHR data for MCQ generation
"""

import json
import os
import random
import re
from typing import Dict, List, Any, Tuple, Set

def load_patient_data(file_path: str) -> List[Dict[str, Any]]:
    """Load processed patient data from parquet file"""
    import pandas as pd
    import os
    import ast
    from typing import List, Dict, Any
    
    # Use the unified parquet file
    parquet_path = os.path.join(os.path.dirname(file_path), 'synthea-unified.parquet')
    print(f"Loading patient data from {parquet_path}")
    
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found at {parquet_path}")
        print("Using mock patient data for testing")
        return create_mock_data(5)
    
    try:
        # First, let's examine the structure of the parquet file
        df = pd.read_parquet(parquet_path)
        print(f"Loaded parquet file with {len(df)} rows and columns: {df.columns.tolist()}")
        
        # Debug output a single row to understand structure
        if len(df) > 0:
            sample_row = df.iloc[0].to_dict()
            print(f"Sample data types: {', '.join([f'{k}: {type(v).__name__}' for k, v in sample_row.items()])}")
        
        # Create a list of patients
        patients = []
        
        # Process each unique patient
        for patient_id in df['patient_id'].unique():
            # Get all rows for this patient
            patient_data = df[df['patient_id'] == patient_id]
            
            # If we have multiple rows per patient, take the first one for demonstration
            patient_row = patient_data.iloc[0]
            
            # Safe extract function for potentially complex columns
            def safe_extract(column_name, default_value=None):
                if column_name not in df.columns:
                    return default_value
                
                val = patient_row[column_name]
                
                # Handle numpy arrays (found in the data)
                import numpy as np
                if isinstance(val, np.ndarray):
                    # Convert numpy array to list
                    return val.tolist() if val.size > 0 else default_value
                
                # Handle NaN or None
                if pd.isna(val):
                    return default_value
                
                # Process different types
                if isinstance(val, (list, dict)):
                    return val
                elif isinstance(val, str):
                    try:
                        # Try to parse as literal if it looks like a list/dict
                        if (val.startswith('[') and val.endswith(']')) or \
                           (val.startswith('{') and val.endswith('}')):
                            return ast.literal_eval(val)
                        return val
                    except:
                        return val
                else:
                    return val
            
            # Extract main fields
            conditions = safe_extract('conditions', [])
            medications = safe_extract('medications', [])
            observations = safe_extract('observations', [])
            
            # Ensure lists are actually lists
            if not isinstance(conditions, list):
                conditions = [conditions] if conditions else []
            if not isinstance(medications, list):
                medications = [medications] if medications else []
            if not isinstance(observations, list):
                observations = [observations] if observations else []
            
            # Build patient object
            patient = {
                'patient_id': patient_id,
                'conditions_count': len(conditions),
                'medications_count': len(medications),
                'observations_count': len(observations),
                'conditions': conditions,
                'medications': medications,
                'observations': observations
            }
            
            patients.append(patient)
        
        print(f"Successfully processed {len(patients)} patients")
        return patients
        
    except Exception as e:
        import traceback
        print(f"Error processing parquet file: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return create_mock_data(5)

def create_mock_data(count: int) -> List[Dict[str, Any]]:
    """Create mock patient data for testing"""
    return [{
        'patient_id': f'test_patient_{i}',
        'conditions_count': 10,
        'medications_count': 3,
        'observations_count': 30,
        'conditions': ['Alzheimer\'s disease', 'Hypertension'],
        'medications': ['Galantamine 4 MG', 'Memantine hydrochloride 2 MG/ML'],
        'observations': [
            {'description': 'Blood Pressure', 'value': '120/80', 'type': 'text'},
            {'description': 'MMSE', 'value': '21.8', 'type': 'numeric', 'units': '{score}'},
            {'description': 'Body Mass Index', 'value': '28.1', 'type': 'numeric', 'units': 'kg/m2'}
        ]
    } for i in range(count)]

def select_patients_with_rich_data(patients: List[Dict[str, Any]], min_conditions: int = 5, 
                                 min_medications: int = 1, min_observations: int = 20) -> List[Dict[str, Any]]:
    """Select patients with sufficient data for creating meaningful questions"""
    rich_patients = []
    
    for patient in patients:
        conditions_count = patient.get('conditions_count', 0)
        medications_count = patient.get('medications_count', 0)
        observations_count = patient.get('observations_count', 0)
        
        if (conditions_count >= min_conditions and 
            medications_count >= min_medications and 
            observations_count >= min_observations):
            rich_patients.append(patient)
    
    return rich_patients

def extract_time_series_observations(patient: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract observations that have multiple measurements over time"""
    if 'observations' not in patient:
        return {}
    
    # Group observations by description
    grouped_obs = {}
    for obs in patient['observations']:
        desc = obs.get('description')
        if desc:
            if desc not in grouped_obs:
                grouped_obs[desc] = []
            grouped_obs[desc].append(obs)
    
    # Keep only observations with multiple measurements
    time_series = {k: v for k, v in grouped_obs.items() if len(v) > 1}
    
    # Sort each time series by value if numeric
    for desc, series in time_series.items():
        if all(obs.get('type') == 'numeric' for obs in series):
            time_series[desc] = sorted(series, key=lambda x: float(x.get('value', 0)))
    
    return time_series

def extract_social_determinants(patient: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract social determinants of health from observations"""
    if 'observations' not in patient:
        return []
    
    social_keywords = [
        'stress', 'social', 'transportation', 'housing', 'income', 
        'education', 'employment', 'safety', 'food', 'abuse'
    ]
    
    social_factors = []
    for obs in patient['observations']:
        desc = obs.get('description', '').lower()
        if any(keyword in desc for keyword in social_keywords):
            social_factors.append(obs)
    
    return social_factors

def get_meaningful_conditions(patient: Dict[str, Any], max_conditions: int = 5) -> List[str]:
    """Get the most clinically meaningful conditions for the patient"""
    if 'conditions' not in patient or not patient['conditions']:
        return []
    
    # Priority conditions (more clinically significant)
    priority_keywords = [
        'disease', 'syndrome', 'disorder', 'failure', 'cancer', 
        'infection', 'diabetes', 'hypertension', 'dementia',
        'alzheimer', 'stroke', 'infarction', 'copd', 'asthma'
    ]
    
    # Sort conditions: priority conditions first, then others
    conditions = patient['conditions']
    sorted_conditions = sorted(
        conditions,
        key=lambda x: any(kw in x.lower() for kw in priority_keywords),
        reverse=True
    )
    
    return sorted_conditions[:max_conditions]

def get_meaningful_medications(patient: Dict[str, Any]) -> List[str]:
    """Get medications, preferring those with clear therapeutic purpose"""
    if 'medications' not in patient or not patient['medications']:
        return []
    
    return patient['medications']

def identify_applicable_question_types(patient: Dict[str, Any], question_types: Dict[str, Dict]) -> List[str]:
    """Identify which question types are applicable to this patient based on available data"""
    applicable_types = []
    
    # Extract data elements needed for evaluation
    has_multiple_conditions = patient.get('conditions_count', 0) >= 2
    has_medications = patient.get('medications_count', 0) >= 1
    has_multiple_medications = patient.get('medications_count', 0) >= 2
    time_series = extract_time_series_observations(patient)
    has_time_series = len(time_series) > 0
    social_factors = extract_social_determinants(patient)
    has_social_factors = len(social_factors) > 0
    
    # Check each question type for applicability
    for q_type, config in question_types.items():
        if q_type == "CLINICAL_INTERPRETATION" and has_multiple_conditions:
            applicable_types.append(q_type)
        elif q_type == "TREATMENT_REASONING" and has_medications and has_multiple_conditions:
            applicable_types.append(q_type)
        elif q_type == "RISK_ASSESSMENT" and has_multiple_conditions and has_social_factors:
            applicable_types.append(q_type)
        elif q_type == "MECHANISM_INTEGRATION" and has_multiple_medications:
            applicable_types.append(q_type)
        elif q_type == "SYMPTOM_PROGRESSION" and has_time_series:
            applicable_types.append(q_type)
    
    return applicable_types
