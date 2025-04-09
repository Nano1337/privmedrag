#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
import gc  # Garbage collector for memory management

def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def read_csv_files(csv_dir):
    """Read all CSV files from the specified directory.
    
    Args:
        csv_dir (str): Path to directory containing CSV files
        
    Returns:
        dict: Dictionary with filename (without extension) as key and DataFrame as value
    """
    csv_files = {}
    for file_path in Path(csv_dir).glob('*.csv'):
        filename = file_path.stem
        try:
            df = pd.read_csv(file_path)
            csv_files[filename] = df
            logging.info(f"Read {filename}.csv: {len(df)} rows, {df.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error reading {filename}.csv: {e}")
    return csv_files

def inspect_dataframes(dataframes, sample_rows=5):
    """Print sample rows and column information for each DataFrame.
    
    Args:
        dataframes (dict): Dictionary of DataFrames
        sample_rows (int): Number of sample rows to display
    """
    for name, df in dataframes.items():
        logging.info(f"\nInspecting {name}:")
        logging.info(f"Shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"Sample rows:\n{df.head(sample_rows)}")
        # Try to identify patient ID column
        patient_cols = [col for col in df.columns if 'patient' in col.lower()]
        if patient_cols:
            logging.info(f"Potential patient ID columns: {patient_cols}")

def identify_relationships(dataframes):
    """Identify relationships between tables to map indirect patient relationships.
    
    Args:
        dataframes (dict): Dictionary of DataFrames
        
    Returns:
        dict: Dictionary mapping tables without patient IDs to intermediary tables
    """
    # We're not mapping administrative tables (payers, organizations, providers)
    # as per user request to focus only on patient health information
    return {}

def group_by_patient(dataframes):
    """Group each DataFrame by patient ID.
    
    Args:
        dataframes (dict): Dictionary of DataFrames
        
    Returns:
        dict: Dictionary with grouped DataFrames and relationship info
    """
    result = {
        'grouped_dfs': {},
        'patient_id_columns': {}
    }
    
    # Identify the patient ID column for each DataFrame
    patient_id_columns = {
        'patients': 'Id',  # Assuming 'Id' is the patient ID in patients table
        'conditions': 'PATIENT',
        'medications': 'PATIENT',
        'encounters': 'PATIENT',
        'observations': 'PATIENT',
        'procedures': 'PATIENT',
        'immunizations': 'PATIENT',
        'careplans': 'PATIENT',
        'allergies': 'PATIENT',
        'devices': 'PATIENT',
    }
    
    # Dynamically determine patient ID columns if they're not in our mapping
    for name, df in dataframes.items():
        if name not in patient_id_columns:
            # Look for columns containing 'patient' (case insensitive)
            patient_cols = [col for col in df.columns if 'patient' in col.lower()]
            if patient_cols:
                patient_id_columns[name] = patient_cols[0]
    
    result['patient_id_columns'] = patient_id_columns
    
    # Group DataFrames by patient ID
    for name, df in dataframes.items():
        if name in patient_id_columns and patient_id_columns[name] in df.columns:
            patient_col = patient_id_columns[name]
            # Group by patient and convert to dictionary for easier joining
            grouped = df.groupby(patient_col)
            result['grouped_dfs'][name] = grouped
            logging.info(f"Grouped {name} by {patient_col}: {grouped.ngroups} patients")
    
    return result

def process_tables_without_direct_patient_id(dataframes, group_result, relationships):
    """Process tables that don't have direct patient IDs using relationship mapping.
    
    Args:
        dataframes (dict): Original DataFrames
        group_result (dict): Result from group_by_patient
        relationships (dict): Relationship mappings
        
    Returns:
        dict: Updated dataframes with patient ID mappings added
    """
    # We're skipping administrative tables (payers, organizations, providers)
    # as per user request to focus only on patient health information
    logging.info("Skipping administrative tables as requested")
    return group_result

def join_patient_data(dataframes, group_result):
    """Join all patient data into a unified DataFrame.
    
    Args:
        dataframes (dict): Original DataFrames
        group_result (dict): Result containing grouped DataFrames
        
    Returns:
        pd.DataFrame: Unified patient data
    """
    grouped_dfs = group_result['grouped_dfs']
    patient_id_columns = group_result['patient_id_columns']
    
    # Start with the patients DataFrame as our base
    if 'patients' in dataframes:
        base_df = dataframes['patients'].copy()
        patient_id_col = 'Id'  # Assuming this is the ID column in patients table
    else:
        # If no patients table, use the largest table with patient IDs as base
        largest_table = max([(name, df) for name, df in dataframes.items() 
                            if any('patient' in col.lower() for col in df.columns)],
                           key=lambda x: len(x[1]))
        base_df = largest_table[1].copy()
        # Find patient ID column
        patient_id_col = next(col for col in base_df.columns if 'patient' in col.lower())
        logging.warning(f"No patients table found, using {largest_table[0]} as base with ID column {patient_id_col}")
    
    # Create a unique list of all patient IDs
    all_patient_ids = set(base_df[patient_id_col])
    
    # Define the order of tables to process for consistent column ordering
    # Start with patients, then clinical tables (excluding administrative tables)
    table_processing_order = [
        'patients',  # Demographics first
        # Clinical data - focus only on patient health information
        'encounters', 'conditions', 'observations', 'procedures', 'medications',
        'immunizations', 'allergies', 'devices', 'careplans'
    ]
    
    # Filter the order to only include tables that exist in our dataset
    table_processing_order = [table for table in table_processing_order if table in grouped_dfs or table == 'patients']
    
    # Add any remaining tables that weren't in our predefined order
    for table in grouped_dfs:
        if table not in table_processing_order:
            table_processing_order.append(table)
    
    logging.info(f"Processing tables in order: {table_processing_order}")
    
    # Identify date columns for proper conversion
    date_columns = set()
    for table_name, df in dataframes.items():
        for col in df.columns:
            if col.upper() in ('START', 'STOP', 'DATE', 'BIRTHDATE', 'DEATHDATE') or 'DATE' in col.upper():
                date_columns.add((table_name, col))
    
    # Define categorical columns that should be deduplicated
    categorical_suffixes = ('_TYPE', '_UNITS', '_CODE', '_DESCRIPTION', '_CATEGORY', '_SYSTEM')
    id_suffixes = ('_Id', '_ID', 'Id')
    reference_columns = ('ENCOUNTER', 'PROVIDER', 'ORGANIZATION', 'PAYER') 
    
    # Prepare unified DataFrame
    unified_data = []
    total_patients = len(all_patient_ids)
    
    # Process each patient
    for i, patient_id in enumerate(all_patient_ids):
        if i % 20 == 0:  # Log progress every 20 patients
            logging.info(f"Processing patient {i+1}/{total_patients}")
            
        patient_data = {'patient_id': patient_id}
        
        # Process tables in the defined order
        for table_name in table_processing_order:
            # Handle patients table separately
            if table_name == 'patients':
                if 'patients' in dataframes:
                    patient_row = dataframes['patients'][dataframes['patients']['Id'] == patient_id]
                    if not patient_row.empty:
                        for col in patient_row.columns:
                            if col != 'Id':  # Skip ID column as we already have it
                                # Convert dates properly if applicable
                                if ('patients', col) in date_columns:
                                    val = patient_row[col].iloc[0]
                                    if pd.notna(val):
                                        try:
                                            # Try to parse datetime and store as ISO format string
                                            dt = pd.to_datetime(val)
                                            patient_data[f'demographic_{col}'] = dt.isoformat()
                                        except:
                                            patient_data[f'demographic_{col}'] = val
                                    else:
                                        patient_data[f'demographic_{col}'] = None
                                else:
                                    patient_data[f'demographic_{col}'] = patient_row[col].iloc[0]
                continue
            
            # Skip tables not in grouped_dfs
            if table_name not in grouped_dfs:
                continue
                
            grouped = grouped_dfs[table_name]
            patient_col = patient_id_columns.get(table_name)
            
            if not patient_col:
                continue
                
            # Get patient's data from this table
            try:
                patient_group = grouped.get_group(patient_id)
                
                # Add summarized data from this table
                patient_data[f'{table_name}_count'] = len(patient_group)
                
                # Store the actual data as lists for each column
                # Sort columns for consistency
                cols_to_process = sorted([col for col in patient_group.columns 
                                          if col != patient_col and col != 'PATIENT_IDS'])
                
                # Special handling for observations table to better couple descriptions with values
                if table_name == 'observations':
                    # Extract data into separate parallel arrays
                    descriptions = []
                    values = []
                    types = []
                    units = []
                    
                    # Track unique observations to deduplicate
                    unique_observations = set()
                    
                    # Process each observation row
                    for idx, obs_row in patient_group.iterrows():
                        if 'DESCRIPTION' in obs_row and 'VALUE' in obs_row and pd.notna(obs_row['DESCRIPTION']):
                            # Create a key for deduplication based on description and value
                            obs_key = (str(obs_row['DESCRIPTION']), str(obs_row['VALUE'] if pd.notna(obs_row['VALUE']) else 'None'))
                            
                            # Skip duplicates
                            if obs_key in unique_observations:
                                continue
                                
                            unique_observations.add(obs_key)
                            
                            descriptions.append(obs_row['DESCRIPTION'])
                            values.append(obs_row['VALUE'] if pd.notna(obs_row['VALUE']) else None)
                            types.append(obs_row['TYPE'] if 'TYPE' in obs_row and pd.notna(obs_row['TYPE']) else None)
                            units.append(obs_row['UNITS'] if 'UNITS' in obs_row and pd.notna(obs_row['UNITS']) else None)
                    
                    # Only store the combined representation to avoid redundancy
                    combined_observations = []
                    for i in range(len(descriptions)):
                        if i < len(values):  # Ensure we don't go out of bounds
                            combined_observations.append({
                                'description': descriptions[i],
                                'value': values[i],
                                'type': types[i] if i < len(types) else None,
                                'units': units[i] if i < len(units) else None
                            })
                    
                    # Store only the combined representation which contains all the needed information
                    patient_data['observations'] = combined_observations
                
                # Process all medical entities tables using a consistent pattern
                elif table_name in ['conditions', 'medications', 'procedures', 'immunizations', 'allergies', 'devices', 'careplans']:
                    # Track unique descriptions to deduplicate
                    unique_descriptions = set()
                    descriptions = []
                    
                    for idx, row in patient_group.iterrows():
                        if 'DESCRIPTION' in row and pd.notna(row['DESCRIPTION']):
                            description = row['DESCRIPTION']
                            
                            # Skip if we've already seen this description
                            if description in unique_descriptions:
                                continue
                                
                            unique_descriptions.add(description)
                            descriptions.append(description)
                    
                    # Store using a simplified naming pattern
                    if descriptions:  # Only add if there are items
                        patient_data[table_name] = descriptions
                
                # Keep demographic information (patient table) as is - it will be processed later
                elif table_name == 'patients':
                    # Process all important patient demographics - keeping PII as requested
                    for col in cols_to_process:
                        if col in patient_group.columns and pd.notna(patient_group[col].iloc[0]):
                            patient_data[f'demographic_{col}'] = patient_group[col].iloc[0]
                
                # Process encounters table - focus on types of care received
                elif table_name == 'encounters':
                    # Track unique encounter types to deduplicate
                    unique_encounters = set()
                    encounters = []
                    
                    # Also track encounter classes, which are important for medical context
                    encounter_classes = set()
                    
                    for idx, encounter_row in patient_group.iterrows():
                        # Store encounter types
                        if 'DESCRIPTION' in encounter_row and pd.notna(encounter_row['DESCRIPTION']):
                            encounter_desc = encounter_row['DESCRIPTION']
                            if encounter_desc not in unique_encounters:
                                unique_encounters.add(encounter_desc)
                                encounters.append(encounter_desc)
                        
                        # Store encounter classes
                        if 'ENCOUNTERCLASS' in encounter_row and pd.notna(encounter_row['ENCOUNTERCLASS']):
                            encounter_class = encounter_row['ENCOUNTERCLASS']
                            encounter_classes.add(encounter_class)
                    
                    # Store using consistent naming pattern
                    if encounters:  # Only add if there are items
                        patient_data['encounters'] = encounters
                    
                    # Also include encounter classes for medical context
                    if encounter_classes:
                        patient_data['encounter_classes'] = list(encounter_classes)
            except KeyError:
                # This patient doesn't have data in this table
                patient_data[f'{table_name}_count'] = 0
        
        unified_data.append(patient_data)
        
        # Periodically clean up memory (Python's garbage collector)
        if i % 50 == 0 and i > 0:
            gc.collect()
    
    # Convert list of dictionaries to DataFrame
    logging.info("Creating unified DataFrame...")
    unified_df = pd.DataFrame(unified_data)
    
    # Report final dataframe size
    memory_usage = unified_df.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert to MB
    logging.info(f"Unified DataFrame size: {memory_usage:.2f} MB")
    
    return unified_df

def include_ungrouped_tables(unified_df, dataframes, grouped_dfs):
    """Include tables that couldn't be grouped by patient ID.
    
    Args:
        unified_df (pd.DataFrame): Unified patient data
        dataframes (dict): Original DataFrames
        grouped_dfs (dict): Grouped DataFrames
        
    Returns:
        pd.DataFrame: Enhanced unified patient data
    """
    # As per user request, we're only focusing on patient health information
    # Administrative tables are already excluded, so we'll just check for any remaining ones
    
    # Find tables that weren't grouped
    ungrouped_tables = [name for name in dataframes if name not in grouped_dfs and name != 'patients']
    
    if ungrouped_tables:
        logging.warning(f"The following tables could not be linked to patients: {ungrouped_tables}")
        logging.info("Skipping ungrouped tables to focus only on patient health information")
    
    return unified_df

def main():
    parser = argparse.ArgumentParser(description='Preprocess Synthea dataset')
    parser.add_argument('--input-dir', type=str, default='dataset/synthea-dataset-100/set100/csv',
                        help='Path to directory containing CSV files')
    parser.add_argument('--output-file', type=str, default='dataset/synthea-unified.parquet',
                        help='Path to output parquet file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--info-file', type=str, default='',
                        help='Optional path to save dataset information as JSON')
    parser.add_argument('--exclude-admin', action='store_true', default=True, 
                        help='Exclude administrative tables (payers, organizations, providers)')
    parser.add_argument('--sample', type=int, default=0,
                        help='Process only a sample of patients (0 for all patients)')
    args = parser.parse_args()
    
    # Track processing time
    start_time = datetime.now()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Get absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_file = os.path.abspath(args.output_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process data
    logger.info(f"Reading CSV files from {input_dir}...")
    dataframes = read_csv_files(input_dir)
    
    if not dataframes:
        logger.error("No CSV files found or could be read")
        return
    
    # Remove administrative tables if requested
    admin_tables = ['payers', 'organizations', 'providers']
    if args.exclude_admin:
        for table in admin_tables:
            if table in dataframes:
                logger.info(f"Excluding administrative table: {table}")
                dataframes.pop(table)
    
    # Check for date columns and convert appropriate formats
    for name, df in dataframes.items():
        date_cols = [col for col in df.columns if col.upper() in ('START', 'STOP', 'DATE', 'BIRTHDATE', 'DEATHDATE')]
        if date_cols:
            logger.info(f"Detected potential date columns in {name}: {date_cols}")
    
    logger.info(f"Found {len(dataframes)} tables after filtering")
    
    # Inspect data
    if args.verbose:
        inspect_dataframes(dataframes)
    
    # Identify relationships between tables
    logger.info("Identifying relationships between tables...")
    relationships = identify_relationships(dataframes)
    
    # Group tables by patient where possible
    logger.info("Grouping data by patient...")
    group_result = group_by_patient(dataframes)
    
    # Process tables without direct patient IDs
    logger.info("Processing tables without direct patient IDs...")
    group_result = process_tables_without_direct_patient_id(dataframes, group_result, relationships)
    
    # Join patient data
    logger.info("Joining patient data...")
    unified_df = join_patient_data(dataframes, group_result)
    
    # Include any remaining ungrouped tables as global features
    logger.info("Including ungrouped tables as global features...")
    unified_df = include_ungrouped_tables(unified_df, dataframes, group_result['grouped_dfs'])
    
    # Print columns for review
    logger.info(f"Final dataset has {len(unified_df)} rows and {len(unified_df.columns)} columns")
    if args.verbose:
        logger.info(f"Columns: {unified_df.columns.tolist()}")
    
    # Create dataset information if requested
    if args.info_file:
        import json
        info_file = os.path.abspath(args.info_file)
        logger.info(f"Saving dataset information to {info_file}")
        
        # Create dataset information
        dataset_info = {
            'num_patients': len(unified_df),
            'num_columns': len(unified_df.columns),
            'columns': {
                # Group columns by table
                table: [col for col in unified_df.columns if col.startswith(f"{table}_") or 
                       (table == 'patients' and col.startswith("demographic_"))]
                for table in list(dataframes.keys()) + ['patient']
            },
            'tables': {
                table: {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': df.columns.tolist()
                }
                for table, df in dataframes.items()
            }
        }
        
        # Save to JSON
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    # Remove any columns that only contain null values
    null_columns = [col for col in unified_df.columns if unified_df[col].isna().all()]
    if null_columns:
        logger.info(f"Removing {len(null_columns)} columns that contain only null values")
        unified_df = unified_df.drop(columns=null_columns)
    
    # Save to parquet with compression
    logger.info(f"Saving unified data to {output_file}...")
    unified_df.to_parquet(output_file, index=False, compression='snappy')
    
    # Log details about the output
    logger.info(f"Successfully saved unified data with {len(unified_df)} patients and {len(unified_df.columns)} columns")
    
    # For large datasets, dump column names to a JSON file for easier analysis
    if args.info_file:
        import json
        with open(args.info_file, 'w') as f:
            json.dump({
                'patient_count': len(unified_df),
                'column_count': len(unified_df.columns),
                'columns': unified_df.columns.tolist(),
                'memory_usage_mb': unified_df.memory_usage(deep=True).sum() / (1024 * 1024),
                'processing_time': str(datetime.now() - start_time)
            }, f, indent=2)
        logger.info(f"Saved dataset information to {args.info_file}")

if __name__ == '__main__':
    main()