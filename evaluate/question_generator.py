import os
import sys
import random
import json
import numpy as np
import pandas as pd
import dgl
import torch
from typing import Dict, List, Any, Optional, Tuple, Set

from openai import OpenAI
import ast # For safely evaluating string representations of lists

# Import PrimeKG dataset class and vector search
from rgl.datasets.primekg import PrimeKGDataset
from rgl.node_retrieval.vector_search import VectorSearchEngine

# --- Constants ---
DEFAULT_MODEL = "gpt-4o" # Or whichever model you prefer/have access to

# Basic structure for question types (can be expanded)
DEFAULT_QUESTION_TYPES = {
    "diagnosis": {"description": "Determine the most likely diagnosis based on the patient's presentation."},
    "treatment_selection": {"description": "Select the most appropriate initial treatment or management step."},
    "side_effects": {"description": "Identify potential side effects or complications of the patient's current medications or conditions."},
    "drug_interaction": {"description": "Identify potential interactions between the patient's medications."},
    "mechanism_of_action": {"description": "Explain the mechanism of action of a relevant drug."},
    "prognosis": {"description": "Assess the likely prognosis or course of the patient's condition."},
    "monitoring": {"description": "Determine appropriate monitoring parameters for the patient's condition or treatment."}
}

class QuestionGenerator:
    """
    Generates Multiple Choice Questions (MCQs) for medical cases using LLMs,
    optionally integrating knowledge graph context from PrimeKG.
    """
    def __init__(self, question_types: Optional[Dict] = None, primekg_path: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initializes the QuestionGenerator.

        Args:
            question_types: Dictionary of question types with descriptions.
            primekg_path: Optional path to the PrimeKG dataset directory (e.g., containing 'kg.pt').
                          If None, KG features will be disabled.
            openai_api_key: Optional OpenAI API key. If None, attempts to read from
                            the OPENAI_API_KEY environment variable.
        """
        self.model = DEFAULT_MODEL
        self.question_types = question_types or DEFAULT_QUESTION_TYPES

        # --- Setup OpenAI ---
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # --- Setup PrimeKG (Optional) ---
        self.primekg_path = primekg_path
        self.graph = None
        self.raw_ndata = None
        self.raw_edata = None
        self.node_names = None
        self.node_categories = None
        self.relation_df = None

        # Cache for KG search results by patient ID
        self.kg_cache = {}

        if self.primekg_path:
            try:
                self._load_primekg_data()
                print(f"Successfully loaded PrimeKG data from: {self.primekg_path}")
            except FileNotFoundError:
                print(f"Error: PrimeKG directory or required files not found at {self.primekg_path}. Disabling KG features.")
                self._disable_kg_features()
            except Exception as e:
                print(f"Error loading PrimeKG data: {e}. Disabling KG features.")
                self._disable_kg_features()
        else:
            print("PrimeKG path not provided. Knowledge graph features will be disabled.")
            self._disable_kg_features() # Explicitly disable

        if not api_key:
            print("Warning: OPENAI_API_KEY environment variable not set. LLM calls will likely fail.")

    def _load_primekg_data(self):
        """Load PrimeKG data from the specified path."""
        try:
            print(f"Loading PrimeKG dataset from {self.primekg_path}...")
            
            # Use the PrimeKGDataset class to load the data
            dataset = PrimeKGDataset(self.primekg_path)
            self.graph = dataset.graph
            
            # Extract node names and types from raw_ndata
            self.node_names = dataset.raw_ndata['name']
            self.node_types = dataset.raw_ndata['category']
            
            # Create a relation dataframe for easier lookup
            edges = self.graph.edges()
            
            # Get relation types from raw_edata
            edge_relation_types = dataset.raw_edata['relation_type']
            edge_display_relations = dataset.raw_edata['display_relation']
            
            # Convert to DataFrame for easier filtering
            self.relation_df = pd.DataFrame({
                'x_index': edges[0].tolist(),
                'y_index': edges[1].tolist(),
                'relation': edge_relation_types,
                'display_relation': edge_display_relations
            })
            
            print(f"Successfully loaded PrimeKG graph with {len(self.node_names)} nodes and {len(self.relation_df)} edges")
            print(f"Successfully loaded PrimeKG data from: {self.primekg_path}")
            return True
        except Exception as e:
            print(f"Error loading PrimeKG data: {e}")
            # Disable KG features on error
            self.graph = None
            self.node_names = None
            self.node_types = None
            self.relation_df = None
            return False
            
    def _format_relation(self, relation):
        """Format a relation type into a more readable display format.
        
        Args:
            relation: The relation type string from the knowledge graph.
            
        Returns:
            A formatted, human-readable version of the relation.
        """
        # Convert snake_case or CamelCase to spaces and capitalize first letter
        if isinstance(relation, str):
            # Replace underscores with spaces
            formatted = relation.replace('_', ' ')
            # Add spaces before capital letters in CamelCase
            formatted = ''.join([' ' + c if c.isupper() else c for c in formatted]).strip()
            # Capitalize first letter
            formatted = formatted.capitalize()
            return formatted
        else:
            return "related to"  # Default fallback

    def _disable_kg_features(self):
        """Explicitly disables KG features by nullifying related attributes."""
        self.primekg_dataset = None
        self.graph = None
        self.raw_ndata = None
        self.raw_edata = None
        self.node_names = None
        self.node_categories = None
        self.relation_df = None
        print("Knowledge Graph features are disabled.")

    def setup_model(self, model_name: str):
        """Configure the LLM model to use for generation."""
        self.model = model_name
        print(f"LLM model set to: {self.model}")

    def generate_mcq(self, patient: Dict[str, Any], question_type: str) -> Optional[Dict[str, Any]]:
        """
        Generate a single MCQ for a specific question type, leveraging LLM and KG context if available.

        Args:
            patient: Dictionary containing patient data.
            question_type: The specific question type to generate.

        Returns:
            A dictionary representing the generated MCQ, or None if generation fails.
        """
        patient_id = patient.get('patient_id', 'unknown')
        
        # Extract core patient information needed for the question
        try:
            patient_info = self._extract_patient_info(patient)
            if not patient_info or not (patient_info['conditions'] or patient_info['medications']):
                print(f"Cannot generate MCQ for patient {patient_id} due to insufficient core data.")
                return None
        except Exception as e:
            print(f"Error extracting info for patient {patient_id}: {e}")
            return None

        try:
            print(f"Generating '{question_type}' MCQ for patient {patient_id}...")

            # 1. Find relevant KG context (if KG is enabled)
            kg_context = ""
            if self.graph is not None:
                # Check if we already have cached KG results for this patient
                if patient_id in self.kg_cache:
                    print(f"Using cached KG search results for patient {patient_id}")
                    related_entities_info = self.kg_cache[patient_id]
                    kg_context = self._format_kg_context(related_entities_info)
                else:
                    # Identify key entities from patient data to query in KG
                    # Prioritize conditions and medications as starting points
                    query_entities = patient_info['conditions'][:3] + patient_info['medications'][:2] # Limit number for performance
                    if query_entities:
                        print(f"Searching for matching entities in PrimeKG...")
                        related_entities_info = self._find_related_kg_entities(query_entities)
                        # Cache the results for future questions about this patient
                        self.kg_cache[patient_id] = related_entities_info
                        kg_context = self._format_kg_context(related_entities_info)
                    else:
                        print("No specific conditions/medications to query in KG.")

            # 2. Generate MCQ using LLM, potentially with KG context
            mcq = self._generate_mcq_with_llm(patient_info, question_type, kg_context)

            if mcq:
                mcq["patient_id"] = patient_id
                mcq["question_type"] = question_type
                mcq["kg_context_used"] = bool(kg_context) # Track if KG was used
                return mcq
            else:
                print(f"Failed to generate '{question_type}' for patient {patient_id}.")
                return None

        except Exception as e:
            print(f"Error generating '{question_type}' question for patient {patient_id}: {e}")
            return None

    def generate_mcqs_for_patient(self, patient: Dict[str, Any], applicable_types: List[str], questions_per_type: int = 1) -> List[Dict[str, Any]]:
        """
        Generate MCQs for a patient, leveraging LLM and KG context if available.

        Args:
            patient: Dictionary containing patient data.
            applicable_types: List of question types applicable to this patient.
            questions_per_type: Number of questions to generate per question type (default: 1).

        Returns:
            List of generated MCQ dictionaries.
        """
        patient_id = patient.get('patient_id', 'unknown')
        generated_mcqs = []

        # Extract core patient information needed for all questions
        try:
            patient_info = self._extract_patient_info(patient)
            if not patient_info or not (patient_info['conditions'] or patient_info['medications']):
                 print(f"Skipping patient {patient_id} due to insufficient core data (conditions/medications).")
                 return [] # Skip patient if essential data is missing
        except Exception as e:
            print(f"Error extracting info for patient {patient_id}: {e}. Skipping patient.")
            return []

        # Generate MCQs for each applicable question type, limited by questions_per_type
        # Shuffle the types to get a good mix if we're limiting the total
        import random
        random.shuffle(applicable_types)
        
        # Limit to the specified number of question types
        selected_types = applicable_types[:questions_per_type]
        
        for question_type in selected_types:
            try:
                print(f"Attempting to generate '{question_type}' for patient {patient_id}...")

                # 1. Find relevant KG context (if KG is enabled)
                kg_context = ""
                if self.graph is not None:
                    # Identify key entities from patient data to query in KG
                    # Prioritize conditions and medications as starting points
                    query_entities = patient_info['conditions'][:3] + patient_info['medications'][:2] # Limit number for performance
                    if query_entities:
                         related_entities_info = self._find_related_kg_entities(query_entities)
                         kg_context = self._format_kg_context(related_entities_info)
                    else:
                         print("No specific conditions/medications to query in KG.")


                # 2. Generate MCQ using LLM, potentially with KG context
                mcq = self._generate_mcq_with_llm(patient_info, question_type, kg_context)

                if mcq:
                    mcq["patient_id"] = patient_id
                    mcq["question_type"] = question_type
                    mcq["kg_context_used"] = bool(kg_context) # Track if KG was used
                    generated_mcqs.append(mcq)
                    print(f"Successfully generated '{question_type}' for patient {patient_id}.")
                else:
                     print(f"Failed to generate '{question_type}' for patient {patient_id}.")

            except Exception as e:
                print(f"Error generating '{question_type}' question for patient {patient_id}: {e}")
                # Optionally, log the error in more detail

        return generated_mcqs

    def _extract_patient_info(self, patient: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract and structure relevant patient information for MCQ generation.

        Returns:
             Dict with keys like 'demographics', 'conditions', 'observations', 'medications', or None if critical data is missing.
        """
        try:
            # Basic demographics - Use .get with defaults for robustness
            birthdate = patient.get("demographic_BIRTHDATE", "")
            demographics = {
                "age": self._calculate_age(birthdate),
                "gender": patient.get("demographic_GENDER", "Unknown"),
                "race": patient.get("demographic_RACE", "Unknown"),
                "ethnicity": patient.get("demographic_ETHNICITY", "Unknown")
            }

            # Helper to safely parse list-like string columns if needed
            def parse_list_column(data):
                if isinstance(data, list): return data # Already a list
                if isinstance(data, str):
                    try:
                        # Handle potential string representations of lists
                        # Basic check for list-like structure
                        if data.startswith('[') and data.endswith(']'):
                             # Be cautious with eval, prefer safer methods if format is known
                             # Using ast.literal_eval is safer than eval
                             # import ast # Imported at top level now
                             parsed_list = ast.literal_eval(data)
                             if isinstance(parsed_list, list):
                                 return parsed_list
                    except (ValueError, SyntaxError):
                        # If parsing fails, treat as a single-item list if non-empty
                        return [data] if data else []
                # If not list or parsable string, return empty list
                return []

            # Extract conditions - Handle potential string formats
            raw_conditions = patient.get('conditions')
            conditions = parse_list_column(raw_conditions)
            # Further clean up - extract description if items are dicts
            cleaned_conditions = []
            for cond in conditions:
                 if isinstance(cond, dict) and 'description' in cond:
                     cleaned_conditions.append(cond['description'])
                 elif isinstance(cond, str) and cond: # Ensure non-empty strings
                     cleaned_conditions.append(cond)
            conditions = cleaned_conditions

            # Extract observations (simplified: just descriptions and values)
            raw_observations = patient.get('observations')
            observations = parse_list_column(raw_observations)
            structured_obs = []
            for obs in observations:
                 if isinstance(obs, dict):
                      desc = obs.get('description', 'Unknown Observation')
                      val = obs.get('value', 'N/A')
                      units = obs.get('units', '')
                      structured_obs.append(f"{desc}: {val} {units}".strip())
                 elif isinstance(obs, str) and obs:
                      structured_obs.append(obs) # Assume string is descriptive enough


            # Extract medications - Handle potential string formats
            raw_medications = patient.get('medications')
            medications = parse_list_column(raw_medications)
            # Further clean up
            cleaned_meds = []
            for med in medications:
                 if isinstance(med, dict) and 'description' in med:
                     cleaned_meds.append(med['description'])
                 elif isinstance(med, str) and med:
                     cleaned_meds.append(med)
            medications = cleaned_meds

            # Return structured info
            return {
                "demographics": demographics,
                "conditions": conditions,
                "observations": structured_obs, # List of strings now
                "medications": medications
            }
        except Exception as e:
             print(f"Error during patient info extraction: {e}")
             return None # Indicate failure

    def _calculate_age(self, birthdate: str) -> str:
        """Calculate age from birthdate string (YYYY-MM-DD or just YYYY)."""
        if not birthdate or not isinstance(birthdate, str):
            return "Unknown"

        try:
            # Handle full date or just year
            if '-' in birthdate:
                birth_year = int(birthdate.split('-')[0])
            else:
                birth_year = int(birthdate)

            # Use a fixed current year for consistency/reproducibility if needed,
            # otherwise use the actual current year.
            # from datetime import datetime
            # current_year = datetime.now().year
            current_year = 2024 # Fixed year for example

            if 1900 < birth_year <= current_year: # Basic sanity check
                age = current_year - birth_year
                return str(age)
            else:
                 return "Invalid Year"
        except ValueError:
            # Handle cases where conversion to int fails
             return "Unknown"
        except Exception:
             # Catch-all for other potential errors (network issues, etc.)
             return "Unknown"

    def _find_related_kg_entities(self, query_entities: List[str], n_hops: int = 1, max_neighbors_per_entity: int = 10) -> Dict[str, List[Dict]]:
        """
        Find entities in PrimeKG related to the given query entities (e.g., patient conditions/medications).

        Args:
            query_entities: List of entity names (strings) to search for in the KG.
            n_hops: Number of hops to traverse from the initial entities (1-hop default).
            max_neighbors_per_entity: Max neighbors to consider for each initial query entity found.

        Returns:
            A dictionary where keys are the original query entity names and values are lists
            of dictionaries, each representing a related entity and the relationship.
            Example: {'Aspirin': [{'related_entity': 'Bleeding', 'relation': 'may_cause', 'direction': '->'}]}
        """
        if self.graph is None or self.node_names is None or self.relation_df is None:
            print("KG features disabled or not loaded, skipping KG entity search.")
            return {}

        # Filter out non-healthcare related terms
        non_medical_terms = [
            "higher education", "education", "employment", "full-time", "part-time", 
            "transportation", "access", "finding", "received", "lack of"
        ]
        
        # Only keep healthcare-related entities
        filtered_entities = []
        for entity in query_entities:
            # Skip entities that match non-medical terms
            if any(term in entity.lower() for term in non_medical_terms):
                continue
                
            # Remove common non-medical qualifiers
            clean_entity = entity
            for suffix in [" (finding)", " (disorder)", " (procedure)"]:
                if clean_entity.endswith(suffix):
                    clean_entity = clean_entity[:-len(suffix)]
            
            filtered_entities.append(clean_entity)
            
        if not filtered_entities:
            print("No healthcare-related entities found for KG search.")
            return {}
            
        results = {query: [] for query in filtered_entities}
        found_node_indices = set()
        query_entity_map = {query: [] for query in filtered_entities}

        # Find matching entities using category-based search and name similarity
        print("Searching for matching entities in PrimeKG...")
        
        # Get relevant medical categories
        medical_categories = ['drug', 'disease', 'symptom', 'chemical', 'gene', 'protein', 'anatomy']
        
        # For each query entity, try to find matches in the knowledge graph
        for query_entity in filtered_entities:
            clean_query = query_entity.lower().strip()
            found_match = False
            
            # 1. Try exact name match first (highest priority)
            exact_matches = []
            for i, name in enumerate(self.node_names):
                if name.lower() == clean_query:
                    exact_matches.append(i)
            
            if exact_matches:
                idx = exact_matches[0]  # Take the first exact match
                query_entity_map[query_entity].append(idx)
                found_node_indices.add(idx)
                print(f"KG Exact Match: '{query_entity}' -> '{self.node_names[idx]}'")
                found_match = True
                continue
            
            # 2. Try medical term matching with category filtering
            # This is more accurate than simple substring matching
            category_matches = []
            
            # First, identify potential matches by significant word overlap
            query_words = set(clean_query.split())
            if len(query_words) >= 1:
                for i, name in enumerate(self.node_names):
                    # Skip if node category doesn't match medical categories
                    node_category = self.node_types[i].lower()
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
                query_entity_map[query_entity].append(idx)
                found_node_indices.add(idx)
                print(f"KG {match_type.title()} Match: '{query_entity}' -> '{self.node_names[idx]}' (Score: {score:.2f})")
                found_match = True
            
            if not found_match:
                print(f"KG No Match: '{query_entity}' not found in PrimeKG node names.")


        if not found_node_indices:
            return {} # No query entities found in the graph

        # 2. Perform graph traversal using DGL's efficient neighbor sampling
        # This is much faster than iterating through the entire relation DataFrame
        
        collected_relations: Dict[int, List[Tuple[str, int, str]]] = {idx: [] for idx in found_node_indices}
        
        print(f"Querying relations for {len(found_node_indices)} matched KG entities...")
        
        # Use DGL's efficient neighbor sampling instead of iterating through all edges
        for node_idx in found_node_indices:
            # Get outgoing edges (much faster than DataFrame iteration)
            out_edges = self.graph.out_edges(node_idx)
            if isinstance(out_edges, tuple) and len(out_edges) == 2:
                src_nodes, dst_nodes = out_edges
                
                # Limit to max_neighbors_per_entity
                max_out = min(len(dst_nodes), max_neighbors_per_entity // 2)
                
                for i in range(max_out):
                    try:
                        dst = dst_nodes[i].item() if hasattr(dst_nodes[i], 'item') else int(dst_nodes[i])
                        # Find the relation type from the edge data
                        # This is more efficient than searching the entire DataFrame
                        edge_id = self.graph.edge_ids(node_idx, dst)
                        
                        # Handle different return types from edge_ids
                        if hasattr(edge_id, 'numel') and edge_id.numel() > 0:
                            # Get relation type from edge data
                            if 'type' in self.graph.edata:
                                rel_type_id = self.graph.edata['type'][edge_id].item() if hasattr(edge_id, 'item') else int(edge_id)
                                # Convert to readable format
                                rel_display = f"relation_{rel_type_id}"
                                collected_relations[node_idx].append((rel_display, dst, '->'))
                        elif isinstance(edge_id, int) and edge_id >= 0:
                            # Handle case where edge_id is an integer
                            if 'type' in self.graph.edata:
                                rel_type_id = self.graph.edata['type'][edge_id] if isinstance(self.graph.edata['type'], list) else 0
                                rel_display = f"relation_{rel_type_id}"
                                collected_relations[node_idx].append((rel_display, dst, '->'))
                    except Exception as e:
                        # Skip this edge if there's an error
                        continue
            
            # Get incoming edges
            in_edges = self.graph.in_edges(node_idx)
            if isinstance(in_edges, tuple) and len(in_edges) == 2:
                src_nodes, dst_nodes = in_edges
                
                # Limit to max_neighbors_per_entity
                max_in = min(len(src_nodes), max_neighbors_per_entity // 2)
                
                for i in range(max_in):
                    try:
                        src = src_nodes[i].item() if hasattr(src_nodes[i], 'item') else int(src_nodes[i])
                        # Find the relation type
                        edge_id = self.graph.edge_ids(src, node_idx)
                        
                        # Handle different return types from edge_ids
                        if hasattr(edge_id, 'numel') and edge_id.numel() > 0:
                            # Get relation type from edge data
                            if 'type' in self.graph.edata:
                                rel_type_id = self.graph.edata['type'][edge_id].item() if hasattr(edge_id, 'item') else int(edge_id)
                                # Convert to readable format
                                rel_display = f"relation_{rel_type_id}"
                                collected_relations[node_idx].append((rel_display, src, '<-'))
                        elif isinstance(edge_id, int) and edge_id >= 0:
                            # Handle case where edge_id is an integer
                            if 'type' in self.graph.edata:
                                rel_type_id = self.graph.edata['type'][edge_id] if isinstance(self.graph.edata['type'], list) else 0
                                rel_display = f"relation_{rel_type_id}"
                                collected_relations[node_idx].append((rel_display, src, '<-'))
                    except Exception as e:
                        # Skip this edge if there's an error
                        continue


        # 3. Format results mapping back to original query names
        for query_entity, node_indices in query_entity_map.items():
            for node_idx in node_indices: # Should typically be one index per query now
                if node_idx in collected_relations:
                     for relation, related_node_idx, direction in collected_relations[node_idx]:
                         try:
                             related_entity_name = self.node_names[related_node_idx]
                             results[query_entity].append({
                                 "related_entity": related_entity_name,
                                 "relation": relation,
                                 "direction": direction,
                                 "related_entity_category": self.node_categories[related_node_idx] if self.node_categories is not None else "Unknown"
                             })
                         except IndexError:
                             print(f"Warning: Index {related_node_idx} out of bounds for node names/categories.")
                         except Exception as e:
                              print(f"Warning: Error processing relation for node {node_idx}: {e}")

        # Clean up empty entries
        results = {k: v for k, v in results.items() if v}
        print(f"Found {sum(len(v) for v in results.values())} KG relations for query entities.")
        return results

    def _format_kg_context(self, related_entities_info: Dict[str, List[Dict]]) -> str:
        """
        Format the found KG relationships into a string for the LLM prompt.

        Args:
            related_entities_info: The output from _find_related_kg_entities.

        Returns:
            A formatted string summarizing the KG context, or empty string if no info.
        """
        if not related_entities_info:
            return ""

        context_str = "Knowledge Graph Context:\n"
        for query_entity, relations in related_entities_info.items():
            if relations:
                context_str += f"- Context for '{query_entity}':\n"
                # Group by relation type for clarity
                relations_by_type: Dict[str, List[str]] = {}
                for rel_info in relations:
                    relation = rel_info['relation']
                    related_entity = rel_info['related_entity']
                    direction = rel_info['direction']
                    # Simple formatting of the relationship
                    rel_str = f"{related_entity}" # Start with the related entity
                    if direction == '<-':
                         rel_str = f"{related_entity} {relation} {query_entity}" # Incoming
                    else: # direction == '->' or default
                         rel_str = f"{query_entity} {relation} {related_entity}" # Outgoing

                    if relation not in relations_by_type:
                         relations_by_type[relation] = []
                    relations_by_type[relation].append(f"  - {rel_str} (Category: {rel_info.get('related_entity_category', 'Unknown')})")

                # Append grouped relations to the main string
                for relation, rel_strs in relations_by_type.items():
                     # context_str += f"  Relations of type '{relation}':\n" # Could add this header
                     context_str += "\n".join(rel_strs) + "\n"
                context_str += "\n" # Add space between query entities

        return context_str.strip()

    def _generate_mcq_with_llm(self, patient_info: Dict[str, Any], question_type: str, kg_context: str) -> Optional[Dict[str, Any]]:
        """
        Generate a single MCQ using an LLM call, incorporating patient data and KG context.

        Args:
            patient_info: Structured patient information.
            question_type: The type of question to generate.
            kg_context: Formatted string of relevant KG information.

        Returns:
            A dictionary representing the parsed and processed MCQ, or None on failure.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OpenAI API key not configured. Cannot generate MCQ.")
            return None

        # Construct the prompt
        prompt = self._construct_mcq_prompt(patient_info, question_type, kg_context)
        system_prompt = "You are a medical expert generating challenging, clinically relevant multiple-choice questions (MCQs) for medical education. Integrate the provided patient-specific data and knowledge graph context (if available) to create questions that require reasoning, not just recall. Ensure the distractors are plausible but definitively incorrect."

        print(f"--- Generating MCQ ---")
        # print(f"System Prompt: {system_prompt}") # Optional: Log prompts for debugging
        # print(f"User Prompt:\n{prompt}") # Optional: Log prompts for debugging

        try:
            # Call the LLM using the modern client-based API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6, # Slightly lower temp for more focused output
                max_tokens=1024, # Adjust as needed
                response_format={"type": "json_object"} # Request JSON output if using compatible models
            )

            # Extract and parse the response
            mcq_text = response.choices[0].message.content.strip()
            # print(f"LLM Raw Response:\n{mcq_text}") # Optional: Log raw response
            return self._parse_mcq_response(mcq_text, patient_info, question_type)

        except Exception as e:
            print(f"An error occurred during the OpenAI API call: {e}")
            # Optional: Add more detailed logging or re-raise if needed
            return None # Or raise the exception: raise e

    def _parse_mcq_response(self, mcq_response_text: str, patient_info: Dict, question_type: str) -> Optional[Dict]:
        """
        Parse the JSON response from the LLM to extract MCQ details.

        Args:
            mcq_response_text: The raw text response from the LLM, expected to be JSON.
            patient_info: Used potentially as fallback if parsing fails completely.
            question_type: For context if creating a fallback.

        Returns:
            A validated and processed MCQ dictionary, or None if parsing/validation fails.
        """
        try:
            # Basic cleanup - LLM might sometimes add backticks or "json" prefix
            if mcq_response_text.startswith("```json"):
                mcq_response_text = mcq_response_text[7:]
            if mcq_response_text.endswith("```"):
                mcq_response_text = mcq_response_text[:-3]
            mcq_response_text = mcq_response_text.strip()

            # Parse the JSON string
            mcq_data = json.loads(mcq_response_text)

            # --- Validation ---
            required_fields = ['question', 'options', 'correct_answer_rationale', 'distractor_rationale', 'key_data_sources']
            if not all(field in mcq_data for field in required_fields):
                print(f"Error: Parsed JSON missing required fields. Fields found: {list(mcq_data.keys())}")
                return None # Failed validation

            if not isinstance(mcq_data['options'], list) or len(mcq_data['options']) != 4:
                print(f"Error: 'options' field is not a list of exactly 4 items. Found: {mcq_data.get('options')}")
                return None # Failed validation

            # Ensure rationale structure is roughly correct
            if not isinstance(mcq_data.get('distractor_rationale'), dict) or len(mcq_data['distractor_rationale']) != 3:
                 print(f"Warning: 'distractor_rationale' structure might be incorrect. Found: {mcq_data.get('distractor_rationale')}")
                 # Don't fail parsing for this, but log it.

            # --- Processing ---
            options = mcq_data['options']
            correct_option_text = options[0] # Per prompt instructions

            # Randomize options order
            random.shuffle(options) # Shuffle in place

            # Find the new index of the original correct answer
            try:
                # This assumes the correct answer text is unique within the options
                final_correct_index = options.index(correct_option_text)
            except ValueError:
                # This *shouldn't* happen if the LLM followed instructions, but handle defensively
                print("Error: Original correct answer text not found after shuffling options. Assigning index 0.")
                # Re-insert correct answer at a known position if lost
                if correct_option_text not in options:
                    # Replace the first option (arbitrarily)
                    options[0] = correct_option_text
                final_correct_index = options.index(correct_option_text) # Find it again


            # Prepare final output dictionary
            final_mcq = {
                "question": mcq_data['question'],
                "options": options,
                "correct_index": final_correct_index, # Store the index after shuffling
                "correct_answer": correct_option_text, # Store the text of the correct answer
                "correct_answer_rationale": mcq_data.get('correct_answer_rationale', "N/A"),
                "distractor_rationale": mcq_data.get('distractor_rationale', {}),
                "key_data_sources": mcq_data.get('key_data_sources', [])
                # Add patient_id and question_type later in generate_mcqs_for_patient
            }
        except Exception as e:
            print(f"Error parsing MCQ response: {e}")
            return None

        return final_mcq

    def _construct_mcq_prompt(self, patient_info: Dict[str, Any], question_type: str, kg_context: str) -> str:
        """
        Construct the detailed prompt for the LLM, combining patient info, KG context, and instructions.
        """
        # Format patient info sections concisely
        demo = patient_info['demographics']
        demo_text = f"Patient Profile: {demo['age']} y/o {demo['gender']} {demo['race']} {demo['ethnicity']}"
        conditions_text = "Conditions: " + (", ".join(patient_info['conditions'][:5]) if patient_info['conditions'] else "None listed")
        meds_text = "Medications: " + (", ".join(patient_info['medications'][:5]) if patient_info['medications'] else "None listed")
        obs_text = "Key Observations:\n" + ("\n".join([f"- {o}" for o in patient_info['observations'][:5]]) if patient_info['observations'] else "None listed")

        # Question type focus description
        type_focus = self.question_types.get(question_type, {}).get('description', 'general medical knowledge')

        # Construct the prompt with clear sections
        prompt = f"""
        **Task:** Generate a single, high-quality medical multiple-choice question (MCQ).

        **Patient Summary:**
        {demo_text}
        {conditions_text}
        {meds_text}
        {obs_text}

        **Knowledge Graph Context:**
        {kg_context if kg_context else "No specific KG context provided for this question."}

        **MCQ Requirements:**
        1.  **Focus:** The question MUST specifically address '{question_type}' ({type_focus}).
        2.  **Integration:** Synthesize information from the Patient Summary and Knowledge Graph Context (if provided). Do NOT simply restate facts; require clinical reasoning.
        3.  **Format:** Respond ONLY with a valid JSON object adhering strictly to the following schema:
            ```json
            {{
              "question": "string (The question text, posed clearly)",
              "options": [
                "string (Correct Answer - Placed FIRST in this list for generation)",
                "string (Plausible Distractor 1)",
                "string (Plausible Distractor 2)",
                "string (Plausible Distractor 3)"
              ],
              "correct_answer_rationale": "string (Brief explanation why the first option is correct, citing patient/KG data)",
              "distractor_rationale": {{
                "1": "string (Brief explanation why option 2 is incorrect)",
                "2": "string (Brief explanation why option 3 is incorrect)",
                "3": "string (Brief explanation why option 4 is incorrect)"
               }},
              "key_data_sources": ["string (List specific patient data points or KG facts used, e.g., 'Patient condition: Diabetes', 'KG: Drug X inhibits Enzyme Y')"]
            }}
            ```
        4.  **Content:**
            *   The `question` should be clinically relevant and challenging.
            *   The `options` list MUST contain exactly 4 strings: 1 correct answer followed by 3 distractors.
            *   The correct answer MUST be the *first* item in the `options` list.
            *   Distractors MUST be plausible but incorrect. Avoid obviously wrong answers.
            *   Provide clear `correct_answer_rationale` and `distractor_rationale`.
            *   List the specific `key_data_sources` used from the patient summary or KG context.

        **Generate the MCQ JSON now:**
        """
        return prompt.strip()