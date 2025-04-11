# Medical MCQ Generation and Evaluation Framework

This framework generates knowledge-based multiple-choice questions (MCQs) from Electronic Health Record (EHR) data for evaluating RAG (Retrieval Augmented Generation) systems versus LLM-only approaches.

## Overview

The generated MCQs focus on questions that require medical knowledge integration with patient-specific data, creating a balanced evaluation that:

1. Tests retrieval capabilities without biasing toward any specific knowledge base
2. Focuses on clinically relevant questions patients might actually ask
3. Requires integration of EHR data with external medical knowledge
4. Covers diverse question types with varying complexity

## Directory Structure

- `/data` - Storage for intermediate data files
- `/templates` - Question templates (can be expanded)
- `/output` - Generated MCQs and evaluation results

## Question Types

The framework generates five types of knowledge-enhanced questions:

1. **Clinical Interpretation** - Interpreting patient lab values and measurements
2. **Treatment Reasoning** - Understanding medication and treatment rationales
3. **Risk Assessment** - Predicting health risks based on patient profile
4. **Mechanism Integration** - Understanding how multiple treatments work together
5. **Symptom Progression** - Interpreting changes in symptoms or measurements over time

## Usage

### 1. Generate MCQs (Optional, in HF Datasets already)

```bash
python generate_mcqs.py --input_file ../dataset/synthea-info.json --output_file ./output/medical_mcqs.json --num_questions 400 --primekg_path ../dataset/primekg/ --questions_per_type 3 --checkpoint_file ./output/mcq_checkpoint.json --resume --seed 42

python convert_to_parquet.py
python upload_to_hf.py
```

### 2. Evaluate Model Performance

```bash
python evaluate/evaluate_mcq_performance.py --model gemini --gemini_model gemini-2.0-flash --output_file ./evaluate/output/level0_evals.json --seed 09052023 --num_questions 400 --privacy_level 0
```

### 3. Evaluate Model Performance with different privacy levels

To modify privacy level, add the `--privacy_level` argument to the command with choices of: 
- 0: No anonymization
- 1: Remove PII
- 2: k-anonymity and l-diversity

```bash
python evaluate/evaluate_mcq_performance.py --model gemini --gemini_model gemini-2.0-flash --output_file ./evaluate/output/level1_evals.json --seed 09052023 --num_questions 400 --privacy_level 1
```

## Implementation Details

The framework follows these steps:

1. **Data Processing** - Extract relevant clinical elements from EHR data
2. **Question Generation** - Create knowledge-enhanced questions from KG + LLM
3. **Answer Generation** - Generate correct answers and challenging distractors using LLM
4. **Quality Control** - Ensure diversity and clinical relevance
5. **Performance Evaluation** - Compare RAG vs. LLM-only performance

## Evaluation Metrics

The evaluation framework reports:

- Overall accuracy for both LLM and RAG approaches
- Accuracy breakdown by question type
- Improvement percentage from RAG integration
