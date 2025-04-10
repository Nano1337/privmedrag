"""
Configuration file for Medical MCQ Generation System
"""

# Paths
DATA_PATH = "../dataset"
OUTPUT_PATH = "./output"
TEMPLATES_PATH = "./templates"

# Question Type Definitions
QUESTION_TYPES = {
    "CLINICAL_INTERPRETATION": {
        "description": "Interpreting clinical values in context",
        "template": "My {measure_type} values show {measure_value}. What does this indicate about my {condition}?",
        "knowledge_required": "Clinical interpretation standards",
        "weight": 0.25
    },
    "TREATMENT_REASONING": {
        "description": "Understanding treatment rationales",
        "template": "Given my diagnosis of {condition} and {clinical_factor}, why might my doctor have prescribed {medication}?",
        "knowledge_required": "Treatment guidelines and mechanisms",
        "weight": 0.25
    },
    "RISK_ASSESSMENT": {
        "description": "Predicting health risks",
        "template": "Based on my {condition_1}, {condition_2}, and {social_factor}, which complication am I at highest risk for?",
        "knowledge_required": "Risk factors and complication patterns",
        "weight": 0.2
    },
    "MECHANISM_INTEGRATION": {
        "description": "Understanding how treatments work together",
        "template": "My medications include {medication_1} and {medication_2}. How do these work together to address my {condition}?",
        "knowledge_required": "Pharmacological mechanisms and interactions",
        "weight": 0.15
    },
    "SYMPTOM_PROGRESSION": {
        "description": "Understanding disease trajectories",
        "template": "My {measure_type} has changed from {value_initial} to {value_current} over {time_period}. What does this suggest about my {condition} progression?",
        "knowledge_required": "Disease progression patterns",
        "weight": 0.15
    }
}

# Number of questions to generate
NUM_QUESTIONS = 500

# Number of options per question (including correct answer)
NUM_OPTIONS = 4
