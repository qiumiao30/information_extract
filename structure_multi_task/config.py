# config.py
"""
Configuration file - Stores all configuration information
"""

# Model configuration
MODEL_CONFIG = {
    "model_name": "****Model Path or Model Name*****",  # deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    "torch_dtype": "float16",
    "device_map": "auto",
    "generation_config": {
        "max_new_tokens": 10,
        "temperature": 0.1,
        "do_sample": True,
    }
}

# Relation label mapping
RELATION_LABELS = {
    'CPR:1': 'UPREGULATOR',
    'CPR:2': 'DIRECT-REGULATOR',
    'CPR:3': 'INDIRECT-UPREGULATOR', 
    'CPR:4': 'INDIRECT-DOWNREGULATOR',
    'CPR:5': 'AGONIST',
    'CPR:6': 'ANTAGONIST',
    'CPR:7': 'SUBSTRATE',
    'CPR:9': 'PRODUCT-OF',
    'CPR:10': 'BIOMARKER',
    'false': 'no relation'
}

# Entity label ID mapping ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA', 'B-cell_line', 
#                          'I-cell_line', 'B-cell_type', 'I-cell_type', 'B-protein', 'I-protein']
RELATION_LABELS_IDS = {
    'O': 0,
    'B-DNA': 1,
    'I-DNA': 2,
    'B-RNA': 3,
    'I-RNA': 4,
    'B-cell_line': 5,
    'I-cell_line': 6,
    'B-cell_type': 7,
    'I-cell_type': 8,
    'B-protein': 9,
    'I-protein': 10
}      

# Evaluation configuration
EVALUATION_CONFIG = {
    "max_samples": 20,
    "metrics": ["precision", "recall", "f1", "accuracy"],
    "average_methods": ["macro", "micro"]
}

# Dataset configuration
DATASET_CONFIG = {
    "chemprot": {
        "name": "bigbio/chemprot",
        "split": "test"
    },
    "gad": {
        "name": "bigbio/gad",
        "split": "test"
    },
}
