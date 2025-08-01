import json
import random
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import re

from sklearn.preprocessing import MultiLabelBinarizer

# ----------------------------------------------------------------------------
# 1. Config dataset 
# ----------------------------------------------------------------------------
DATASET_CONFIG = {
    # --- (NER) ---
    "bc2gm": {  ### id; tokens-->list; ner_tags-->list[0,0,1,2,0]
        "name": "spyysalo/bc2gm_corpus",
        "type": "ner",
        "split": "train" 
    },
    "jnlpba": { ### id; rokens-->list; ner_tags-->list
        "name": "jnlpba/jnlpba",
        "type": "ner",
        "split": "train"
    },
    # --- (RE) ---
    "chemprot": {
        "name": "bigbio/chemprot",
        "type": "re",
    },
    "ddi": {
        "name": "bigbio/ddi_corpus",
        "type": "re",
    },
    "gad": {
        "name": "bigbio/gad",
        "type": "re",
        "split": "train"
    },
    # --- (QA) ---
    "pubmedqa": {
        "name": "qiaojin/PubMedQA",
        "type": "qa",
        "subset": "pqa_labeled", 
        "split": "train"
    },
    "bioasq": {
        "name": "kroshan/BioASQ",
        "type": "qa"
    },
    "NCBI-disease":{
        "name": "NCBI-disease",
        "type": "ner"
    },
    "medqa": {
        "name": "GBaker/MedQA-USMLE-4-options",
        "type": "qa",
        "split": "train"
    },
}

OUTPUT_FILE = "multitask_bio_dataset.json"

# ----------------------------------------------------------------------------
# 2. Task Processor
# ----------------------------------------------------------------------------

def process_ner_dataset(dataset, config):
    """NER"""
    print(f"Processing NER dataset: {config['name']}")
    processed_samples = []
    
    # tag_names = dataset.features['ner_tags'].feature.names
    if config['name'] == "spyysalo/bc2gm_corpus" or config['name'] == "jnlpba/jnlpba":
        for example in tqdm(dataset, desc=f"  -> {config['name']}"):
            tokens = example['tokens']
            tags = example['ner_tags']
            
            if not tokens:
                continue

            if config['name'] == "spyysalo/bc2gm_corpus":
                id2label = {
                    0: "O",
                    1: "B-DISEASE",
                    2: "I-DISEASE"
                }
                bio_tags = [id2label[tag] for tag in tags]
            elif config['name'] == "jnlpba/jnlpba":
                id2label = {
                    0: "O",
                    1: "B-DNA",
                    2: "I-DNA",
                    3: "B-RNA",
                    4: "I-RNA",
                    5: "B-cell_line",
                    6: "I-cell_line",
                    7: "B-cell_type",
                    8: "I-cell_type",
                    9: "B-protein",
                    10: "I-protein",
                }
                bio_tags = [id2label[tag] for tag in tags]
            else:
                None

            Instruction = f"[Task: NER][Domain: {config['name']}] You are a biomedical named entity recognition (NER) expert. " \
                "Please extract and label all predefined biomedical entities using the BIO format. " \
                f"Supported entity labels list (BIO format): {list(id2label.values())}. " 
                
            input_text = " ".join(tokens)
            input_text += f" Tokens: {tokens}"

            # prompt += input_text
            # bio_tags = ", ".join(bio_tags)

            processed_samples.append({
                "Instruction": Instruction,
                "input": input_text,
                "output": str(bio_tags),
                "task": "NER"
            })
    elif config['name'] == "NCBI-disease":
        data = pd.read_json("data/tokcls/NCBI-disease_hf/train.json", lines=True)

        id2label = {
                    0: "O",
                    1: "B",
                    2: "I"
                }
                
        for _, item in data.iterrows():

            Instruction = f"[Task: NER][Domain: {config['name']}].You are a biomedical named entity recognition (NER) expert. " \
                        "Please extract and label all predefined biomedical entities using the BIO format. " \
                        f"Supported entity labels list (BIO format): {list(id2label.values())}. "
            
            
            # 将上下文列表合并
            tokens = item['tokens']
            input_text = " ".join(tokens)
            input_text += f" Tokens: {tokens}"
            
            answer = str(item['ner_tags'])

            processed_samples.append({
                    "Instruction": Instruction,
                    "input": input_text,
                    "output": answer,
                    "task": "NER"
                })
        
    return processed_samples

def process_re_dataset(dataset, config):
    """RE"""
    print(f"Processing RE dataset: {config['name']}")
    processed_samples = []

    if config['name'] == "bigbio/chemprot":
        data = pd.read_csv("data/RE/ChemProt_Corpus/chemprot_training/chemprot_merged_training_output.tsv", sep='\t', encoding='utf-8')
        
        for _, item in data.iterrows():
            Instruction = f"[Task: RE][Domain: {config['name']}]. You are an expert in biomedical relation extraction. " \
                "Given a text and two entities, identify the relationship between the entities based on the predefined relation types." \
                "Relation types:" \
                "- CPR:3: UPREGULATOR" \
                "- CPR:4: DOWNREGULATOR" \
                "- CPR:5: AGONIST" \
                "- CPR:6: ANTAGONIST" \
                "- CPR:9: SUBSTRATE" \
                f"- CPR:10: NO. What is the relationship between {item.get('entity1', '')} and {item.get('entity2', '')} in the following text?" \
                "Respond with: 'CPR:3', 'CPR:4','CPR:5','CPR:6''CPR:9'."
            
            input_text = item.get('text', '')
            output_text = f"{item.get('relation', 'false')}"

            # prompt += input_text

            processed_samples.append({
            "Instruction": Instruction,
            "input": input_text,
            "output": output_text,
            "task": "RE"
        })
        
    elif config['name'] == 'bigbio/ddi_corpus':

        print("Processing DDI dataset...")

        data = pd.read_json("data/seqcls/DDI_hf/train.json", lines=True)
                
        for _, item in data.iterrows():
            Instruction = f"[Task: RE][Domain: {config['name']}]. You are an expert in biomedical relation extraction." \
                                "Given a sentence containing two marked entities (@DRUG$ and @DRUG$), determine whether there is a meaningful biomedical relationship between them." \
                                "Respond with: '0', 'DDI-mechanism', 'DDI-effect', 'DDI-advise', 'DDI-int'." 
            
            input_text = item.get('sentence', '')
            output_text = f"{item.get('label', 'false')}"

            processed_samples.append({
                    "Instruction": Instruction,
                    "input": input_text,
                    "output": output_text,
                    "task": "RE"
                })
    else:
        None

    return processed_samples

def process_qa_dataset(dataset, config):
    print(f"Processing QA dataset: {config['name']}")
    processed_samples = []

    if config['name'] == 'qiaojin/PubMedQA':
        dataset_list = list(dataset)  
        dataset = dataset_list[:500]  

        for example in tqdm(dataset, desc=f"  -> {config['name']}"):
            Instruction = f"[Task: QA][Domain: {config['name']}]. You are an expert in biomedical question answering. Based on the following context and question, response with: 'yes', 'no', 'maybe'."
            question = ""
            context = ""
            answer = ""

            question = example['question']
            answer = example['final_decision']

            context = example['context'].get('contexts', [])
            if isinstance(context, list):
                context = " ".join(context)  
            else:
                context = str(context)  

            if not question or not answer:
                continue  

            input_text = f"Question: {question}\n\nContext: {context}"

            processed_samples.append({
                "Instruction": Instruction,
                "input": input_text,
                "output": answer,
                "task": 'QA'
            })
        
        return processed_samples

    elif config['name'] == 'GBaker/MedQA-USMLE-4-options':
        for example in tqdm(dataset, desc=f"  -> {config['name']}"):
            question = example['question']
            options = example['options']
            answer_idx = example['answer_idx']


            Instruction = f"[Task: QA][Domain: {config['name']}]. You are an expert in biomedical question answering." \
                            "Given a question and four multiple choice options (A, B, C, D), select the most appropriate answer based on your biomedical knowledge." \
                            "Respond with only the letter (A, B, C, or D) corresponding to your chosen answer."
            
            input_text = f"Question: {question}\nOpttions: {options}"
            output_text = answer_idx

            processed_samples.append({
                "Instruction": Instruction,
                "input": input_text,
                "output": str(output_text),
                "task": "QA"
            })
        
        return processed_samples


# ----------------------------------------------------------------------------
# 3. main
# ----------------------------------------------------------------------------
def main():
    all_processed_data = []
    
    PROCESSOR_MAP = {
        "ner": process_ner_dataset,
        "re": process_re_dataset,
        "qa": process_qa_dataset,
    }

    print("Begining...")

    for task_key, config in DATASET_CONFIG.items():
        print(f"\nLoading dataset '{config['name']}'...")
        try:
            if "subset" in config:
                dataset = load_dataset(config['name'], config['subset'], split=config['split'])
            elif "split" in config:
                dataset = load_dataset(config['name'], split=config['split'])
            else:
                dataset = None

            processor_func = PROCESSOR_MAP.get(config['type'])
            if processor_func:
                processed_data = processor_func(dataset, config)
                all_processed_data.extend(processed_data)
                print(f"  -> {config['name']} success，get {len(processed_data)} samples.")
            else:
                print(f"  -> Warning：Get'nt dataset '{config['type']}' Processor。")
        except Exception as e:
            print(f"  -> Dataset '{config['name']}' Error: {e}")

    print(f"\nDone! Get {len(all_processed_data)} samples.")

    print("Shuffle all data...")
    random.shuffle(all_processed_data)

    print(f"Save it to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, ensure_ascii=False, indent=2)

            
    print("\nAll Done！you can training with 'multitask_bio_dataset.jsonl'")


if __name__ == "__main__":
    main()
