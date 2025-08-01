# data_processors.py
"""
Data processing module - for handling data from various sources
"""
import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from config import DATASET_CONFIG

class BaseDataProcessor(ABC):
    def __init__(self, dataset_name: str = None):
        self.dataset_name = dataset_name
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def process_data(self, dataset) -> pd.DataFrame:
        pass
    
    def get_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
        data = self.load_data()
        return data.head(n_samples)
    
class GADDataProcessor(BaseDataProcessor):
    
    def __init__(self):
        super().__init__("bigbio/gad")
    
    def load_data(self) -> pd.DataFrame:
        config = DATASET_CONFIG["gad"]
        dataset = load_dataset(config["name"], "gad_fold5_bigbio_text", split=config["split"])
        print(f"Load dataset: {len(dataset)} samples")
        return self.process_data(dataset)
        
    def process_data(self, dataset) -> pd.DataFrame:
        processed_data = []
        
        for item in dataset:
            print(item)
            text = item['text']
            label = item['labels']

            processed_data.append({
                'text': text,
                'label': label,
            })
        
        return pd.DataFrame(processed_data)

class CustomDataProcessor(BaseDataProcessor):
    
    def __init__(self, dataset_name: str, dataset_path: str = None):
        super().__init__(dataset_name)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
    
    def load_data(self) -> pd.DataFrame:
        try:
            if self.dataset_path:
                # From local
                if self.dataset_path.endswith('.tsv'):
                    return pd.read_csv(self.dataset_path, sep='\t', encoding='utf-8')
                elif self.dataset_path.endswith('.json'):
                    return pd.read_json(self.dataset_path, lines=True)
                else:
                    raise ValueError(f"Unsupported: {self.dataset_path}")
            else:
                # From HuggingFace
                dataset = load_dataset(self.dataset_name, split="test")
                return self.process_data(dataset)
        except Exception as e:
            print(f"Load dataset {self.dataset_name} unsuccess: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        """custom dataset"""
        processed_data = []
        
        if self.dataset_name == "chemport":
            for _, item in dataset.iterrows():
                processed_data.append({
                    'text': item.get('text', ''),
                    'entity1': item.get('entity1', ''),
                    'entity2': item.get('entity2', ''),
                    'relation': item.get('relation', 'false')
                })
        elif self.dataset_name == "ncbi":
            for _, item in dataset.iterrows():
                tokens = item['tokens']
                ner_tags = item['ner_tags']
                text = ' '.join(tokens)

                processed_data.append({
                    'tokens': tokens,
                    'ner_tags': ner_tags,
                    'text': text,
                })
        elif self.dataset_name == "bioasq":
            for _, item in dataset.iterrows():
                sentence1 = item.get('sentence1', '')
                sentence2 = item.get('sentence2', [])
                answer = item.get('label', '')

                processed_data.append({
                    'question': sentence1,
                    'context': sentence2,
                    'answer': answer,
                })
        elif self.dataset_name == "ddi":
            for _, item in dataset.iterrows():
                sentence = item.get('sentence', '')
                label = item.get('label', '')

                processed_data.append({
                    'sentence': sentence,
                    'label': label,
                })
        elif self.dataset_name == "gad":
            for _, item in dataset.iterrows():
                sentence = item.get('sentence', '')
                label = item.get('label', '')

                processed_data.append({
                    'sentence': sentence,
                    'label': label,
                })

        return pd.DataFrame(processed_data)

class JNLPBADataProcessor(BaseDataProcessor):
    
    def __init__(self):
        super().__init__("jnlpba")
    
    def load_data(self) -> pd.DataFrame:
        try:
            # Load JNLPBA From huggingface
            dataset = load_dataset("jnlpba", split="validation")
            print(f"Success Load JNLPBA Dataset: {len(dataset)} samples")
            return self.process_data(dataset)
        except Exception as e:
            print(f"Unsuccess Load JNLPBA Dataset: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        processed_data = []
        
        label_names = ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA', 'B-cell_line', 
                      'I-cell_line', 'B-cell_type', 'I-cell_type', 'B-protein', 'I-protein']
        
        for item in dataset:
            tokens = item['tokens']
            ner_tags = item['ner_tags']
            ner_labels = [label_names[tag] for tag in ner_tags]
            ner_tags = [label_names.index(tag) for tag in ner_labels]
            text = ' '.join(tokens)
            
            processed_data.append({
                    'pmid': item.get('id', ''),
                    'text': text,
                    'tokens': tokens,
                    'ner_tags': ner_labels,
                    'ner_tags_ids': ner_tags
                })
        
        return pd.DataFrame(processed_data)
    
class BC2GMDataProcessor(BaseDataProcessor):
    
    def __init__(self):
        super().__init__("bc2gm_corpus")
    
    def load_data(self) -> pd.DataFrame:
        try:
            dataset = load_dataset("bc2gm_corpus", split="test")
            print(f"Success Load BC2GM Dataset: {len(dataset)} samples")
            return self.process_data(dataset)
        except Exception as e:
            print(f"Unsuccess Load BC2GM: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        processed_data = []

        label_names = ['0', '1', '2']
        
        for item in dataset:
            tokens = item['tokens']
            ner_tags = item['ner_tags']
            ner_labels = [label_names[tag] for tag in ner_tags]
            ner_tags = [label_names.index(tag) for tag in ner_labels]

            text = ' '.join(tokens)
            
            processed_data.append({
                    'pmid': item.get('id', ''),
                    'text': tokens,
                    'tokens': tokens,
                    'ner_tags': ner_labels,
                    'ner_tags_ids': ner_tags
                })
        
        return pd.DataFrame(processed_data)
    
class PubMedQADataProcessor(BaseDataProcessor): 
    def __init__(self):
        super().__init__("PubMedQA")
    
    def load_data(self) -> pd.DataFrame:
        try:
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

            print(f"Success Load PubMedQA Dataset: {len(dataset)} samples")
            return self.process_data(dataset)
        except Exception as e:
            print(f"Unsuccess Load PubMedQA: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        processed_data = []
        
        for item in dataset:
            question = item['question']
            context = item['context']
            final_decision = item['final_decision']
            
            processed_data.append({
                    'pmid': item.get('pubid', ''),
                    'question': question,
                    'context': context,
                    'final_decision': final_decision
                })
        
        return pd.DataFrame(processed_data)

class MedQADataProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__("MedQA")
    
    def load_data(self) -> pd.DataFrame:
        try:
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options",  split="test")

            print(f"Success Load MedQA Dataset: {len(dataset)} samples")
            return self.process_data(dataset)
        except Exception as e:
            print(f"Unsuccess Load MedQA: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        processed_data = []
        
        for item in dataset:
            question = item['question']
            context = item['options']
            final_decision = item['answer_idx']
            
            processed_data.append({
                    'question': question,
                    'context': context,
                    'final_decision': final_decision
                })
        
        return pd.DataFrame(processed_data) 

class BioASQDataProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__("BioASQ")
    
    def load_data(self) -> pd.DataFrame:
        try:
            dataset = load_dataset("kroshan/BioASQ", split="validation")
            print(f"Success Load PubMedQA Dataset: {len(dataset)} samples")
            return self.process_data(dataset)
        except Exception as e:
            print(f"Unsuccess Load PubMedQA: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        processed_data = []
        
        for item in dataset:
            question = item['question']
            text = item['text']
            
            answer_start = text.find("<answer>") + len("<answer>")
            context_start = text.find("<context>")

            answer = text[answer_start:context_start].strip()
            context = text[context_start + len("<context>"):].strip()

            context = context
            final_decision = answer
            
            processed_data.append({
                    'question': question,
                    'context': context,
                    'final_decision': final_decision
                })
        
        return pd.DataFrame(processed_data)

class DataProcessorFactory:
    @staticmethod
    def create_processor(processor_type: str, **kwargs):
        if processor_type == "chemport":
            return CustomDataProcessor(dataset_name="chemport", dataset_path="data/RE/ChemProt_Corpus/chemprot_test_gs/chemprot_merged_output.tsv")
        elif processor_type == "jnlpba":
            return JNLPBADataProcessor()
        elif processor_type == "bc2gm":
            return BC2GMDataProcessor()
        elif processor_type == "pubmedqa":
            return PubMedQADataProcessor()
        elif processor_type == "medqa":
            return MedQADataProcessor()
        elif processor_type == "bioasq":
            return CustomDataProcessor(dataset_name="bioasq", dataset_path="data/seqcls/bioasq_hf/test.json")
        elif processor_type == "gad":
            return CustomDataProcessor(dataset_name="gad", dataset_path="data/seqcls/GAD_hf/test.json")
        elif processor_type == "ncbi":
            return CustomDataProcessor(dataset_name="ncbi", dataset_path="data/tokcls/NCBI-disease_hf/test.json")
        elif processor_type == "ddi":
            return CustomDataProcessor(dataset_name="ddi", dataset_path="data/seqcls/DDI_hf/test.json")
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")
