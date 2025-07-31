# data_processors.py
"""
数据处理模块 - 处理不同数据源的数据
"""

import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from config import DATASET_CONFIG

class BaseDataProcessor(ABC):
    """数据处理器基类"""
    
    def __init__(self, dataset_name: str = None):
        self.dataset_name = dataset_name
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """加载数据，子类需要实现此方法"""
        pass
    
    @abstractmethod
    def process_data(self, dataset) -> pd.DataFrame:
        """处理数据为统一格式，子类需要实现此方法"""
        pass
    
    def get_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
        """获取样本数据"""
        data = self.load_data()
        return data.head(n_samples)


class ChemProtDataProcessor(BaseDataProcessor):
    """ChemProt数据处理器"""
    
    def __init__(self):
        super().__init__("bigbio/chemprot")
    
    def load_data(self) -> pd.DataFrame:
        """加载ChemProt数据集"""

        config = DATASET_CONFIG["chemprot"]
        dataset = load_dataset(config["name"], split=config["split"])
        print(f"成功加载数据集: {len(dataset)} 条样本")
        return self.process_data(dataset)
        
    
    def process_data(self, dataset) -> pd.DataFrame:
        """将数据集处理为标准格式"""
        processed_data = []
        
        for item in dataset:
            # 构建实体ID到文本的映射
            entity_map = {}
            if 'entities' in item and item['entities']:
                for i, entity_id in enumerate(item['entities']['id']):
                    entity_map[entity_id] = {
                        'text': item['entities']['text'][i],
                        'type': item['entities']['type'][i]
                    }
            
            # 处理关系
            if 'relations' in item and item['relations'] and item['relations']['type']:
                for i, rel_type in enumerate(item['relations']['type']):
                    arg1_id = item['relations']['arg1'][i]
                    arg2_id = item['relations']['arg2'][i]
                    
                    if arg1_id in entity_map and arg2_id in entity_map:
                        processed_data.append({
                            'pmid': item.get('pmid', ''),
                            'text': item.get('text', ''),
                            'entity1': entity_map[arg1_id]['text'],
                            'entity2': entity_map[arg2_id]['text'],
                            'entity1_type': entity_map[arg1_id]['type'],
                            'entity2_type': entity_map[arg2_id]['type'],
                            'relation': rel_type,
                            'arg1_id': arg1_id,
                            'arg2_id': arg2_id
                        })
            else:
                # 如果没有关系，可以添加负样本
                if len(entity_map) >= 2:
                    entities = list(entity_map.items())
                    processed_data.append({
                        'pmid': item.get('pmid', ''),
                        'text': item.get('text', ''),
                        'entity1': entities[0][1]['text'],
                        'entity2': entities[1][1]['text'],
                        'entity1_type': entities[0][1]['type'],
                        'entity2_type': entities[1][1]['type'],
                        'relation': 'false',
                        'arg1_id': entities[0][0],
                        'arg2_id': entities[1][0]
                    })
        
        return pd.DataFrame(processed_data)
    
class GADDataProcessor(BaseDataProcessor):
    """ChemProt数据处理器"""
    
    def __init__(self):
        super().__init__("bigbio/gad")
    
    def load_data(self) -> pd.DataFrame:
        """加载ChemProt数据集"""

        config = DATASET_CONFIG["gad"]
        dataset = load_dataset(config["name"], "gad_fold5_bigbio_text", split=config["split"])
        print(f"成功加载数据集: {len(dataset)} 条样本")
        return self.process_data(dataset)
        
    def process_data(self, dataset) -> pd.DataFrame:
        """将数据集处理为标准格式"""
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
    """自定义数据处理器模板"""
    
    def __init__(self, dataset_name: str, dataset_path: str = None):
        super().__init__(dataset_name)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
    
    def load_data(self) -> pd.DataFrame:
        """加载自定义数据集"""
        try:
            if self.dataset_path:
                # 从本地文件加载
                if self.dataset_path.endswith('.tsv'):
                    return pd.read_csv(self.dataset_path, sep='\t', encoding='utf-8')
                elif self.dataset_path.endswith('.json'):
                    return pd.read_json(self.dataset_path, lines=True)
                else:
                    raise ValueError(f"不支持的文件格式: {self.dataset_path}")
            else:
                # 从HuggingFace加载
                dataset = load_dataset(self.dataset_name, split="test")
                return self.process_data(dataset)
        except Exception as e:
            print(f"加载数据集 {self.dataset_name} 失败: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        """处理自定义数据集格式"""
        # 根据具体数据集格式进行处理
        # 返回包含以下列的DataFrame: text, entity1, entity2, relation
        processed_data = []
        
        if self.dataset_name == "chemport":
            for _, item in dataset.iterrows():
                # TODO: 根据实际数据格式实现处理逻辑
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
                # 重构句子文本
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
    """JNLPBA命名实体识别数据处理器"""
    
    def __init__(self):
        super().__init__("jnlpba")
    
    def load_data(self) -> pd.DataFrame:
        """加载JNLPBA数据集"""
        try:
            # 从HuggingFace加载JNLPBA数据集
            dataset = load_dataset("jnlpba", split="validation")
            print(f"成功加载JNLPBA数据集: {len(dataset)} 条样本")
            return self.process_data(dataset)
        except Exception as e:
            print(f"加载JNLPBA数据集失败: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        """将JNLPBA数据集处理为标准格式"""
        processed_data = []
        
        # 标签映射
        label_names = ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA', 'B-cell_line', 
                      'I-cell_line', 'B-cell_type', 'I-cell_type', 'B-protein', 'I-protein']
        
        for item in dataset:
            tokens = item['tokens']
            ner_tags = item['ner_tags']
            
            # 将token级别的标签转换为字符串标签
            ner_labels = [label_names[tag] for tag in ner_tags]

            # 将标签转换为ID
            ner_tags = [label_names.index(tag) for tag in ner_labels]
            
            # 重构句子文本
            text = ' '.join(tokens)
            
            # 提取实体
            # entities = self._extract_entities(tokens, ner_labels)
            
            processed_data.append({
                    'pmid': item.get('id', ''),
                    'text': text,
                    'tokens': tokens,
                    'ner_tags': ner_labels,
                    'ner_tags_ids': ner_tags
                })
        
        return pd.DataFrame(processed_data)
    
class BC2GMDataProcessor(BaseDataProcessor):
    """BC2GM命名实体识别数据处理器"""
    
    def __init__(self):
        super().__init__("bc2gm_corpus")
    
    def load_data(self) -> pd.DataFrame:
        """加载BC2GM数据集"""
        try:
            # 从HuggingFace加载BC2GM数据集
            dataset = load_dataset("bc2gm_corpus", split="test")
            print(f"成功加载BC2GM数据集: {len(dataset)} 条样本")
            return self.process_data(dataset)
        except Exception as e:
            print(f"加载BC2GM数据集失败: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        """将BC2GM数据集处理为标准格式"""
        processed_data = []
        
        # 标签映射
        label_names = ['0', '1', '2']
        
        for item in dataset:
            tokens = item['tokens']
            ner_tags = item['ner_tags']
            
            # 将token级别的标签转换为字符串标签
            ner_labels = [label_names[tag] for tag in ner_tags]

            # 将标签转换为ID
            ner_tags = [label_names.index(tag) for tag in ner_labels]
            
            # 重构句子文本
            text = ' '.join(tokens)
            
            # 提取实体
            # entities = self._extract_entities(tokens, ner_labels)
            
            processed_data.append({
                    'pmid': item.get('id', ''),
                    'text': tokens,
                    'tokens': tokens,
                    'ner_tags': ner_labels,
                    'ner_tags_ids': ner_tags
                })
        
        return pd.DataFrame(processed_data)
    
class PubMedQADataProcessor(BaseDataProcessor):
    """PubMedQA问答推理数据处理器"""
    
    def __init__(self):
        super().__init__("PubMedQA")
    
    def load_data(self) -> pd.DataFrame:
        """加载PubMedQA数据集"""
        try:
            # 从HuggingFace加载PubMedQA数据集
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

            print(f"成功加载PubMedQA数据集: {len(dataset)} 条样本")
            return self.process_data(dataset)
        except Exception as e:
            print(f"加载PubMedQA数据集失败: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        """将PubMedQA数据集处理为标准格式"""
        processed_data = []
        
        # 标签映射
        # label_names = ['0', '1', '2']
        
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
    """MedQA问答推理数据处理器"""
    
    def __init__(self):
        super().__init__("MedQA")
    
    def load_data(self) -> pd.DataFrame:
        """加载MedQA数据集"""
        try:
            # 从HuggingFace加载MedQA数据集
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options",  split="test")

            print(f"成功加载MedQA数据集: {len(dataset)} 条样本")
            return self.process_data(dataset)
        except Exception as e:
            print(f"加载MedQA数据集失败: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        """将MedQA数据集处理为标准格式"""
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
    """BioASQ问答推理数据处理器"""
    
    def __init__(self):
        super().__init__("BioASQ")
    
    def load_data(self) -> pd.DataFrame:
        """加载PubMedQA数据集"""
        try:
            # 从HuggingFace加载PubMedQA数据集
            dataset = load_dataset("kroshan/BioASQ", split="validation")
            print(f"成功加载PubMedQA数据集: {len(dataset)} 条样本")
            return self.process_data(dataset)
        except Exception as e:
            print(f"加载PubMedQA数据集失败: {e}")
            return pd.DataFrame()
    
    def process_data(self, dataset) -> pd.DataFrame:
        """将PubMedQA数据集处理为标准格式"""
        processed_data = []
        
        # 标签映射
        # label_names = ['0', '1', '2']
        
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
    """数据处理器工厂"""
    
    @staticmethod
    def create_processor(processor_type: str, **kwargs):
        """创建数据处理器实例"""
        if processor_type == "chemprot_null":
            return ChemProtDataProcessor()
        elif processor_type == "chemport":
            return CustomDataProcessor(dataset_name="chemport", dataset_path="/home/siat/Data/synthetic/MinerU/RE/ChemProt_Corpus/chemprot_test_gs/chemprot_merged_output.tsv")
        elif processor_type == "jnlpba":
            return JNLPBADataProcessor()
        elif processor_type == "bc2gm":
            return BC2GMDataProcessor()
        elif processor_type == "pubmedqa":
            return PubMedQADataProcessor()
        elif processor_type == "medqa":
            return MedQADataProcessor()
        elif processor_type == "bioasq":
            return CustomDataProcessor(dataset_name="bioasq", dataset_path="/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/seqcls/bioasq_hf/test.json")
        elif processor_type == "gad":
            # return GADDataProcessor()
            return CustomDataProcessor(dataset_name="gad", dataset_path="/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/seqcls/GAD_hf/test.json")
        elif processor_type == "ncbi":
            return CustomDataProcessor(dataset_name="ncbi", dataset_path="/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/tokcls/NCBI-disease_hf/test.json")
        elif processor_type == "ddi":
            return CustomDataProcessor(dataset_name="ddi", dataset_path="/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/seqcls/DDI_hf/test.json")
        else:
            raise ValueError(f"不支持的处理器类型: {processor_type}")
        
# if __name__ == "__main__":
#     # 测试JNLPBA数据处理器
#     jnlpba_processor = DataProcessorFactory.create_processor("jnlpba")
    
#     # 获取样本数据
#     print(jnlpba_processor)
#     sample_data = jnlpba_processor.get_sample_data(-1)
#     print("JNLPBA样本数据:")
#     sample_data.to_csv("jnlpba_sample_data_train.csv", index=False)
    
#     # # 获取统计信息
#     # stats = jnlpba_processor.get_ner_statistics()
#     # print("\nJNLPBA数据集统计:")
#     # for key, value in stats.items():
#     #     print(f"{key}: {value}")