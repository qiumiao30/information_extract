# tasks.py
"""
任务管理模块 - 集成所有组件，执行具体任务
"""

import pandas as pd
import ast
from typing import Dict, List, Optional
from models import ModelFactory
from data_processors import DataProcessorFactory
from evaluators import EvaluatorFactory
from prompts import PromptManager
from config import RELATION_LABELS, EVALUATION_CONFIG, RELATION_LABELS_IDS


class BaseTask:
    """基础任务类"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.model = None
        self.data_processor = None
        self.evaluator = None
        self.prompt_manager = PromptManager()
    
    def setup(self, **kwargs):
        """设置任务组件"""
        raise NotImplementedError
    
    def run(self, **kwargs):
        """运行任务"""
        raise NotImplementedError

class RelationExtractionTask(BaseTask):
    """关系抽取任务"""
    
    def __init__(self):
        super().__init__("relation_extraction")
    
    def setup(self, model_name: str = None, data_source: str = "custom", **kwargs):
        """设置关系抽取任务"""
        print(f"设置关系抽取任务...")
        
        # 初始化模型
        self.model = ModelFactory.create_model("relation_extraction", model_name)
        
        # 初始化数据处理器
        if data_source == "chemport":
            self.data_processor = DataProcessorFactory.create_processor("chemport")
        elif data_source == "ddi":
            self.data_processor = DataProcessorFactory.create_processor("ddi")
        elif data_source == "gad":
            self.data_processor = DataProcessorFactory.create_processor("gad")
        else:
            self.data_processor = DataProcessorFactory.create_processor("custom", **kwargs)
        
        # 初始化评估器
        self.evaluator = EvaluatorFactory.create_evaluator(
            "relation_extraction", 
            relation_labels=RELATION_LABELS
        )
        
        print("任务设置完成!")
    
    def run(self, max_samples: int = None, verbose: bool = True, data_source='chemport') -> Dict:
        """运行关系抽取任务"""
        if not all([self.model, self.data_processor, self.evaluator]):
            raise ValueError("请先调用setup()方法设置任务组件")
        
        max_samples = max_samples or EVALUATION_CONFIG["max_samples"]
        
        print(f"=== 开始运行关系抽取任务 ===")
        
        # 1. 加载数据
        print("1. 加载数据...")
        test_data = self.data_processor.load_data()
        test_data = self.data_processor.process_data(test_data)
        if test_data.empty:
            print("错误: 数据加载失败!")
            return {}
        
        # 限制样本数量
        test_data = test_data.head(max_samples)
        print(f"测试样本数量: {len(test_data)}")
        
        # 2. 执行预测
        print("\n2. 执行关系抽取...")
        predictions = []
        if data_source == "chemport":
            true_labels = test_data['relation'].tolist()
        elif data_source == "ddi":
            true_labels = test_data['label'].tolist()
            # 改为大写
            true_labels = [label.upper() for label in true_labels]
            # print(true_labels)
        elif data_source == "gad":
            true_labels = test_data['label'].tolist()
            # # 改为大写
            # true_labels = [label.upper() for label in true_labels]
        
        for idx, row in test_data.iterrows():
            # 生成prompt
            if data_source == 'chemport':
                prompt = self.prompt_manager.get_relation_extraction_prompt(
                    data_source, row['text'], row['entity1'], row['entity2']
                )
            elif data_source == 'ddi':
                prompt = self.prompt_manager.get_relation_extraction_prompt(
                    data_source, row['sentence']
                )
            elif data_source == 'gad':
                prompt = self.prompt_manager.get_relation_extraction_prompt(
                    data_source, row['sentence']
                )
            # print(prompt)

            # 执行预测
            pred_relation = self.model.extract_relation(prompt)

            print(f"样本 {idx+1} 预测关系: {pred_relation}")
            predictions.append(pred_relation)
            
            if data_source == "chemport":
                if verbose:
                    print(f"样本 {idx+1}:")
                    print(f"  PMID: {row.get('pmid', 'N/A')}")
                    print(f"  文本: {row['text'][:100]}...")
                    print(f"  实体1: {row['entity1']} ({row.get('entity1_type', 'Unknown')})")
                    print(f"  实体2: {row['entity2']} ({row.get('entity2_type', 'Unknown')})")
                    print(f"  真实关系: {row['relation']} ({RELATION_LABELS.get(row['relation'], 'unknown')})")
                    print(f"  预测关系: {pred_relation} ({RELATION_LABELS.get(pred_relation, 'unknown')})")
                    print(f"  正确: {'✓' if pred_relation == row['relation'] else '✗'}")
                    print()
            elif data_source == "ddi":
                if verbose:
                    print(f"样本 {idx+1}:")
                    print(f"Sentence: {row['sentence']}")
                    print(f"真实关系: {row['label'].upper()}")
                    print(f"预测关系: {pred_relation}") 
                    print(f"正确: {'✓' if pred_relation == row['label'].upper() else '✗'}")
                    print()
            elif data_source == "gad":
                if verbose:
                    print(f"样本 {idx+1}:")
                    print(f"Sentence: {row['sentence']}")
                    print(f"真实关系: {row['label']}")
                    print(f"预测关系: {pred_relation}") 
                    print(f"正确: {'✓' if pred_relation == row['label'] else '✗'}")
                    print()
        
        # 3. 评估结果
        print("3. 评估结果...")
        if data_source == "gad":
            true_labels = list(map(str, true_labels))
            valid_labels = ["0", "1"]
            predictions = [p if p in valid_labels else "1" for p in predictions]
            predictions = list(map(str, predictions))
            # true_labels = [ast.literal_eval(item)[0] for item in true_labels]
            print("true label: ", true_labels)
            print("predictions: ", predictions)

        elif data_source == "chemport":
            valid_labels = ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9']
            predictions = [p if p in valid_labels else "CPR:4" for p in predictions]

        elif data_source == "ddi":
            valid_labels = ["DDI-EFFECT", "DDI-MECHANISM", "DDI-ADVISE", "0", "DDI-INT"]
            predictions = [label.upper() for label in predictions]
            predictions = [p if p in valid_labels else "0" for p in predictions]
            
        results = self.evaluator.evaluate_with_details(true_labels, predictions, test_data)
        
        if verbose:
            self.evaluator.print_detailed_results(true_labels, predictions)
            
            # 详细分类报告
            detailed_report = self.evaluator.get_detailed_report(true_labels, predictions)
            print("\n=== 详细分类报告 ===")
            print(detailed_report)
        
        return results


class NERTask(BaseTask):
    """命名实体识别任务"""
    
    def __init__(self):
        super().__init__("ner")
    
    def setup(self, model_name: str = None, data_source: str="ncbi", **kwargs):
        """设置NER任务"""
        print(f"设置命名实体识别任务...")
        
        # 初始化模型
        self.model = ModelFactory.create_model("ner", model_name)

        # 初始化数据处理器
        if data_source == "bc2gm":
            self.data_processor = DataProcessorFactory.create_processor("bc2gm", **kwargs)
        elif data_source == "jnlpba":
            self.data_processor = DataProcessorFactory.create_processor("jnlpba", **kwargs)
        elif data_source == "ncbi":
            self.data_processor = DataProcessorFactory.create_processor("ncbi", **kwargs)
        
        # 添加NER数据处理器
        self.evaluator = EvaluatorFactory.create_evaluator("ner", 
                                                           ner_labels=RELATION_LABELS)
        
        print("任务设置完成!")
    
    def run(self, max_samples: int=None, verbose: bool = True, data_source: str="ncbi") -> Dict:
        """运行NER任务"""
        if not all([self.model, self.data_processor, self.evaluator]):
            raise ValueError("请先调用setup()方法设置任务组件")
        
        max_samples = max_samples or EVALUATION_CONFIG["max_samples"]
        
        print(f"=== 开始运行关系抽取任务 ===")
        
        # 1. 加载数据
        print("1. 加载数据...")
        if data_source == 'ncbi':
            test_data = self.data_processor.load_data()
            test_data = self.data_processor.process_data(test_data)
        else:
            test_data = self.data_processor.load_data()
        if test_data.empty:
            print("错误: 数据加载失败!")
            return {}
        
        # 限制样本数量
        test_data = test_data.head(max_samples)
        print(f"测试样本数量: {len(test_data)}")
        
        # 2. 执行预测
        print("\n2. 执行关系抽取...")
        predictions = []
        true_labels = test_data['ner_tags'].tolist()
        true_labels = [label for sublist in true_labels for label in sublist]
        # 转换为大写
        true_labels = [label.upper() for label in true_labels]  
        if data_source == "bc2gm":
            converted_tags = [
                        'O' if tag == '0' else 
                        'B-DISEASE' if tag == '1' else 
                        'I-DISEASE' if tag == '2' else tag
                        for tag in true_labels
                        ]

            true_labels = converted_tags

        print("真实实体标签长度:", len(true_labels))
        
        for idx, row in test_data.iterrows():
            # 生成prompt
            if data_source == 'bc2gm':
                prompt = self.prompt_manager.get_ner_prompt(
                    data_source, row['text'], row['tokens'], row['ner_tags']
                )
            elif data_source == 'jnlpba':
                prompt = self.prompt_manager.get_ner_prompt(
                    data_source, row['text'], row['tokens'], row['ner_tags']
                )
            elif data_source == 'ncbi':
                prompt = self.prompt_manager.get_ner_prompt(
                    data_source, row['text'], row['tokens'], row['ner_tags']
                )

            pred_ner = self.model.extract_entities(prompt)

            print(f"样本 {idx+1} 预测关系: {pred_ner}")
            ner_tags_upper = [label.upper() for label in row['ner_tags']]

            
            converted_tags = [
                    'O' if tag == '0' else 
                    'B-DISEASE' if tag == '1' else 
                    'I-DISEASE' if tag == '2' else tag
                    for tag in ner_tags_upper
                    ]
            ner_tags_upper = converted_tags
            
            if verbose:
                print(f"样本 {idx+1}:")
                print(f"  PMID: {row.get('pmid', 'N/A')}")
                print(f"  文本: {row['text'][:100]}...")
                print(f"  Tokens: {row['tokens']}")
                print(f"  真实实体标签: {ner_tags_upper}")
                print(f"  预测实体标签: {pred_ner}")
                
                print(f"  正确: {'✓' if pred_ner == ner_tags_upper else '✗'}")
                
                # 判断真实实体标签与预测实体标签的数量是否一致
                if len(ner_tags_upper) == len(pred_ner):
                    print(f"  实体数量一致: {len(ner_tags_upper)}")

                # 最后还是不一致，就强制对齐
                if len(pred_ner) != len(ner_tags_upper):
                    # print(f"  ⚠️ 已重试 {MAX_RETRIES} 次，仍然不一致，执行对齐处理")
                    print(f"  注意: 真实实体标签与预测实体标签数量不一致! ")
                    if len(pred_ner) > len(ner_tags_upper):
                        pred_ner = pred_ner[:len(ner_tags_upper)]
                    else:
                        pred_ner += ['O'] * (len(ner_tags_upper) - len(pred_ner))

                    print(f"  ✅ 对齐后的预测实体标签: {pred_ner}")

                print()
            # self.model.clear_history()  # 清除模型历史记录，避免影响后续预测
            predictions.append(pred_ner)
        predictions = [label for sublist in predictions for label in sublist]
        
        # 3. 评估结果
        print("3. 评估结果...")
        results = self.evaluator.evaluate_with_details(true_labels, predictions, test_data)
        
        if verbose:
            self.evaluator.print_detailed_results(true_labels, predictions)
            
            # 详细分类报告
            detailed_report = self.evaluator.get_detailed_report(true_labels, predictions)
            print("\n=== 详细分类报告 ===")
            print(detailed_report)
        
        return results


class QATask(BaseTask):
    """问答推理任务"""
    
    def __init__(self):
        super().__init__("qa")
    
    def setup(self, model_name: str = None, data_source: str="pubmedqa", **kwargs):
        """设置问答推理任务"""
        print(f"设置问答推理任务...")
        
        # 初始化模型
        self.model = ModelFactory.create_model("qa", model_name)
        
        # 初始化数据处理器
        if data_source == "pubmedqa":
            self.data_processor = DataProcessorFactory.create_processor("pubmedqa", **kwargs)
        elif data_source == "bioasq":
            self.data_processor = DataProcessorFactory.create_processor("bioasq", **kwargs)
        elif data_source == "medqa":
            self.data_processor = DataProcessorFactory.create_processor("medqa", **kwargs)
        
        
        # 初始化评估器
        self.evaluator = EvaluatorFactory.create_evaluator(
            "qa", 
            relation_labels=RELATION_LABELS
        )
        
        print("任务设置完成!")
    
    def run(self, max_samples: int=None, verbose: bool = True, data_source: str="pubmedqa", **kwargs) -> Dict:
        """运行问答推理任务"""
        print("=== 开始运行文本推理任务 ===")
        if not all([self.model, self.data_processor, self.evaluator]):
            raise ValueError("请先调用setup()方法设置任务组件")

        max_samples = max_samples or EVALUATION_CONFIG["max_samples"]
        
        # 1. 加载数据
        print("1. 加载数据...")
        if data_source == 'pubmedqa':
            test_data = self.data_processor.load_data()
            test_data = test_data.iloc[500:-1]
        elif data_source == 'medqa':
            test_data = self.data_processor.load_data()
        else:
            test_data = self.data_processor.load_data()
            test_data = self.data_processor.process_data(test_data)
        if test_data.empty:
            print("错误: 数据加载失败!")
            return {}
        
        # 限制样本数量
        test_data = test_data.head(max_samples)
        print(f"测试样本数量: {len(test_data)}")
        
        # 2. 执行预测
        print("\n2. 执行关系抽取...")
        predictions = []
        if data_source == "pubmedqa" or data_source == "medqa":
            true_labels = test_data['final_decision'].tolist()
            true_labels = [label.lower() for label in true_labels]
        else:
            true_labels = test_data['answer'].tolist()
            true_labels = [label.lower() for label in true_labels]

        print(true_labels)
        
        for idx, row in test_data.iterrows():
            # 生成prompt
            prompt = self.prompt_manager.get_qa_prompt(
                data_source, row['question'], row['context']
            )
                
            # 执行预测
            pred_qa = self.model.extract_answer(prompt)
            print(f"样本 {idx+1} 预测关系: {pred_qa}")
            print(f"样本 {idx+1} 真实关系: {row['final_decision'] if (data_source == 'pubmedqa' or data_source == 'medqa') else row['answer']}")
            predictions.append(pred_qa)
        if data_source == "pubmedqa":
            valid_labels = ['yes', 'no', 'maybe']
            predictions = [p if p in valid_labels else "no" for p in predictions]
        elif data_source == "medqa":
            valid_labels = ['a', 'b', 'c', 'd']
            predictions = [p if p in valid_labels else "a" for p in predictions]
        
        results = {"predictions": predictions}
        
        if true_labels:
            if data_source == "pubmedqa":
                # 取true_labels的后500个元素
                if len(true_labels) > 300:
                    true_labels = true_labels[-300:]
                    print(len(true_labels))
                if len(predictions) > 300:
                    predictions = predictions[-300:]
                    print(len(predictions))
            eval_results = self.evaluator.evaluate_with_details(true_labels, predictions)
            results.update(eval_results)
            
            print("=== 评估结果 ===")
            self.evaluator.print_results()
        
        return results

class ClassificationTask(BaseTask):
    """文本分类任务"""
    
    def __init__(self):
        super().__init__("classification")
    
    def setup(self, model_name: str = None, classes: List[str] = None, **kwargs):
        """设置分类任务"""
        print(f"设置文本分类任务...")
        
        self.model = ModelFactory.create_model("classification", model_name)
        self.classes = classes or ["positive", "negative", "neutral"]
        self.evaluator = EvaluatorFactory.create_evaluator("classification")
        
        print("任务设置完成!")
    
    def run(self, text_list: List[str], true_labels: List[str] = None, **kwargs) -> Dict:
        """运行分类任务"""
        print("=== 开始运行文本分类任务 ===")
        
        predictions = []
        for text in text_list:
            prompt = self.prompt_manager.get_classification_prompt(text, self.classes)
            pred_class = self.model.classify(prompt, self.classes)
            predictions.append(pred_class)
        
        results = {"predictions": predictions}
        
        if true_labels:
            eval_results = self.evaluator.evaluate(true_labels, predictions)
            results.update(eval_results)
            
            print("=== 评估结果 ===")
            self.evaluator.print_results()
        
        return results


class TaskFactory:
    """任务工厂"""
    
    @staticmethod
    def create_task(task_type: str):
        """创建任务实例"""
        if task_type == "relation_extraction":
            return RelationExtractionTask()
        elif task_type == "ner":
            return NERTask()
        elif task_type == "classification":
            return ClassificationTask()
        elif task_type == "qa":
            return QATask()
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")


def quick_relation_extraction(max_samples: int = 20, verbose: bool = True) -> Dict:
    """快速运行关系抽取任务"""
    task = TaskFactory.create_task("relation_extraction")
    task.setup()
    return task.run(max_samples=max_samples, verbose=verbose)


def quick_classification(text_list: List[str], true_labels: List[str] = None, 
                        classes: List[str] = None) -> Dict:
    """快速运行分类任务"""
    task = TaskFactory.create_task("classification")
    task.setup(classes=classes)
    return task.run(text_list, true_labels)


def quick_ner(text_list: List[str]) -> Dict:
    """快速运行NER任务"""
    task = TaskFactory.create_task("ner")
    task.setup()
    return task.run(text_list)

def quick_qa(text_list: List[str]) -> Dict:
    """快速运行NER任务"""
    task = TaskFactory.create_task("qa")
    task.setup()
    return task.run(text_list)
