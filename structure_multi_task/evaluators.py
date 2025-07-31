# evaluators.py
"""
评估模块 - 统一管理各种评估指标和方法
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, 
    classification_report,
    accuracy_score,
    confusion_matrix
)
from typing import List, Dict, Tuple, Optional
import pandas as pd


class BaseEvaluator:
    """基础评估器"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, y_true: List, y_pred: List, **kwargs) -> Dict:
        """评估预测结果"""
        raise NotImplementedError
    
    def print_results(self):
        """打印评估结果"""
        for metric, value in self.results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")


class ClassificationEvaluator(BaseEvaluator):
    """分类任务评估器"""
    
    def evaluate(self, y_true: List[str], y_pred: List[str], 
                 labels: Optional[List[str]] = None) -> Dict:
        """评估分类结果"""
        if labels is None:
            labels = list(set(y_true + y_pred))
        
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        
        # 宏平均指标
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        
        # 微平均指标
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='micro', zero_division=0
        )
        
        # 加权平均指标
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='weighted', zero_division=0
        )
        
        self.results = {
            'accuracy': accuracy,
            'macro_precision': precision_macro,
            'macro_recall': recall_macro,
            'macro_f1': f1_macro,
            'micro_precision': precision_micro,
            'micro_recall': recall_micro,
            'micro_f1': f1_micro,
            'weighted_precision': precision_weighted,
            'weighted_recall': recall_weighted,
            'weighted_f1': f1_weighted
        }
        
        return self.results
    
    def get_detailed_report(self, y_true: List[str], y_pred: List[str], 
                           labels: Optional[List[str]] = None) -> str:
        """获取详细分类报告"""
        if labels is None:
            labels = list(set(y_true + y_pred))
        
        print("labels", labels)
        
        return classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    
    def get_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            labels: Optional[List[str]] = None) -> np.ndarray:
        """获取混淆矩阵"""
        if labels is None:
            labels = list(set(y_true + y_pred))
        
        return confusion_matrix(y_true, y_pred, labels=labels)

class RelationExtractionEvaluator(ClassificationEvaluator):
    """关系抽取评估器"""
    
    def __init__(self, relation_labels: Dict[str, str] = None):
        super().__init__()
        self.relation_labels = relation_labels or {}
    
    def evaluate_with_details(self, y_true: List[str], y_pred: List[str], 
                             data: pd.DataFrame = None) -> Dict:
        """带详细信息的评估"""
        # 基础评估
        results = self.evaluate(y_true, y_pred)
        
        # 添加关系类型统计
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()
        
        results['true_label_distribution'] = true_counts.to_dict()
        results['pred_label_distribution'] = pred_counts.to_dict()
        
        # 正确预测的样本数
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        results['correct_predictions'] = correct_predictions
        results['total_predictions'] = len(y_true)
        
        return results
    
    def print_detailed_results(self, y_true: List[str], y_pred: List[str]):
        """打印详细结果"""
        results = self.evaluate_with_details(y_true, y_pred)
        
        print("=== 关系抽取评估结果 ===")
        print(f"总样本数: {results['total_predictions']}")
        print(f"正确预测数: {results['correct_predictions']}")
        print(f"准确率: {results['accuracy']:.4f}")
        print()
        
        print("=== 各项指标 ===")
        print(f"Macro Precision: {results['macro_precision']:.4f}")
        print(f"Macro Recall: {results['macro_recall']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print()
        print(f"Micro Precision: {results['micro_precision']:.4f}")
        print(f"Micro Recall: {results['micro_recall']:.4f}")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        print()
        
        print("=== 真实标签分布 ===")
        for label, count in results['true_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")
        print()
        
        print("=== 预测标签分布 ===")
        for label, count in results['pred_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")


class NEREvaluator(ClassificationEvaluator):
    """命名实体识别评估器"""
    
    def __init__(self, relation_labels: Dict[str, str] = None):
        super().__init__()
        self.relation_labels = relation_labels or {}
    
    def evaluate_with_details(self, y_true: List[str], y_pred: List[str], 
                             data: pd.DataFrame = None) -> Dict:
        """带详细信息的评估"""
        # 基础评估

        print("True labels length:", len(y_true))
        print("Predicted labels length:", len(y_pred))
        results = self.evaluate(y_true, y_pred)
        
        # 添加关系类型统计
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()
        
        results['true_label_distribution'] = true_counts.to_dict()
        results['pred_label_distribution'] = pred_counts.to_dict()
        
        # 正确预测的样本数
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        results['correct_predictions'] = correct_predictions
        results['total_predictions'] = len(y_true)
        
        return results
    
    def print_detailed_results(self, y_true: List[str], y_pred: List[str]):
        """打印详细结果"""
        results = self.evaluate_with_details(y_true, y_pred)
        
        print("=== 关系抽取评估结果 ===")
        print(f"总样本数: {results['total_predictions']}")
        print(f"正确预测数: {results['correct_predictions']}")
        print(f"准确率: {results['accuracy']:.4f}")
        print()
        
        print("=== 各项指标 ===")
        print(f"Macro Precision: {results['macro_precision']:.4f}")
        print(f"Macro Recall: {results['macro_recall']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print()
        print(f"Micro Precision: {results['micro_precision']:.4f}")
        print(f"Micro Recall: {results['micro_recall']:.4f}")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        print()
        
        print("=== 真实标签分布 ===")
        for label, count in results['true_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")
        print()
        
        print("=== 预测标签分布 ===")
        for label, count in results['pred_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")

class QaEvaluator(ClassificationEvaluator):
    """分类任务评估器"""
    def __init__(self, relation_labels: Dict[str, str] = None):
        super().__init__()
        self.relation_labels = relation_labels or {}
    
    def evaluate_with_details(self, y_true: List[str], y_pred: List[str], 
                 labels: Optional[List[str]] = None) -> Dict:
        """评估分类结果"""
        if labels is None:
            labels = list(set(y_true + y_pred))

        # #####################
        # y_pred_fixed = []
        # for true, pred in zip(y_true, y_pred):
        #     pred_stripped = pred.strip().lower()
        #     true_stripped = true.strip().lower()
            
        #     if true_stripped in pred_stripped:
        #         y_pred_fixed.append(true.strip())  # 修正为标准答案格式
        #     else:
        #         y_pred_fixed.append(pred.strip())  # 原样保留但去掉多余空格
                
        # y_pred = y_pred_fixed
        # #####################

        results = self.evaluate(y_true, y_pred)

        # 基础指标
        print("y_true", y_true)
        print("y_pred", y_pred)
        
        # 添加关系类型统计
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()
        
        results['true_label_distribution'] = true_counts.to_dict()
        results['pred_label_distribution'] = pred_counts.to_dict()
        
        # 正确预测的样本数
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        results['correct_predictions'] = correct_predictions
        results['total_predictions'] = len(y_true)
        
        return results
    
    def print_detailed_results(self, y_true: List[str], y_pred: List[str]):
        """打印详细结果"""
        results = self.evaluate_with_details(y_true, y_pred)
        
        print("=== 关系抽取评估结果 ===")
        print(f"总样本数: {results['total_predictions']}")
        print(f"正确预测数: {results['correct_predictions']}")
        print(f"准确率: {results['accuracy']:.4f}")
        print()
        
        print("=== 各项指标 ===")
        print(f"Macro Precision: {results['macro_precision']:.4f}")
        print(f"Macro Recall: {results['macro_recall']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print()
        print(f"Micro Precision: {results['micro_precision']:.4f}")
        print(f"Micro Recall: {results['micro_recall']:.4f}")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        print()
        
        print("=== 真实标签分布 ===")
        for label, count in results['true_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")
        print()
        
        print("=== 预测标签分布 ===")
        for label, count in results['pred_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")

class EvaluatorFactory:
    """评估器工厂"""
    
    @staticmethod
    def create_evaluator(task_type: str, **kwargs):
        """创建评估器实例"""
        if task_type == "classification":
            return ClassificationEvaluator()
        elif task_type == "relation_extraction":
            return RelationExtractionEvaluator(**kwargs)
        elif task_type == "ner":
            return NEREvaluator()
        elif task_type == "qa":
            return QaEvaluator()
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")