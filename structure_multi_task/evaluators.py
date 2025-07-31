# evaluators.py
"""
Evaluation Module - Uniformly manages various evaluation metrics and methods.
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
    """Base evaluator class"""

    def __init__(self):
        self.results = {}

    def evaluate(self, y_true: List, y_pred: List, **kwargs) -> Dict:
        """Evaluates prediction results."""
        raise NotImplementedError

    def print_results(self):
        """Prints the evaluation results."""
        for metric, value in self.results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")


class ClassificationEvaluator(BaseEvaluator):
    """Classification task evaluator."""

    def evaluate(self, y_true: List[str], y_pred: List[str],
                 labels: Optional[List[str]] = None) -> Dict:
        """Evaluates classification results."""
        if labels is None:
            labels = list(set(y_true + y_pred))

        # Basic metric
        accuracy = accuracy_score(y_true, y_pred)

        # Macro-average metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )

        # Micro-average metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='micro', zero_division=0
        )

        # Weighted-average metrics
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
        """Gets a detailed classification report."""
        if labels is None:
            labels = list(set(y_true + y_pred))

        print("labels", labels)

        return classification_report(y_true, y_pred, target_names=labels, zero_division=0)

    def get_confusion_matrix(self, y_true: List[str], y_pred: List[str],
                               labels: Optional[List[str]] = None) -> np.ndarray:
        """Gets the confusion matrix."""
        if labels is None:
            labels = list(set(y_true + y_pred))

        return confusion_matrix(y_true, y_pred, labels=labels)

class RelationExtractionEvaluator(ClassificationEvaluator):
    """Relation Extraction evaluator."""

    def __init__(self, relation_labels: Dict[str, str] = None):
        super().__init__()
        self.relation_labels = relation_labels or {}

    def evaluate_with_details(self, y_true: List[str], y_pred: List[str],
                                  data: pd.DataFrame = None) -> Dict:
        """Evaluates with detailed information."""
        # Basic evaluation
        results = self.evaluate(y_true, y_pred)

        # Add relation type statistics
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()

        results['true_label_distribution'] = true_counts.to_dict()
        results['pred_label_distribution'] = pred_counts.to_dict()

        # Number of correctly predicted samples
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        results['correct_predictions'] = correct_predictions
        results['total_predictions'] = len(y_true)

        return results

    def print_detailed_results(self, y_true: List[str], y_pred: List[str]):
        """Prints detailed results."""
        results = self.evaluate_with_details(y_true, y_pred)

        print("=== Relation Extraction Evaluation Results ===")
        print(f"Total Samples: {results['total_predictions']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print()

        print("=== Performance Metrics ===")
        print(f"Macro Precision: {results['macro_precision']:.4f}")
        print(f"Macro Recall: {results['macro_recall']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print()
        print(f"Micro Precision: {results['micro_precision']:.4f}")
        print(f"Micro Recall: {results['micro_recall']:.4f}")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        print()

        print("=== True Label Distribution ===")
        for label, count in results['true_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")
        print()

        print("=== Predicted Label Distribution ===")
        for label, count in results['pred_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")


class NEREvaluator(ClassificationEvaluator):
    """Named Entity Recognition evaluator."""

    def __init__(self, relation_labels: Dict[str, str] = None):
        super().__init__()
        self.relation_labels = relation_labels or {}

    def evaluate_with_details(self, y_true: List[str], y_pred: List[str],
                                  data: pd.DataFrame = None) -> Dict:
        """Evaluates with detailed information."""
        # Basic evaluation

        print("True labels length:", len(y_true))
        print("Predicted labels length:", len(y_pred))
        results = self.evaluate(y_true, y_pred)

        # Add relation type statistics
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()

        results['true_label_distribution'] = true_counts.to_dict()
        results['pred_label_distribution'] = pred_counts.to_dict()

        # Number of correctly predicted samples
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        results['correct_predictions'] = correct_predictions
        results['total_predictions'] = len(y_true)

        return results

    def print_detailed_results(self, y_true: List[str], y_pred: List[str]):
        """Prints detailed results."""
        results = self.evaluate_with_details(y_true, y_pred)

        print("=== NER Evaluation Results ===") # Changed from Relation Extraction
        print(f"Total Tokens: {results['total_predictions']}") # Changed from Total Samples
        print(f"Correct Predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print()

        print("=== Performance Metrics ===")
        print(f"Macro Precision: {results['macro_precision']:.4f}")
        print(f"Macro Recall: {results['macro_recall']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print()
        print(f"Micro Precision: {results['micro_precision']:.4f}")
        print(f"Micro Recall: {results['micro_recall']:.4f}")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        print()

        print("=== True Label Distribution ===")
        for label, count in results['true_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")
        print()

        print("=== Predicted Label Distribution ===")
        for label, count in results['pred_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")


class QaEvaluator(ClassificationEvaluator):
    """Question Answering evaluator.""" # Changed from Classification task evaluator
    def __init__(self, relation_labels: Dict[str, str] = None):
        super().__init__()
        self.relation_labels = relation_labels or {}

    def evaluate_with_details(self, y_true: List[str], y_pred: List[str],
                                labels: Optional[List[str]] = None) -> Dict:
        """Evaluates QA results.""" # Changed from classification results
        if labels is None:
            labels = list(set(y_true + y_pred))

        # #####################
        # y_pred_fixed = []
        # for true, pred in zip(y_true, y_pred):
        #     pred_stripped = pred.strip().lower()
        #     true_stripped = true.strip().lower()
        #
        #     if true_stripped in pred_stripped:
        #         y_pred_fixed.append(true.strip())  # Correct to standard answer format
        #     else:
        #         y_pred_fixed.append(pred.strip())  # Keep as is but strip extra spaces
        #
        # y_pred = y_pred_fixed
        # #####################

        results = self.evaluate(y_true, y_pred)

        # Basic metrics
        print("y_true", y_true)
        print("y_pred", y_pred)

        # Add label type statistics
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()

        results['true_label_distribution'] = true_counts.to_dict()
        results['pred_label_distribution'] = pred_counts.to_dict()

        # Number of correctly predicted samples
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        results['correct_predictions'] = correct_predictions
        results['total_predictions'] = len(y_true)

        return results

    def print_detailed_results(self, y_true: List[str], y_pred: List[str]):
        """Prints detailed results."""
        results = self.evaluate_with_details(y_true, y_pred)

        print("=== QA Evaluation Results ===") # Changed from Relation Extraction
        print(f"Total Samples: {results['total_predictions']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print()

        print("=== Performance Metrics ===")
        print(f"Macro Precision: {results['macro_precision']:.4f}")
        print(f"Macro Recall: {results['macro_recall']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print()
        print(f"Micro Precision: {results['micro_precision']:.4f}")
        print(f"Micro Recall: {results['micro_recall']:.4f}")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        print()

        print("=== True Label Distribution ===")
        for label, count in results['true_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")
        print()

        print("=== Predicted Label Distribution ===")
        for label, count in results['pred_label_distribution'].items():
            label_name = self.relation_labels.get(label, label)
            print(f"{label} ({label_name}): {count}")


class EvaluatorFactory:
    """Evaluator Factory"""

    @staticmethod
    def create_evaluator(task_type: str, **kwargs):
        """Creates an evaluator instance."""
        if task_type == "classification":
            return ClassificationEvaluator()
        elif task_type == "relation_extraction":
            return RelationExtractionEvaluator(**kwargs)
        elif task_type == "ner":
            return NEREvaluator()
        elif task_type == "qa":
            return QaEvaluator()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
