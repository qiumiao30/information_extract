# tasks.py
"""
Task Management Module - Integrates all components to execute specific tasks.
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
    """Base class for tasks."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.model = None
        self.data_processor = None
        self.evaluator = None
        self.prompt_manager = PromptManager()

    def setup(self, **kwargs):
        """Sets up the task components."""
        raise NotImplementedError

    def run(self, **kwargs):
        """Runs the task."""
        raise NotImplementedError

class RelationExtractionTask(BaseTask):
    """Relation Extraction Task"""

    def __init__(self):
        super().__init__("relation_extraction")

    def setup(self, model_name: str = None, data_source: str = "custom", **kwargs):
        """Sets up the relation extraction task."""
        print(f"Setting up relation extraction task...")

        # Initialize model
        self.model = ModelFactory.create_model("relation_extraction", model_name)

        # Initialize data processor
        if data_source == "chemport":
            self.data_processor = DataProcessorFactory.create_processor("chemport")
        elif data_source == "ddi":
            self.data_processor = DataProcessorFactory.create_processor("ddi")
        elif data_source == "gad":
            self.data_processor = DataProcessorFactory.create_processor("gad")
        else:
            self.data_processor = DataProcessorFactory.create_processor("custom", **kwargs)

        # Initialize evaluator
        self.evaluator = EvaluatorFactory.create_evaluator(
            "relation_extraction",
            relation_labels=RELATION_LABELS
        )

        print("Task setup complete!")

    def run(self, max_samples: int = None, verbose: bool = True, data_source='chemport') -> Dict:
        """Runs the relation extraction task."""
        if not all([self.model, self.data_processor, self.evaluator]):
            raise ValueError("Please call the setup() method first to set up task components.")

        max_samples = max_samples or EVALUATION_CONFIG["max_samples"]

        print(f"=== Starting Relation Extraction Task ===")

        # 1. Load data
        print("1. Loading data...")
        test_data = self.data_processor.load_data()
        test_data = self.data_processor.process_data(test_data)
        if test_data.empty:
            print("Error: Data loading failed!")
            return {}

        # Limit sample size
        test_data = test_data.head(max_samples)
        print(f"Number of test samples: {len(test_data)}")

        # 2. Execute predictions
        print("\n2. Executing relation extraction...")
        predictions = []
        if data_source == "chemport":
            true_labels = test_data['relation'].tolist()
        elif data_source == "ddi":
            true_labels = test_data['label'].tolist()
            # Convert to uppercase
            true_labels = [label.upper() for label in true_labels]
            # print(true_labels)
        elif data_source == "gad":
            true_labels = test_data['label'].tolist()
            # # Convert to uppercase
            # true_labels = [label.upper() for label in true_labels]

        for idx, row in test_data.iterrows():
            # Generate prompt
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

            # Execute prediction
            pred_relation = self.model.extract_relation(prompt)

            print(f"Sample {idx+1} predicted relation: {pred_relation}")
            predictions.append(pred_relation)

            if data_source == "chemport":
                if verbose:
                    print(f"Sample {idx+1}:")
                    print(f"  PMID: {row.get('pmid', 'N/A')}")
                    print(f"  Text: {row['text'][:100]}...")
                    print(f"  Entity 1: {row['entity1']} ({row.get('entity1_type', 'Unknown')})")
                    print(f"  Entity 2: {row['entity2']} ({row.get('entity2_type', 'Unknown')})")
                    print(f"  True Relation: {row['relation']} ({RELATION_LABELS.get(row['relation'], 'unknown')})")
                    print(f"  Predicted Relation: {pred_relation} ({RELATION_LABELS.get(pred_relation, 'unknown')})")
                    print(f"  Correct: {'✓' if pred_relation == row['relation'] else '✗'}")
                    print()
            elif data_source == "ddi":
                if verbose:
                    print(f"Sample {idx+1}:")
                    print(f"Sentence: {row['sentence']}")
                    print(f"  True Relation: {row['label'].upper()}")
                    print(f"  Predicted Relation: {pred_relation}")
                    print(f"  Correct: {'✓' if pred_relation == row['label'].upper() else '✗'}")
                    print()
            elif data_source == "gad":
                if verbose:
                    print(f"Sample {idx+1}:")
                    print(f"Sentence: {row['sentence']}")
                    print(f"  True Relation: {row['label']}")
                    print(f"  Predicted Relation: {pred_relation}")
                    print(f"  Correct: {'✓' if pred_relation == row['label'] else '✗'}")
                    print()

        # 3. Evaluate results
        print("3. Evaluating results...")
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

            # Detailed classification report
            detailed_report = self.evaluator.get_detailed_report(true_labels, predictions)
            print("\n=== Detailed Classification Report ===")
            print(detailed_report)

        return results


class NERTask(BaseTask):
    """Named Entity Recognition Task"""

    def __init__(self):
        super().__init__("ner")

    def setup(self, model_name: str = None, data_source: str="ncbi", **kwargs):
        """Sets up the NER task."""
        print(f"Setting up Named Entity Recognition task...")

        # Initialize model
        self.model = ModelFactory.create_model("ner", model_name)

        # Initialize data processor
        if data_source == "bc2gm":
            self.data_processor = DataProcessorFactory.create_processor("bc2gm", **kwargs)
        elif data_source == "jnlpba":
            self.data_processor = DataProcessorFactory.create_processor("jnlpba", **kwargs)
        elif data_source == "ncbi":
            self.data_processor = DataProcessorFactory.create_processor("ncbi", **kwargs)

        # Add NER data processor
        self.evaluator = EvaluatorFactory.create_evaluator("ner",
                                                           ner_labels=RELATION_LABELS)

        print("Task setup complete!")

    def run(self, max_samples: int=None, verbose: bool = True, data_source: str="ncbi") -> Dict:
        """Runs the NER task."""
        if not all([self.model, self.data_processor, self.evaluator]):
            raise ValueError("Please call the setup() method first to set up task components.")

        max_samples = max_samples or EVALUATION_CONFIG["max_samples"]

        print(f"=== Starting NER Task ===") # Changed from Relation Extraction

        # 1. Load data
        print("1. Loading data...")
        if data_source == 'ncbi':
            test_data = self.data_processor.load_data()
            test_data = self.data_processor.process_data(test_data)
        else:
            test_data = self.data_processor.load_data()
        if test_data.empty:
            print("Error: Data loading failed!")
            return {}

        # Limit sample size
        test_data = test_data.head(max_samples)
        print(f"Number of test samples: {len(test_data)}")

        # 2. Execute predictions
        print("\n2. Executing entity extraction...") # Changed from relation extraction
        predictions = []
        true_labels = test_data['ner_tags'].tolist()
        true_labels = [label for sublist in true_labels for label in sublist]
        # Convert to uppercase
        true_labels = [label.upper() for label in true_labels]
        if data_source == "bc2gm":
            converted_tags = [
                'O' if tag == '0' else
                'B-DISEASE' if tag == '1' else
                'I-DISEASE' if tag == '2' else tag
                for tag in true_labels
            ]
            true_labels = converted_tags

        print("Length of true entity labels:", len(true_labels))

        for idx, row in test_data.iterrows():
            # Generate prompt
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

            print(f"Sample {idx+1} predicted entities: {pred_ner}") # Changed from relation
            ner_tags_upper = [label.upper() for label in row['ner_tags']]

            converted_tags = [
                'O' if tag == '0' else
                'B-DISEASE' if tag == '1' else
                'I-DISEASE' if tag == '2' else tag
                for tag in ner_tags_upper
            ]
            ner_tags_upper = converted_tags

            if verbose:
                print(f"Sample {idx+1}:")
                print(f"  PMID: {row.get('pmid', 'N/A')}")
                print(f"  Text: {row['text'][:100]}...")
                print(f"  Tokens: {row['tokens']}")
                print(f"  True Entity Tags: {ner_tags_upper}")
                print(f"  Predicted Entity Tags: {pred_ner}")

                print(f"  Correct: {'✓' if pred_ner == ner_tags_upper else '✗'}")

                # Check if the number of true and predicted entity tags are consistent
                if len(ner_tags_upper) == len(pred_ner):
                    print(f"  Entity counts are consistent: {len(ner_tags_upper)}")

                # If still inconsistent, force alignment
                if len(pred_ner) != len(ner_tags_upper):
                    # print(f"  ⚠️ Retried {MAX_RETRIES} times, still inconsistent, performing alignment.")
                    print(f"  Note: Mismatch between true and predicted entity tag counts!")
                    if len(pred_ner) > len(ner_tags_upper):
                        pred_ner = pred_ner[:len(ner_tags_upper)]
                    else:
                        pred_ner += ['O'] * (len(ner_tags_upper) - len(pred_ner))
                    print(f"  ✅ Aligned predicted entity tags: {pred_ner}")

                print()
            # self.model.clear_history()  # Clear model history to avoid affecting subsequent predictions
            predictions.append(pred_ner)
        predictions = [label for sublist in predictions for label in sublist]

        # 3. Evaluate results
        print("3. Evaluating results...")
        results = self.evaluator.evaluate_with_details(true_labels, predictions, test_data)

        if verbose:
            self.evaluator.print_detailed_results(true_labels, predictions)

            # Detailed classification report
            detailed_report = self.evaluator.get_detailed_report(true_labels, predictions)
            print("\n=== Detailed Classification Report ===")
            print(detailed_report)

        return results


class QATask(BaseTask):
    """Question Answering Task""" # Changed from QA and Reasoning

    def __init__(self):
        super().__init__("qa")

    def setup(self, model_name: str = None, data_source: str="pubmedqa", **kwargs):
        """Sets up the Question Answering task."""
        print(f"Setting up Question Answering task...")

        # Initialize model
        self.model = ModelFactory.create_model("qa", model_name)

        # Initialize data processor
        if data_source == "pubmedqa":
            self.data_processor = DataProcessorFactory.create_processor("pubmedqa", **kwargs)
        elif data_source == "bioasq":
            self.data_processor = DataProcessorFactory.create_processor("bioasq", **kwargs)
        elif data_source == "medqa":
            self.data_processor = DataProcessorFactory.create_processor("medqa", **kwargs)

        # Initialize evaluator
        self.evaluator = EvaluatorFactory.create_evaluator(
            "qa",
            relation_labels=RELATION_LABELS
        )

        print("Task setup complete!")

    def run(self, max_samples: int=None, verbose: bool = True, data_source: str="pubmedqa", **kwargs) -> Dict:
        """Runs the Question Answering task."""
        print("=== Starting Question Answering Task ===") # Changed from Text Reasoning
        if not all([self.model, self.data_processor, self.evaluator]):
            raise ValueError("Please call the setup() method first to set up task components.")

        max_samples = max_samples or EVALUATION_CONFIG["max_samples"]

        # 1. Load data
        print("1. Loading data...")
        if data_source == 'pubmedqa':
            test_data = self.data_processor.load_data()
            test_data = test_data.iloc[500:-1]
        elif data_source == 'medqa':
            test_data = self.data_processor.load_data()
        else:
            test_data = self.data_processor.load_data()
            test_data = self.data_processor.process_data(test_data)
        if test_data.empty:
            print("Error: Data loading failed!")
            return {}

        # Limit sample size
        test_data = test_data.head(max_samples)
        print(f"Number of test samples: {len(test_data)}")

        # 2. Execute predictions
        print("\n2. Executing predictions...") # Changed from relation extraction
        predictions = []
        if data_source == "pubmedqa" or data_source == "medqa":
            true_labels = test_data['final_decision'].tolist()
            true_labels = [label.lower() for label in true_labels]
        else:
            true_labels = test_data['answer'].tolist()
            true_labels = [label.lower() for label in true_labels]

        print(true_labels)

        for idx, row in test_data.iterrows():
            # Generate prompt
            prompt = self.prompt_manager.get_qa_prompt(
                data_source, row['question'], row['context']
            )

            # Execute prediction
            pred_qa = self.model.extract_answer(prompt)
            print(f"Sample {idx+1} predicted answer: {pred_qa}") # Changed from relation
            print(f"Sample {idx+1} true answer: {row['final_decision'] if (data_source == 'pubmedqa' or data_source == 'medqa') else row['answer']}") # Changed from relation
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
                # Take the last 300 elements of true_labels
                if len(true_labels) > 300:
                    true_labels = true_labels[-300:]
                    print(len(true_labels))
                if len(predictions) > 300:
                    predictions = predictions[-300:]
                    print(len(predictions))
            eval_results = self.evaluator.evaluate_with_details(true_labels, predictions)
            results.update(eval_results)

            print("=== Evaluation Results ===")
            self.evaluator.print_results()

        return results

class ClassificationTask(BaseTask):
    """Text Classification Task"""

    def __init__(self):
        super().__init__("classification")

    def setup(self, model_name: str = None, classes: List[str] = None, **kwargs):
        """Sets up the classification task."""
        print(f"Setting up text classification task...")

        self.model = ModelFactory.create_model("classification", model_name)
        self.classes = classes or ["positive", "negative", "neutral"]
        self.evaluator = EvaluatorFactory.create_evaluator("classification")

        print("Task setup complete!")

    def run(self, text_list: List[str], true_labels: List[str] = None, **kwargs) -> Dict:
        """Runs the classification task."""
        print("=== Starting Text Classification Task ===")

        predictions = []
        for text in text_list:
            prompt = self.prompt_manager.get_classification_prompt(text, self.classes)
            pred_class = self.model.classify(prompt, self.classes)
            predictions.append(pred_class)

        results = {"predictions": predictions}

        if true_labels:
            eval_results = self.evaluator.evaluate(true_labels, predictions)
            results.update(eval_results)

            print("=== Evaluation Results ===")
            self.evaluator.print_results()

        return results


class TaskFactory:
    """Task Factory"""

    @staticmethod
    def create_task(task_type: str):
        """Creates a task instance."""
        if task_type == "relation_extraction":
            return RelationExtractionTask()
        elif task_type == "ner":
            return NERTask()
        elif task_type == "classification":
            return ClassificationTask()
        elif task_type == "qa":
            return QATask()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")


def quick_relation_extraction(max_samples: int = 20, verbose: bool = True) -> Dict:
    """Quickly runs a relation extraction task."""
    task = TaskFactory.create_task("relation_extraction")
    task.setup()
    return task.run(max_samples=max_samples, verbose=verbose)


def quick_classification(text_list: List[str], true_labels: List[str] = None,
                           classes: List[str] = None) -> Dict:
    """Quickly runs a classification task."""
    task = TaskFactory.create_task("classification")
    task.setup(classes=classes)
    return task.run(text_list, true_labels)


def quick_ner(text_list: List[str]) -> Dict:
    """Quickly runs a NER task."""
    task = TaskFactory.create_task("ner")
    task.setup()
    return task.run(text_list)

def quick_qa(text_list: List[str]) -> Dict:
    """Quickly runs a QA task.""" # Changed from NER
    task = TaskFactory.create_task("qa")
    task.setup()
    return task.run(text_list)
