# main.py
"""
Main program entry - Demonstrates how to use modular code structure
"""

from tasks import TaskFactory, quick_relation_extraction, quick_classification, quick_ner
from models import ModelFactory
from data_processors import DataProcessorFactory
from evaluators import EvaluatorFactory
from prompts import PromptManager
import argparse


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-task NLP System')
    parser.add_argument('--task', type=str, choices=['relation_extraction', 'ner', 'classification', 'qa'], 
                       default='relation_extraction', help='Select task type')
    parser.add_argument('--max_samples', type=int, default=5, help='Maximum number of samples')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--data_source', type=str, default='chemport', help='data source')
    
    
    args = parser.parse_args()
    
    print(f"=== Multi-task NLP System ===")
    print(f"Task type: {args.task}")
    print(f"Maximum samples: {args.max_samples}")
    print(f"Model: {args.model_name or 'default'}")
    print()
    
    if args.task == 'relation_extraction':
        run_relation_extraction_demo(args.max_samples, args.verbose, args.model_name, args.data_source)
    elif args.task == 'ner':
        run_ner_demo(args.max_samples, args.verbose, args.model_name, args.data_source)
    elif args.task == 'classification':
        run_classification_demo(args.max_samples, args.verbose, args.model_name, args.data_source)
    elif args.task == 'qa':
        run_qa_demo(args.max_samples, args.verbose, args.model_name, args.data_source)


def run_relation_extraction_demo(max_samples: int = 20, verbose: bool = True, model_name: str = None, data_source: str = None):
    """Run relation extraction demo"""
    print("=== Relation Extraction Task Demo ===\n")
    
    print("\n" + "="*50)
    print("Method 2: Using complete task interface")
    
    task = TaskFactory.create_task("relation_extraction")
    task.setup(model_name=model_name, data_source=data_source)
    results = task.run(max_samples=max_samples, verbose=verbose, data_source=data_source)
    
    print(f"\nFinal Results Summary:")
    print(f"Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"Macro F1: {results.get('macro_f1', 0):.4f}")
    print(f"Micro F1: {results.get('micro_f1', 0):.4f}")


def run_ner_demo(max_samples: int = 5, verbose: bool = True, model_name: str = None, data_source: str = None):
    """Run NER demo"""
    print("=== Named Entity Recognition Task Demo ===\n")

    # Method 2: Using complete task interface
    print("\n" + "="*50)
    print("Method 2: Using complete task interface")

    task = TaskFactory.create_task("ner")
    task.setup(model_name=model_name, data_source=data_source)
    results = task.run(max_samples=max_samples, verbose=verbose, data_source=data_source)
    
    print(f"\nFinal Results Summary:")
    print(f"Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"Macro F1: {results.get('macro_f1', 0):.4f}")
    print(f"Micro F1: {results.get('micro_f1', 0):.4f}")
    # print("NER Results:", results)

def run_qa_demo(max_samples: int = 5, verbose: bool = True, model_name: str = None, data_source: str = None):
    """Run QA demo"""
    print("=== Question Answering Task Demo ===\n")

    # Method 2: Using complete task interface
    print("\n" + "="*50)
    print("Method 2: Using complete task interface")

    task = TaskFactory.create_task("qa")
    task.setup(model_name=model_name, data_source=data_source)
    results = task.run(max_samples=max_samples, verbose=verbose, data_source=data_source)
    
    print(f"\nFinal Results Summary:")
    print(f"Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"Macro F1: {results.get('macro_f1', 0):.4f}")
    print(f"Micro F1: {results.get('micro_f1', 0):.4f}")
    # print("NER Results:", results)

def run_classification_demo():
    """Run classification demo"""
    print("=== Text Classification Task Demo ===\n")
    
    sample_texts = [
        "This drug is very effective for treating the disease.",
        "The side effects are terrible and dangerous.",
        "The medication shows moderate improvement in symptoms."
    ]
    
    true_labels = ["positive", "negative", "neutral"]
    classes = ["positive", "negative", "neutral"]
    
    results = quick_classification(sample_texts, true_labels, classes)
    print("Classification Results:", results)


def custom_task_example():
    """Custom task example"""
    print("=== Custom Task Example ===\n")
    
    # 1. Create custom model
    model = ModelFactory.create_model("base", "Qwen/Qwen2.5-7B-Instruct")
    
    # 2. Create custom data processor
    data_processor = DataProcessorFactory.create_processor("custom", 
                                                          dataset_name="your_dataset")
    
    # 3. Create custom evaluator
    evaluator = EvaluatorFactory.create_evaluator("classification")
    
    # 4. Use prompt manager
    prompt_manager = PromptManager()
    
    print("Custom components created successfully!")


def component_usage_examples():
    """Component usage examples"""
    print("=== Component Usage Examples ===\n")
    
    # 1. Use model separately
    print("1. Use model separately:")
    model = ModelFactory.create_model("relation_extraction")
    # model.load_model()  # If needed
    
    # 2. Use data processor separately
    print("2. Use data processor separately:")
    processor = DataProcessorFactory.create_processor("chemprot")
    sample_data = processor.get_sample_data(n_samples=3)
    print(f"Sample data shape: {sample_data.shape}")
    
    # 3. Use prompt manager separately
    print("3. Use Prompt manager separately:")
    prompt_manager = PromptManager()
    prompt = prompt_manager.get_relation_extraction_prompt(
        "Aspirin inhibits COX-2.", "Aspirin", "COX-2"
    )
    print(f"Prompt length: {len(prompt)} characters")
    
    # 4. Use evaluator separately
    print("4. Use evaluator separately:")
    evaluator = EvaluatorFactory.create_evaluator("classification")
    y_true = ["positive", "negative", "positive"]
    y_pred = ["positive", "positive", "positive"]
    results = evaluator.evaluate(y_true, y_pred)
    print(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
