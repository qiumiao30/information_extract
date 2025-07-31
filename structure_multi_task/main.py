# main.py
"""
主程序入口 - 演示如何使用模块化的代码结构
"""

from tasks import TaskFactory, quick_relation_extraction, quick_classification, quick_ner
from models import ModelFactory
from data_processors import DataProcessorFactory
from evaluators import EvaluatorFactory
from prompts import PromptManager
import argparse


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多任务NLP系统')
    parser.add_argument('--task', type=str, choices=['relation_extraction', 'ner', 'classification', 'qa'], 
                       default='relation_extraction', help='选择任务类型')
    parser.add_argument('--max_samples', type=int, default=5, help='最大样本数')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--data_source', type=str, default='chemport', help='data source')
    
    
    args = parser.parse_args()
    
    print(f"=== 多任务NLP系统 ===")
    print(f"任务类型: {args.task}")
    print(f"最大样本数: {args.max_samples}")
    print(f"模型: {args.model_name or 'default'}")
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
    """运行关系抽取演示"""
    print("=== 关系抽取任务演示 ===\n")
    
    # # 方法1: 使用快速接口
    # print("方法1: 使用快速接口")
    # results = quick_relation_extraction(max_samples=max_samples, verbose=verbose)
    
    # 方法2: 使用完整的任务接口
    print("\n" + "="*50)
    print("方法2: 使用完整的任务接口")
    
    task = TaskFactory.create_task("relation_extraction")
    task.setup(model_name=model_name, data_source=data_source)
    results = task.run(max_samples=max_samples, verbose=verbose, data_source=data_source)
    
    print(f"\n最终结果汇总:")
    print(f"准确率: {results.get('accuracy', 0):.4f}")
    print(f"宏平均F1: {results.get('macro_f1', 0):.4f}")
    print(f"微平均F1: {results.get('micro_f1', 0):.4f}")


def run_ner_demo(max_samples: int = 5, verbose: bool = True, model_name: str = None, data_source: str = None):
    """运行NER演示"""
    print("=== 命名实体识别任务演示 ===\n")

    # 方法2: 使用完整的任务接口
    print("\n" + "="*50)
    print("方法2: 使用完整的任务接口")

    task = TaskFactory.create_task("ner")
    task.setup(model_name=model_name, data_source=data_source)
    results = task.run(max_samples=max_samples, verbose=verbose, data_source=data_source)
    
    print(f"\n最终结果汇总:")
    print(f"准确率: {results.get('accuracy', 0):.4f}")
    print(f"宏平均F1: {results.get('macro_f1', 0):.4f}")
    print(f"微平均F1: {results.get('micro_f1', 0):.4f}")
    # print("NER结果:", results)

def run_qa_demo(max_samples: int = 5, verbose: bool = True, model_name: str = None, data_source: str = None):
    """运行QA演示"""
    print("=== 问答推理任务演示 ===\n")

    # 方法2: 使用完整的任务接口
    print("\n" + "="*50)
    print("方法2: 使用完整的任务接口")

    task = TaskFactory.create_task("qa")
    task.setup(model_name=model_name, data_source=data_source)
    results = task.run(max_samples=max_samples, verbose=verbose, data_source=data_source)
    
    print(f"\n最终结果汇总:")
    print(f"准确率: {results.get('accuracy', 0):.4f}")
    print(f"宏平均F1: {results.get('macro_f1', 0):.4f}")
    print(f"微平均F1: {results.get('micro_f1', 0):.4f}")
    # print("NER结果:", results)

def run_classification_demo():
    """运行分类演示"""
    print("=== 文本分类任务演示 ===\n")
    
    sample_texts = [
        "This drug is very effective for treating the disease.",
        "The side effects are terrible and dangerous.",
        "The medication shows moderate improvement in symptoms."
    ]
    
    true_labels = ["positive", "negative", "neutral"]
    classes = ["positive", "negative", "neutral"]
    
    results = quick_classification(sample_texts, true_labels, classes)
    print("分类结果:", results)


def custom_task_example():
    """自定义任务示例"""
    print("=== 自定义任务示例 ===\n")
    
    # 1. 创建自定义模型
    model = ModelFactory.create_model("base", "Qwen/Qwen2.5-7B-Instruct")
    
    # 2. 创建自定义数据处理器
    data_processor = DataProcessorFactory.create_processor("custom", 
                                                          dataset_name="your_dataset")
    
    # 3. 创建自定义评估器
    evaluator = EvaluatorFactory.create_evaluator("classification")
    
    # 4. 使用prompt管理器
    prompt_manager = PromptManager()
    
    print("自定义组件创建完成!")


def component_usage_examples():
    """组件使用示例"""
    print("=== 组件使用示例 ===\n")
    
    # 1. 单独使用模型
    print("1. 单独使用模型:")
    model = ModelFactory.create_model("relation_extraction")
    # model.load_model()  # 如果需要的话
    
    # 2. 单独使用数据处理器
    print("2. 单独使用数据处理器:")
    processor = DataProcessorFactory.create_processor("chemprot")
    sample_data = processor.get_sample_data(n_samples=3)
    print(f"样本数据形状: {sample_data.shape}")
    
    # 3. 单独使用prompt管理器
    print("3. 单独使用Prompt管理器:")
    prompt_manager = PromptManager()
    prompt = prompt_manager.get_relation_extraction_prompt(
        "Aspirin inhibits COX-2.", "Aspirin", "COX-2"
    )
    print(f"Prompt长度: {len(prompt)} 字符")
    
    # 4. 单独使用评估器
    print("4. 单独使用评估器:")
    evaluator = EvaluatorFactory.create_evaluator("classification")
    y_true = ["positive", "negative", "positive"]
    y_pred = ["positive", "positive", "positive"]
    results = evaluator.evaluate(y_true, y_pred)
    print(f"评估结果: {results}")


if __name__ == "__main__":
    # 如果想要运行命令行参数版本，取消注释下面这行
    main()
    
    # # 或者直接运行演示
    # print("运行关系抽取演示...")
    # run_relation_extraction_demo(max_samples=5, verbose=True)
    
    # print("\n" + "="*60)
    # print("运行组件使用示例...")
    # component_usage_examples()