# Multi-task NLP System

A structured multi-task natural language processing system based on LLM, supporting relation extraction, named entity recognition, text classification and other tasks.

## 🚀 Features

- **Modular Design**: Independent components, easy to maintain and extend
- **Multi-task Support**: Supports relation extraction, NER, text classification and other tasks
- **Scalability**: Easy to add new models, data processors and evaluators
- **Unified Interface**: Provides consistent API interface, lowering the barrier to use
- **Prompt Management**: Centralized management of all prompt templates for easy optimization

## 📁 Project Structure

```
├── config.py              # Configuration file
├── prompts.py             # Prompt management
├── models.py              # Model management
├── data_processors.py     # Data processors
├── evaluators.py          # Evaluators
├── tasks.py               # Task management
├── main.py                # Main program entry
├── requirements.txt       # Dependencies file
└── README.md             # Project documentation
```

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Quick Start

```python
from tasks import quick_relation_extraction

# Run relation extraction task
results = quick_relation_extraction(max_samples=10, verbose=True)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### 2. Complete Task Workflow

```python
from tasks import TaskFactory

# Create task
task = TaskFactory.create_task("relation_extraction")

# Setup task
task.setup(model_name="Qwen/Qwen2.5-7B-Instruct", data_source="chemprot")

# Run task
results = task.run(max_samples=20, verbose=True)
```

### 3. Custom Components

```python
from models import ModelFactory
from data_processors import DataProcessorFactory
from evaluators import EvaluatorFactory
from prompts import PromptManager

# Create custom model
model = ModelFactory.create_model("relation_extraction")

# Create custom data processor
processor = DataProcessorFactory.create_processor("custom", 
                                                dataset_name="your_dataset")

# Create custom evaluator
evaluator = EvaluatorFactory.create_evaluator("classification")

# Use prompt manager
prompt_manager = PromptManager()
prompt = prompt_manager.get_relation_extraction_prompt(text, entity1, entity2)
```

## 🎯 Supported Tasks

### 1. Relation Extraction
- Supports ChemProt dataset
- 10 relation types recognition
- Complete evaluation metrics

### 2. Named Entity Recognition (NER)
- Extensible entity types
- Supports custom data formats

### 3. Question Answer

## 🔨 Command Line Usage

```bash
# Run relation extraction task
python main.py --task relation_extraction --max_samples 20 --verbose

# Run NER task
python main.py --task ner

# Run QA task
python main.py --task qa
```

## 📊 Configuration

Modify configuration in `config.py`:

```python
# Modify model configuration
MODEL_CONFIG = {
    "model_name": "your_model_name",
    "torch_dtype": "float16",
    "device_map": "auto"
}

# Modify relation labels
RELATION_LABELS = {
    'CPR:1': 'UPREGULATOR',
    # ... add more labels
}
```

## 🎨 Adding New Tasks

### 1. Create New Prompt

```python
# Add in prompts.py
@staticmethod
def get_your_task_prompt(text: str) -> str:
    return f"Your prompt template for {text}"
```

### 2. Create New Model Class

```python
# Add in models.py
class YourTaskModel(BaseModel):
    def your_method(self, prompt: str):
        response = self.generate_response(prompt)
        return self.parse_response(response)
```

### 3. Create New Task Class

```python
# Add in tasks.py
class YourTask(BaseTask):
    def setup(self, **kwargs):
        # Setup components
        pass
    
    def run(self, **kwargs):
        # Execute task
        pass
```

## 📈 Extending Data Processors

```python
class CustomDataProcessor(BaseDataProcessor):
    def load_data(self) -> pd.DataFrame:
        # Implement data loading logic
        pass
    
    def process_data(self, dataset) -> pd.DataFrame:
        # Implement data processing logic
        pass
```

## 📝 Evaluation Metrics

The system supports multiple evaluation metrics:
- Accuracy
- Precision
- Recall  
- F1-Score
- Macro/Micro/Weighted Average

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve the project!

## 📄 License

MIT License

## 🔗 Related Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [ChemProt Dataset](https://huggingface.co/datasets/bigbio/chemprot)
