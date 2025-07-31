# models.py
"""
Model Management Module - Manages all models uniformly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from config import MODEL_CONFIG
import ast
import re
from multi_agent import DCRMADEFramework


class BaseModel:
    """Base model class"""

    def __init__(self, model_name: str = None):
        """Initializes the model"""
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.tokenizer = None
        self.model = None
        self.device = None
        self.conversation_history = []  # Just add this line

    def load_model(self):
        """Loads the model"""
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
            device_map=MODEL_CONFIG["device_map"],
        )

        self.device = self.model.device
        print(f"Model loaded, device: {self.device}")

    def generate_response(self, prompt: str, **kwargs) -> dict:
        """
        Generates a response, supporting Qwen3's thinking mode.

        Args:
            prompt: The input prompt.
            enable_thinking: Whether to enable thinking mode, defaults to True.
            **kwargs: Other generation parameters.

        Returns:
            dict: A dictionary containing 'thinking_content' and 'content'.
        """
        if self.model is None:
            self.load_model()

        generation_config = {**MODEL_CONFIG["generation_config"], **kwargs}

        # Build the conversation format
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True  # Enable thinking mode
        )
        model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)

        # Conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            pad_token_id=self.tokenizer.eos_token_id  # Set explicitly to remove warnings
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            THINK_END_ID = self.tokenizer.convert_tokens_to_ids('</think>')
            index = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print(f"Thinking content: {thinking_content}")
        # print(f"Generated content: {content}")

        return {
            "thinking_content": thinking_content,
            "content": content
        }

class RelationExtractionModel(BaseModel):
    """Relation Extraction Model"""

    def __init__(self, model_name: str = None):
        super().__init__(model_name)
        self.valid_labels = ['CPR:1', 'CPR:2', 'CPR:3', 'CPR:4', 'CPR:5',
                               'CPR:6', 'CPR:7', 'CPR:9', 'CPR:10', "0", "1", "0", "DDI-effect", "DDI-mechanism", "DDI-advise"]

    def extract_relation(self, prompt: str) -> str:
        """Extracts relations"""
        result = self.generate_response(prompt)
        response = result.get("content", "")
        thinking_content = result.get("thinking_content", "")
        print(f"Original model output: {response}")
        response = response.strip().upper()

        print(f"think model: ", thinking_content)
        
        print(f"Model output: {response}")
        
        # # Parse the output to extract the relation label
        # for label in self.valid_labels:
        #     if label in response:
        #         return label
        
        # return 'false'  # Default to no relation
        return response

import re

def extract_and_normalize_values(label_list):
    """Normalize a list of predicted labels: keep valid ones, set others to 'O'."""
    valid_labels = ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA',
                      'B-CELL_LINE', 'I-CELL_LINE',
                      'B-CELL_TYPE', 'I-CELL_TYPE',
                      'B-PROTEIN', 'I-PROTEIN', 'I-DISEASE', 'B-DISEASE', 'B', 'I']
    # valid_labels = ['0', '1', '2']
    
    normalized = []
    for label in label_list:
        clean_label = label.strip().strip("'\"").upper()
        # Match ignoring case, but return label in correct format
        matched = False
        for valid in valid_labels:
            if clean_label == valid.upper():
                normalized.append(valid)
                matched = True
                break
        if not matched:
            normalized.append('O')  # fallback
    return normalized

def normalize_output_to_list(output):
    """
    Converts model output uniformly to a list of BIO labels.
    Supports raw strings (with spaces, newlines, mixed separators) and existing lists.
    """
    if isinstance(output, list):
        return output
    elif isinstance(output, str):
        # Replace various separators with a single space, then split
        output = output.replace("\n", " ").replace(",", " ").strip()
        output = " ".join(output.split())  # Remove extra spaces
        return output.split(" ")
    else:
        raise ValueError(f"Unsupported output type: {type(output)}")

class NERModel(BaseModel):
    """Named Entity Recognition Model"""
    def __init__(self, model_name: str = None):
        super().__init__(model_name)

    def extract_entities(self, prompt: str) -> dict:
        """Extracts entities"""
        # print(prompt)
        result = self.generate_response(prompt)
        response = result.get("content", "")
        thinking_content = result.get("thinking_content", "")
        
        response = response.strip().upper()

        print(f"Original output: {response}")
        print(f"thinking content: ", {thinking_content})

        if isinstance(response, str):
            try:
                response = ast.literal_eval(response)
            except (SyntaxError, ValueError):
                response = [tag.strip('\"') for tag in response.strip('[]').split(',')]

        response = normalize_output_to_list(response)
        response = extract_and_normalize_values(response)

        return response


class ClassificationModel(BaseModel):
    """Text Classification Model"""

    def classify(self, prompt: str, classes: List[str]) -> str:
        """Classifies text"""
        response = self.generate_response(prompt)
        
        # Simple parsing of classification result
        response_lower = response.lower()
        for cls in classes:
            if cls.lower() in response_lower:
                return cls
        
        return classes[0] if classes else "unknown"


class QAModel(BaseModel):
    """Question Answering and Reasoning Model"""
    def __init__(self, model_name: str = None):
        super().__init__(model_name)

    def extract_answer(self, prompt: str) -> str:
        """Performs question answering"""
        result = self.generate_response(prompt)
        response = result.get("content", "")
        thinking_content = result.get("thinking_content", "")
        print(f"Original output: {response}")
        
        response = response.lower()

        return response

class ModelFactory:
    """Model Factory"""

    @staticmethod
    def create_model(model_type: str, model_name: str = None):
        """Creates a model instance"""
        if model_type == "relation_extraction":
            return RelationExtractionModel(model_name)
        elif model_type == "ner":
            return NERModel(model_name)
        elif model_type == "classification":
            return ClassificationModel(model_name)
        elif model_type == "qa":
            return QAModel(model_name)
        elif model_type == "base":
            return BaseModel(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
