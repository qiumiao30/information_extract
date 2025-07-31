# models.py
"""
模型管理模块 - 统一管理所有模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from config import MODEL_CONFIG
import ast
import re
from multi_agent import DCRMADEFramework


class BaseModel:
    """基础模型类"""
    
    def __init__(self, model_name: str = None):
        """初始化模型"""
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.tokenizer = None
        self.model = None
        self.device = None
        self.conversation_history = []  # 只添加这一行
        
    def load_model(self):
        """加载模型"""
        print(f"正在加载模型: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
            device_map=MODEL_CONFIG["device_map"],
        )
        
        self.device = self.model.device
        print(f"模型加载完成，设备: {self.device}")
    
    def generate_response(self, prompt: str, **kwargs) -> dict:
        """
        生成回复，支持Qwen3的思考模式
        
        Args:
            prompt: 输入提示
            enable_thinking: 是否启用思考模式，默认为True
            **kwargs: 其他生成参数
        
        Returns:
            dict: 包含thinking_content和content的字典
        """
        if self.model is None:
            self.load_model()
        
        generation_config = {**MODEL_CONFIG["generation_config"], **kwargs}
        
        # 构建对话格式
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False,
            enable_thinking=True  # 启用思考模式
        )
        model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)
        
        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            pad_token_id=self.tokenizer.eos_token_id  # 明确设置，消除提示
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            THINK_END_ID = self.tokenizer.convert_tokens_to_ids('</think>')
            index = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print(f"思考内容: {thinking_content}")
        # print(f"生成内容: {content}")
                
        return {
            "thinking_content": thinking_content,
            "content": content
        }

class RelationExtractionModel(BaseModel):
    """关系抽取模型"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name)
        self.valid_labels = ['CPR:1', 'CPR:2', 'CPR:3', 'CPR:4', 'CPR:5', 
                           'CPR:6', 'CPR:7', 'CPR:9', 'CPR:10', "0", "1", "0", "DDI-effect", "DDI-mechanism", "DDI-advise"]
    
    def extract_relation(self, prompt: str) -> str:
        """抽取关系"""
        result = self.generate_response(prompt)
        response = result.get("content", "")
        thinking_content = result.get("thinking_content", "")
        print(f"原始模型输出: {response}")
        response = response.strip().upper()

        print(f"think model: ", thinking_content)
        
        print(f"模型输出: {response}")
        
        # # 解析输出，提取关系标签
        # for label in self.valid_labels:
        #     if label in response:
        #         return label
        
        # return 'false'  # 默认无关系
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
    将模型输出统一转换为 BIO 标签的 list 格式。
    支持原始字符串（空格、换行、混合分隔），也支持已有 list。
    """
    if isinstance(output, list):
        return output
    elif isinstance(output, str):
        # 替换多种分隔符为统一空格，再 split
        output = output.replace("\n", " ").replace(",", " ").strip()
        output = " ".join(output.split())  # 去除多余空格
        return output.split(" ")
    else:
        raise ValueError(f"Unsupported output type: {type(output)}")

class NERModel(BaseModel):
    """命名实体识别模型"""
    def __init__(self, model_name: str = None):
        super().__init__(model_name)
    
    def extract_entities(self, prompt: str) -> dict:
        """抽取实体"""
        # print(prompt)
        result = self.generate_response(prompt)
        response = result.get("content", "")
        thinking_content = result.get("thinking_content", "")
        
        response = response.strip().upper()

        print(f"原始输出: {response}")
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
    """文本分类模型"""
    
    def classify(self, prompt: str, classes: List[str]) -> str:
        """分类文本"""
        response = self.generate_response(prompt)
        
        # 简单的分类结果解析
        response_lower = response.lower()
        for cls in classes:
            if cls.lower() in response_lower:
                return cls
        
        return classes[0] if classes else "unknown"


class QAModel(BaseModel):
    """问答推理模型"""
    def __init__(self, model_name: str = None):
        super().__init__(model_name)
    
    def extract_answer(self, prompt: str) -> str:
        """问答"""
        result = self.generate_response(prompt)
        response = result.get("content", "")
        thinking_content = result.get("thinking_content", "")
        print(f"原始输出: {response}")
        
        response = response.lower()

        return response

class ModelFactory:
    """模型工厂"""
    
    @staticmethod
    def create_model(model_type: str, model_name: str = None):
        """创建模型实例"""
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
            raise ValueError(f"不支持的模型类型: {model_type}")
