
#!/usr/bin/env python3

import json
import re
import difflib
from typing import List
from collections import Counter
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorWithPadding,
    GenerationConfig
)
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
os.environ["WANDB_DISABLED"] = "true"

@dataclass
class Config:
    model_name: str = "****Model path or Model name****"
    max_length: int = 1024
    max_prompt_length: int = 512
    batch_size: int = 2 
    gradient_accumulation_steps: int = 32  
    learning_rate: float = 5e-6 
    num_epochs: int = 1  
    output_dir: str = "./grpo_qwen3_outputs"
    beta: float = 0.08  
    epsilon_high = 0.28  
    # epsilon_low = 0.1 
    temperature: float = 0.9  
    top_p: float = 0.95  
    max_new_tokens: int = 512 
    enable_thinking: bool = True  

def calculate_ner_f1_improved(pred, label):
    if not pred or not label:
        return 0.0
    
    if len(pred) == len(label):
        correct = sum(1 for p, g in zip(pred, label) if p == g)
        precision = correct / len(pred) if len(pred) > 0 else 0.0
        recall = correct / len(label) if len(label) > 0 else 0.0
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        else:
            return 0.0
    
    else:
        min_len = min(len(pred), len(label))
        max_len = max(len(pred), len(label))

        pred_aligned = pred[:min_len] + ['O'] * (max_len - len(pred)) if len(pred) < max_len else pred[:max_len]
        label_aligned = label[:min_len] + ['O'] * (max_len - len(label)) if len(label) < max_len else label[:max_len]
        
        correct = sum(1 for p, g in zip(pred_aligned, label_aligned) if p == g)

        length_penalty = min_len / max_len
        
        precision = correct / len(pred_aligned) if len(pred_aligned) > 0 else 0.0
        recall = correct / len(label_aligned) if len(label_aligned) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            return f1 * length_penalty  
        else:
            return 0.0

def calculate_semantic_similarity(pred_text, label_text):
    similarity = difflib.SequenceMatcher(None, pred_text.lower(), label_text.lower()).ratio()
    return similarity

def reward_ner_improved(pred, label):
    try:
        print("Original pred:", pred)
        print("parsed_label", label)

        predicted_labels = pred
      
        f1_score = calculate_ner_f1_improved(predicted_labels, label)

        if len(label) > 0:
            length_similarity = 1.0 - abs(len(predicted_labels) - len(label)) / max(len(predicted_labels), len(label), 1)
            length_bonus = length_similarity * 0.1
        else:
            length_bonus = 0.0

        unique_pred_labels = set(predicted_labels) - {'O'}
        unique_true_labels = set(label) - {'O'}
        if unique_true_labels:
            diversity_score = len(unique_pred_labels & unique_true_labels) / len(unique_true_labels)
            diversity_bonus = diversity_score * 0.1
        else:
            diversity_bonus = 0.0
        
        total_reward = f1_score + length_bonus + diversity_bonus
        return min(total_reward, 1.0) 
        
    except Exception as e:
        print(f"[NER Error] {e}")
        return 0.0

def reward_generative_classification_improved(pred: str, label: str, task: str) -> float:
    print("task:", task)
    print("pred:", pred)
    print("label:", label)
    
    try:
        pred_clean = str(pred).strip().lower()
        label_clean = str(label).strip().lower()

        if pred_clean == label_clean:
            return 1.0
          
        base_similarity = calculate_semantic_similarity(pred_clean, label_clean)

        pattern = r'\b' + re.escape(label_clean) + r'\b'
        if re.search(pattern, pred_clean):
            regex_bonus = 0.3
        else:
            regex_bonus = 0.0

        if len(label_clean) > 0:
            length_similarity = 1.0 - abs(len(pred_clean) - len(label_clean)) / max(len(pred_clean), len(label_clean))
            length_bonus = length_similarity * 0.1
        else:
            length_bonus = 0.0

        label_words = set(label_clean.split())
        pred_words = set(pred_clean.split())
        if label_words:
            keyword_overlap = len(label_words & pred_words) / len(label_words)
            keyword_bonus = keyword_overlap * 0.2
        else:
            keyword_bonus = 0.0

        total_reward = (
            base_similarity * 0.4 +
            regex_bonus +
            length_bonus +
            keyword_bonus
        )
        
        return min(total_reward, 1.0)  
        
    except Exception as e:
        print(f"[Classification Reward Error] {e}")
        return 0.0

def add_diversity_penalty(rewards, completions):
    if len(completions) <= 1:
        return rewards

    diversity_scores = []
    for i, completion in enumerate(completions):
        similarities = []
        for j, other_completion in enumerate(completions):
            if i != j:
                sim = calculate_semantic_similarity(str(completion), str(other_completion))
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0
        diversity_score = 1.0 - avg_similarity
        diversity_scores.append(diversity_score)

    adjusted_rewards = []
    for reward, diversity_score in zip(rewards, diversity_scores):
        diversity_bonus = diversity_score * 0.05 
        adjusted_reward = reward + diversity_bonus
        adjusted_rewards.append(min(adjusted_reward, 1.0))
    
    return adjusted_rewards

def unified_reward_improved(completions, **kwargs):
    rewards = []
    outputs = kwargs.get('output', [])
    tasks = kwargs.get('task', [])
    
    for completion, output, task in zip(completions, outputs, tasks):
        reward = 0.0
        content = ""
        try:
            if isinstance(completion, str):
                if '</think>' in completion:
                    think_end = completion.rfind('</think>')
                    content = completion[think_end + len('</think>'):].strip()
                else:
                    content = completion.strip()
            else:
                content = str(completion).strip()

            if task == "NER":
                reward = reward_ner_improved(content, output)
            elif task in ["QA", "RE"]:
                reward = reward_generative_classification_improved(content, output, task)
            else:
                print(f"[Reward] Unknown task type: {task}")
                reward = 0.0

        except Exception as e:
            print(f"[Unified Reward Error] Task={task}, Error={e}")
            reward = 0.0

        print(f"Reward={reward}")
        rewards.append(reward)

    if len(completions) > 1:
        rewards = add_diversity_penalty(rewards, completions)
        print("Applied diversity bonus")
    
    return rewards

def setup_model_and_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    generation_config = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=True,  
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        repetition_penalty=1.2, 
    )
    
    model.generation_config = generation_config
    return model, tokenizer

def preprocess_dataset_with_thinking(dataset: Dataset, tokenizer, config: Config) -> Dataset:
    
    def format_with_thinking(example):
        messages = [{"role": "user", "content": example['prompt']}]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False,
            enable_thinking=config.enable_thinking
        )
        
        return {
            'prompt': formatted_prompt,
            'output': example['output'],
            'task': example['task']
        }
    
    processed_dataset = dataset.map(format_with_thinking)
    return processed_dataset

def validate_dataset(dataset: Dataset) -> Dataset:
    valid_samples = []
    for sample in dataset:
        if all(key in sample for key in ['prompt', 'output', 'task']):
            if sample['prompt'] and len(sample['prompt']) > 10:
                valid_samples.append(sample)
    
    logger.info(f"Filtered dataset from {len(dataset)} to {len(valid_samples)} valid samples")
    return Dataset.from_list(valid_samples)

def load_json_dataset(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if len(data) > 1000:
        logger.info(f"Sampling dataset from {len(data)} to 1000 samples")
        import random
        random.seed(42)
        data = random.sample(data, 1000)
    
    dataset = Dataset.from_list(data)
    return dataset

def main():
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    config = Config()
    
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)

    print("Loading dataset...")
    json_file_path = "multitask_bio_dataset_sample_prompt.json"
    dataset = load_json_dataset(json_file_path)
    
    print("Preprocessing dataset...")
    dataset = preprocess_dataset_with_thinking(dataset, tokenizer, config)
    dataset = validate_dataset(dataset)
    print(f"Final dataset size: {len(dataset)}")
    
    print("Sample processed prompt:")
    print(dataset[0]['prompt'][:500] + "...")

    grpo_config = GRPOConfig(
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        max_prompt_length=config.max_prompt_length,
        beta=config.beta,
        output_dir=config.output_dir,
        logging_steps=5,
        save_steps=100,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        fp16=False,
        bf16=True,
        warmup_steps=20,
        weight_decay=0.001,
        max_grad_norm=1.0,
        report_to=None,
        epsilon_high=config.epsilon_high,  
        temperature=config.temperature,
        mask_truncated_completions=True,
        loss_type="dr_grpo",
        top_p=config.top_p,
    )
    
    print("Initializing GRPO trainer...")
    try:
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=[unified_reward_improved],
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Saving model...")
        trainer.save_model(config.output_dir + "/final_model")
        tokenizer.save_pretrained(config.output_dir + "/final_model")
        
        print("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        print(f"Error details: {e}")
        raise e

if __name__ == "__main__":
    main()
