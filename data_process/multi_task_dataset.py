import json
import random
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import re

# 用于从BIO标签解码实体
from sklearn.preprocessing import MultiLabelBinarizer

# ----------------------------------------------------------------------------
# 1. 配置要处理的数据集
# ----------------------------------------------------------------------------
# 将数据集名称、类型和要使用的子集（如果需要）进行配置
DATASET_CONFIG = {
    # --- 命名实体识别 (NER) ---
    "bc2gm": {  ### id; tokens-->list; ner_tags-->list[0,0,1,2,0]
        "name": "spyysalo/bc2gm_corpus",
        "type": "ner",
        "split": "train" # 通常我们只处理训练集
    },
    "jnlpba": { ### id; rokens-->list; ner_tags-->list
        "name": "jnlpba/jnlpba",
        "type": "ner",
        "split": "train"
    },
    # --- 关系抽取 (RE) ---
    "chemprot": {
        "name": "bigbio/chemprot",
        "type": "re",
    },
    "ddi": {
        "name": "bigbio/ddi_corpus",
        "type": "re",
    },
    "gad": {
        "name": "bigbio/gad",
        "type": "re",
        "split": "train"
    },
    # --- 问答 (QA) ---
    "pubmedqa": {
        "name": "qiaojin/PubMedQA",
        "type": "qa",
        "subset": "pqa_labeled", # 使用有标签的子集
        "split": "train"
    },
    "bioasq": {
        "name": "kroshan/BioASQ",
        "type": "qa"
    },
    "NCBI-disease":{
        "name": "NCBI-disease",
        "type": "ner"
    },
    "medqa": {
        "name": "GBaker/MedQA-USMLE-4-options",
        "type": "qa",
        "split": "train"
    },
}

# 统一的输出文件
OUTPUT_FILE = "multitask_bio_dataset_all_8_newnew.json"

# ----------------------------------------------------------------------------
# 2. 为每个任务类型定义处理器函数
# ----------------------------------------------------------------------------

def process_ner_dataset(dataset, config):
    """处理命名实体识别数据集"""
    print(f"Processing NER dataset: {config['name']}")
    processed_samples = []
    
    # tag_names = dataset.features['ner_tags'].feature.names
    if config['name'] == "spyysalo/bc2gm_corpus" or config['name'] == "jnlpba/jnlpba":
        for example in tqdm(dataset, desc=f"  -> {config['name']}"):
            tokens = example['tokens']
            tags = example['ner_tags']
            
            if not tokens:
                continue

            if config['name'] == "spyysalo/bc2gm_corpus":
                id2label = {
                    0: "O",
                    1: "B-DISEASE",
                    2: "I-DISEASE"
                }
                bio_tags = [id2label[tag] for tag in tags]
            elif config['name'] == "jnlpba/jnlpba":
                id2label = {
                    0: "O",
                    1: "B-DNA",
                    2: "I-DNA",
                    3: "B-RNA",
                    4: "I-RNA",
                    5: "B-cell_line",
                    6: "I-cell_line",
                    7: "B-cell_type",
                    8: "I-cell_type",
                    9: "B-protein",
                    10: "I-protein",
                }
                # 将 tags 转换为 BIO 标注
                bio_tags = [id2label[tag] for tag in tags]
            else:
                None

            # 构建统一格式
            Instruction = f"[Task: NER][Domain: {config['name']}] You are a biomedical named entity recognition (NER) expert. " \
                "Please extract and label all predefined biomedical entities using the BIO format. " \
                f"Supported entity labels list (BIO format): {list(id2label.values())}. " 
                
            input_text = " ".join(tokens)
            input_text += f" Tokens: {tokens}"

            # prompt += input_text
            # bio_tags = ", ".join(bio_tags)

            processed_samples.append({
                "Instruction": Instruction,
                "input": input_text,
                "output": str(bio_tags),
                "task": "NER"
            })
    elif config['name'] == "NCBI-disease":
        data = pd.read_json("/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/tokcls/NCBI-disease_hf/train.json", lines=True)

        id2label = {
                    0: "O",
                    1: "B",
                    2: "I"
                }
                
        for _, item in data.iterrows():
            # TODO: 根据实际数据格式实现处理逻辑
            Instruction = f"[Task: NER][Domain: {config['name']}].You are a biomedical named entity recognition (NER) expert. " \
                        "Please extract and label all predefined biomedical entities using the BIO format. " \
                        f"Supported entity labels list (BIO format): {list(id2label.values())}. "
            
            
            # 将上下文列表合并
            tokens = item['tokens']
            input_text = " ".join(tokens)
            input_text += f" Tokens: {tokens}"
            
            answer = str(item['ner_tags'])

            processed_samples.append({
                    "Instruction": Instruction,
                    "input": input_text,
                    "output": answer,
                    "task": "NER"
                })
        
    return processed_samples

def process_re_dataset(dataset, config):
    """处理关系抽取数据集"""
    print(f"Processing RE dataset: {config['name']}")
    processed_samples = []

    if config['name'] == "bigbio/chemprot":
        data = pd.read_csv("/home/siat/Data/synthetic/MinerU/RE/ChemProt_Corpus/chemprot_training/chemprot_merged_training_output.tsv", sep='\t', encoding='utf-8')
        
        for _, item in data.iterrows():
            # TODO: 根据实际数据格式实现处理逻辑
            Instruction = f"[Task: RE][Domain: {config['name']}]. You are an expert in biomedical relation extraction. " \
                "Given a text and two entities, identify the relationship between the entities based on the predefined relation types." \
                "Relation types:" \
                "- CPR:3: UPREGULATOR" \
                "- CPR:4: DOWNREGULATOR" \
                "- CPR:5: AGONIST" \
                "- CPR:6: ANTAGONIST" \
                "- CPR:9: SUBSTRATE" \
                f"- CPR:10: NO. What is the relationship between {item.get('entity1', '')} and {item.get('entity2', '')} in the following text?" \
                "Respond with: 'CPR:3', 'CPR:4','CPR:5','CPR:6''CPR:9'."
            
            input_text = item.get('text', '')
            output_text = f"{item.get('relation', 'false')}"

            # prompt += input_text

            processed_samples.append({
            "Instruction": Instruction,
            "input": input_text,
            "output": output_text,
            "task": "RE"
        })
        
    elif config['name'] == 'bigbio/ddi_corpus':
        # 读取json文件，/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/seqcls/DDI_hf/train.json

        print("Processing DDI dataset...")

        data = pd.read_json("/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/seqcls/DDI_hf/train.json", lines=True)
                
        for _, item in data.iterrows():
            # TODO: 根据实际数据格式实现处理逻辑
            Instruction = f"[Task: RE][Domain: {config['name']}]. You are an expert in biomedical relation extraction." \
                                "Given a sentence containing two marked entities (@DRUG$ and @DRUG$), determine whether there is a meaningful biomedical relationship between them." \
                                "Respond with: '0', 'DDI-mechanism', 'DDI-effect', 'DDI-advise', 'DDI-int'." 
            
            input_text = item.get('sentence', '')
            output_text = f"{item.get('label', 'false')}"

            processed_samples.append({
                    "Instruction": Instruction,
                    "input": input_text,
                    "output": output_text,
                    "task": "RE"
                })
            
    # elif config['name'] == 'bigbio/gad':
    #     for example in tqdm(dataset, desc=f"  -> {config['name']}"):
    #         sentence = example['sentence']
    #         label = example['label']

    #         Instruction = f"[Task: RE][Domain: {config['name']}]. You are an expert in biomedical relation extraction." \
    #                     "Given a sentence containing two marked entities (@GENE$ and @DISEASE$), determine whether there is a meaningful biomedical relationship between them." \
    #                     "Respond with either 1 or 0."
            
    #         input_text = sentence
    #         output_text = label

    #         # prompt += input_text

    #         processed_samples.append({
    #             "Instruction": Instruction,
    #             "input": input_text,
    #             "output": str(output_text),
    #             "task": "RE"
    #         })
    else:
        None

    return processed_samples

def process_qa_dataset(dataset, config):
    """处理问答数据集"""
    print(f"Processing QA dataset: {config['name']}")
    processed_samples = []
    
    # ---- 针对不同QA数据集的字段适配 ----
    if config['name'] == 'qiaojin/PubMedQA':
        # 只取前 num_samples 条数据
        # 将数据集转换为列表并切片
        dataset_list = list(dataset)  # 将 Dataset 转换为列表
        dataset = dataset_list[:500]  # 获取前 num_samples 条数据

        for example in tqdm(dataset, desc=f"  -> {config['name']}"):
            Instruction = f"[Task: QA][Domain: {config['name']}]. You are an expert in biomedical question answering. Based on the following context and question, response with: 'yes', 'no', 'maybe'."
            question = ""
            context = ""
            answer = ""

            # 获取问题和答案
            question = example['question']
            answer = example['final_decision']

            # 获取上下文并合并
            context = example['context'].get('contexts', [])
            if isinstance(context, list):
                context = " ".join(context)  # 合并多个上下文段落为一个字符串
            else:
                context = str(context)  # 如果上下文不是列表，直接转为字符串

            if not question or not answer:
                continue  # 跳过没有问题或答案的样本

            # 构建输入文本
            input_text = f"Question: {question}\n\nContext: {context}"

            # 添加处理后的样本
            processed_samples.append({
                "Instruction": Instruction,
                "input": input_text,
                "output": answer,
                "task": 'QA'
            })
        
        # 返回处理后的样本
        return processed_samples

    # ---- 针对不同QA数据集的字段适配 ----
    # if config['name'] == 'qiaojin/PubMedQA':
    #     # 只取前 num_samples 条数据
    #     # 将数据集转换为列表并切片
    #     data = pd.read_json("/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/seqcls/bioasq_hf/train.json", lines=True)
                
    #     for _, item in data.iterrows():
    #         # 打印第一行数据
    #         Instruction = f"[Task: QA][Domain: {config['name']}]. You are an expert in biomedical question answering. Based on the following context and question, response with: 'yes', 'no' or 'maybe'." 
    #         question = item['sentence1']
    #         # 将上下文列表合并
    #         context = item['sentence2']
    #         answer = item['label']

    #         # 构建输入文本
    #         input_text = f"Question: {question}\n\nContext: {context}"

    #         # 添加处理后的样本
    #         processed_samples.append({
    #             "Instruction": Instruction,
    #             "input": input_text,
    #             "output": answer,
    #             "task": 'QA'
    #         })
        
    #     # 返回处理后的样本
    #     return processed_samples

    elif config['name'] == 'GBaker/MedQA-USMLE-4-options':
        for example in tqdm(dataset, desc=f"  -> {config['name']}"):
            question = example['question']
            options = example['options']
            answer_idx = example['answer_idx']


            Instruction = f"[Task: QA][Domain: {config['name']}]. You are an expert in biomedical question answering." \
                            "Given a question and four multiple choice options (A, B, C, D), select the most appropriate answer based on your biomedical knowledge." \
                            "Respond with only the letter (A, B, C, or D) corresponding to your chosen answer."
            
            input_text = f"Question: {question}\nOpttions: {options}"
            output_text = answer_idx

            # prompt += input_text

            processed_samples.append({
                "Instruction": Instruction,
                "input": input_text,
                "output": str(output_text),
                "task": "QA"
            })

    # elif config['name'] == 'kroshan/BioASQ':

    #     print("processing BioASQ...")

    #     data = pd.read_json("/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/data/seqcls/bioasq_hf/train.json", lines=True)
                
    #     for _, item in data.iterrows():
    #         # 打印第一行数据
    #         Instruction = f"[Task: QA][Domain: {config['name']}]. You are an expert in biomedical question answering. Based on the following context and question, response with: 'yes' or 'no'." 
    #         question = item['sentence1']
    #         # 将上下文列表合并
    #         context = item['sentence2']
    #         answer = item['label']

    #         # # 确保所有字段都不为None
    #         # if question is None or context is None or answer is None:
    #         #     print(f"Skipping row : None values found")
    #         #     continue

    #         input_text = f"Question: {question}\n\n Context: {context}"

    #         processed_samples.append({
    #                 "Instruction": Instruction,
    #                 "input": input_text,
    #                 "output": answer,
    #                 "task": "QA"
    #             })
        
        return processed_samples


# ----------------------------------------------------------------------------
# 3. 主执行逻辑
# ----------------------------------------------------------------------------
def main():
    """主函数，协调所有处理流程"""
    all_processed_data = []

    # 映射任务类型到处理函数
    PROCESSOR_MAP = {
        "ner": process_ner_dataset,
        "re": process_re_dataset,
        "qa": process_qa_dataset,
    }

    print("开始进行多任务数据预处理...")

    for task_key, config in DATASET_CONFIG.items():
        # 加载数据集
        print(f"\nLoading dataset '{config['name']}'...")
        try:
            # 有些数据集需要指定子集名称
            if "subset" in config:
                dataset = load_dataset(config['name'], config['subset'], split=config['split'])
            elif "split" in config:
                dataset = load_dataset(config['name'], split=config['split'])
            else:
                dataset = None
            
            # 根据任务类型调用相应的处理器
            processor_func = PROCESSOR_MAP.get(config['type'])
            if processor_func:
                processed_data = processor_func(dataset, config)
                all_processed_data.extend(processed_data)
                print(f"  -> {config['name']} 处理完成，获得 {len(processed_data)} 条样本。")
            else:
                print(f"  -> 警告：未找到任务类型 '{config['type']}' 的处理器。")
        except Exception as e:
            print(f"  -> 处理数据集 '{config['name']}' 时发生错误: {e}")

    print(f"\n所有数据集处理完毕，总共获得 {len(all_processed_data)} 条样本。")

    # 打乱所有数据
    print("正在打乱数据...")
    random.shuffle(all_processed_data)

    # # 保存到JSONL文件
    # print(f"正在将数据写入到 '{OUTPUT_FILE}'...")
    # with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    #     for sample in tqdm(all_processed_data, desc="Writing to file"):
    #         f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    import json

    print(f"正在将数据写入到 '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, ensure_ascii=False, indent=2)

            
    print("\n处理完成！现在你可以使用 'multitask_bio_dataset.jsonl' 文件进行SFT训练了。")


if __name__ == "__main__":
    main()