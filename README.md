# Task-Adaptive Refined Reinforcement Learning with Granular Reward Shaping for Biomedical Information Extraction

## Abstract

The explosive growth of biomedical literature and multi-omics data has created an urgent need for automated knowledge extraction, where large language models (LLMs) have shown great promise in natural language understanding tasks. However, existing LLMs still struggle with structured tasks, as they are primarily optimized for generating free-form text rather than adhering to schema-constrained outputs. Prior efforts to align LLMs using supervised fine-tuning and reinforcement learning methods often fail to generalize in multi-task biomedical settings due to semantic ambiguity in task definitions and unstable policy optimization under sparse rewards and inter-task interference. 

To address these challenges, we introduce GRASP (Group-Relative Adaptive Structured Prompting), a unified framework for biomedical information extraction. GRASP combines a hierarchical task-aware prompt design that explicitly encodes task semantics with a novel Group-Relative Policy Optimization strategy, enabling fine-grained, semantically sensitive reward modeling. This approach resolves task interference and enhances the fidelity of structured outputs across diverse extraction tasks. Extensive experiments across biomedical and general benchmarks demonstrate that GRASP achieves state-of-the-art performance in structural accuracy and semantic consistency, with up to 11% and 7% relative improvements in Micro F1 for NER and relation extraction, respectively, over competitive baselines.

## Model Architecture

![GRASP Framework]([/assets/model.png](/assets/model.png))

*Figure 1: Overview of the GRASP framework showing the hierarchical task-aware prompt design and Group-Relative Policy Optimization strategy.*

## 🔧 Step 1: Data Processing
**Folder**: `data_process`

This step handles data preprocessing, formatting, and organization. It includes:
- Loading raw input data, transforming data into SFT formats
- Sample dataset generation
- Transforming data into RL formats

### Quick Start for Data Processing:
```bash
python multi_task_dataset.py  # Download dataset and Unified Processing
python sample_dataset.py      # Sample 1000 examples
python instruct_input_prompt.py  # RL Dataset Instruction
```

## 🚀 Step 2: Two-Stage Optimization
**Folder**: `two_stage_optimization`

Implements a two-stage optimization strategy:
- **Stage 1**: Task-specific supervised fine-tuning
- **Stage 2**: Reinforcement Learning with Group-Relative Policy Optimization

This module aims to improve performance across multiple structured tasks using a tailored training regime.

## 🧪 Step 3: Testing with Structured Multi-task Setup
**Folder**: `structure_multi_task`

Evaluates the model's performance on multi-task benchmarks. It includes:
- Model loading and inference
- Structured evaluation metrics
- Performance analysis across different biomedical tasks

## Prerequisites

Make sure you have Python installed with the required dependencies. You may need to install additional packages depending on your specific dataset requirements.

## Usage Notes

- Run the scripts in the specified order as each step depends on the output of the previous one
- Ensure you have sufficient disk space for the dataset download and processing
- The sampling size (1000 examples) can be modified in `sample_dataset.py` if needed

## Support

If you encounter any issues with the processing pipeline, please check that all dependencies are properly installed and that you have the necessary permissions for file operations.
