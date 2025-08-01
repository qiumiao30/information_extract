# Multi-Task Dataset Processing

This repository contains scripts for processing multi-task datasets for reinforcement learning applications. The workflow consists of three main steps: dataset download and processing, sampling, and instruction formatting.

## Quick Start

Follow these three steps in order to prepare your dataset:

### Step 1: Download and Process Dataset
```bash
python multi_task_dataset.py
```
This script downloads the raw dataset and performs unified processing to standardize the data format across different tasks.

### Step 2: Sample Dataset
```bash
python sample_dataset.py
```
Samples 1000 examples from the processed dataset to create a manageable subset for training.

### Step 3: Generate RL Instructions
```bash
python instruct_input_prompt.py
```
Converts the sampled dataset into instruction-formatted prompts suitable for reinforcement learning training.

## Prerequisites

Make sure you have Python installed with the required dependencies. You may need to install additional packages depending on your specific dataset requirements.

## Output

After running all three steps, you will have:
- Downloaded and processed multi-task dataset
- A sampled subset of 1000 examples
- Instruction-formatted dataset ready for RL training

## Usage Notes

- Run the scripts in the specified order as each step depends on the output of the previous one
- The sampling size (1000 examples) can likely be modified in `sample_dataset.py` if needed
