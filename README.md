# Project Overview

This repository consists of a multi-stage pipeline for structured multi-task learning. The workflow is organized into three main steps, each encapsulated in its own folder.

---

## 🔧 Step 1: Data Processing

**Folder**: `data_process`

This step handles data preprocessing, formatting, and organization. It includes:
- Loading raw input data, transforming data into SFT formats
- sample
- Transforming data into RL formats

---

## 🚀 Step 2: Two-Stage Optimization

**Folder**: `two_stage_optimization`

Implements a two-stage optimization strategy:
- **Stage 1**: Task-specific supervised fine-tuning
- **Stage 2**: RL 

This module aims to improve performance across multiple structured tasks using a tailored training regime.

---

## 🧪 Step 3: Testing with Structured Multi-task Setup

**Folder**: `structure_multi_task`

Evaluates the model's performance on multi-task benchmarks. It includes:
- Model loading and inference
- Structured evaluation metrics


