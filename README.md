# Fine-Tuning DeepSeek-R1-Distill-Llama-8B on Medical CoT Dataset

This repository contains a Jupyter notebook demonstrating the process of fine-tuning the `DeepSeek-R1-Distill-Llama-8B` model on a medical Chain-of-Thought (CoT) dataset using the `Unsloth` library for efficient and fast fine-tuning. The goal is to enhance the model's ability to perform clinical reasoning, diagnostics, and treatment planning.

---

## Project Overview

The notebook walks through the following steps:

1. **Setup**: Installing dependencies and preparing the environment.
2. **Model Loading**: Loading the pre-trained `DeepSeek-R1-Distill-Llama-8B` model and tokenizer.
3. **Pre-Fine-Tuning Inference**: Evaluating the model's performance on a medical question before fine-tuning.
4. **Dataset Preparation**: Loading and formatting the `medical-o1-reasoning-SFT` dataset.
5. **Fine-Tuning**: Training the model using LoRA (Low-Rank Adaptation) with `Unsloth`.
6. **Post-Fine-Tuning Inference**: Testing the fine-tuned model's performance on the same medical question.
7. **Tracking**: Logging training metrics with Weights & Biases (W&B).

The fine-tuned model demonstrates improved reasoning capabilities for medical queries, as shown by comparing pre- and post-fine-tuning responses.

---

## Prerequisites

To run this notebook, you'll need:

- **Python 3.9+**
- **GPU**: A CUDA-compatible GPU (e.g., T4 used in the notebook).
- **Dependencies**:
  - `unsloth`
  - `torch`
  - `transformers`
  - `datasets`
  - `wandb`
  - `trl`
  - `huggingface_hub`
- **API Keys**:
  - Hugging Face token (`HUGGINGFACE_TOKEN`) for model access.
  - Weights & Biases token (`WANDB_API_KEY`) for logging.

These can be set up in a Kaggle environment or locally with appropriate secrets management.
