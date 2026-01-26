# The Code Implemantation of the OptiScene in NeurIPS 2025.

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/117323)
[![arXiv](https://img.shields.io/badge/arXiv-2506.07570-b31b1b.svg)](https://arxiv.org/abs/2506.07570)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://polysummit.github.io/optiscene.github.io/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Models-yellow.svg)]()

> **OptiScene: LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization**
>
> *Accepted at NeurIPS 2025*


## Introduction

This is the official code repository for our NeurIPS 2025 paper **OptiScene**. We present a novel approach for indoor scene layout generation using Large Language Models (LLMs) through scaled human-aligned data synthesis and multi-stage preference optimization (DPO).


## Environment Setup
To set up the required environment, follow these steps:

```bash
conda env create -f environment.yml
conda activate optiscene
```

## Training Pipeline

The training process consists of three main stages: SFT, DPO Stage 1, and DPO Stage 2, with a LoRA merge step after each training stage. All tasks are run through the `main.py` entry point.

### 1. Supervised Fine-Tuning (SFT)

This is the first stage of training, where the model is fine-tuned on a specific dataset.

```bash
# SFT Training
python main.py --task sft \
    --dataset_file dataset/sft_prompts.json \
    --model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    --bf16 \
    --checkpoint_dir=outputs/Qwen-7B-SFT \
    --per_device_train_batch_size=8 \
    --save_strategy=epoch \
    --epochs=1
```

After SFT training is complete, merge the LoRA adapter with the base model:

```bash
# Merge SFT LoRA
python main.py --task merge \
    --base_model_path Qwen/Qwen2.5-7B-Instruct \
    --lora_path outputs/Qwen-7B-SFT/checkpoint-XXXX \
    --output_path outputs/Qwen-7B-SFT-merged
```

*Note: Replace `outputs/Qwen-7B-SFT/checkpoint-XXXX` with the actual path to your SFT LoRA checkpoint.*

### 2. DPO Stage 1

In the first DPO stage, the merged SFT model is further trained using preference data.

```bash
# DPO Stage 1 Training
python main.py --task dpo \
    --dataset_file dataset/dpo1_prompts.json \
    --model_name_or_path outputs/Qwen-7B-SFT-merged \
    --learning_rate 5.0e-6 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 3200 \
    --max_prompt_length 3200 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --save_steps 100 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --output_dir outputs/Qwen-7B-DPO-Stage1 \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --bf16
```

After DPO Stage 1 is complete, merge the LoRA adapter:

```bash
# Merge DPO Stage 1 LoRA
python main.py --task merge \
    --base_model_path outputs/Qwen-7B-SFT-merged \
    --lora_path outputs/Qwen-7B-DPO-Stage1/checkpoint-XXXX \
    --output_path outputs/Qwen-7B-DPO-Stage1-merged
```

*Note: Replace `outputs/Qwen-7B-DPO-Stage1/checkpoint-XXXX` with the actual path to your DPO Stage 1 LoRA checkpoint.*

### 3. DPO Stage 2

The second DPO stage continues the training from the merged DPO Stage 1 model.

```bash
# DPO Stage 2 Training
python main.py --task dpo \
    --dataset_file dataset/dpo2_prompts.json \
    --model_name_or_path outputs/Qwen-7B-DPO-Stage1-merged \
    --learning_rate 5.0e-6 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 3200 \
    --max_prompt_length 3200 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --save_steps 100 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --output_dir outputs/Qwen-7B-DPO-Stage2 \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --bf16
```

After DPO Stage 2 is complete, merge the LoRA adapter:

```bash
# Merge DPO Stage 2 LoRA
python main.py --task merge \
    --base_model_path outputs/Qwen-7B-DPO-Stage1-merged \
    --lora_path outputs/Qwen-7B-DPO-Stage2/checkpoint-XXXX \
    --output_path outputs/Qwen-7B-DPO-Stage2-merged
```

*Note: Replace `outputs/Qwen-7B-DPO-Stage2/checkpoint-XXXX` with the actual path to your DPO Stage 2 LoRA checkpoint.*

## Inference

Once the final model is trained and merged, you can run inference using the following command.

```bash
python main.py --task inference --checkpoint_dir outputs/Qwen-7B-DPO-Stage2-merged
```

## Model Weights

We will release the final pre-trained model weights on Hugging Face soon:

| Model | Description | Link |
|-------|-------------|------|
| OptiScene-Qwen-7.5B | Final model after Two-Stage DPO | Coming Soon |

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{yang2025optiscene,
  title={Optiscene: Llm-driven indoor scene layout generation via scaled human-aligned data synthesis and multi-stage preference optimization},
  author={Yang, Yixuan and Luo, Zhen and Ding, Tongsheng and Lu, Junru and Gao, Mingqi and Yang, Jinyu and Sanchez, Victor and Zheng, Feng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

We thank the open-source community for their valuable contributions that made this work possible.
