import argparse
from scripts.sft_train import train as sft_train
from scripts.dpo_train import main as dpo_train
from scripts.inference import infer
from scripts.merge_lora import apply_lora
from trl import TrlParser
from trl import (
    DPOConfig,
    ModelConfig,
    ScriptArguments,
)

def main():
    parser = argparse.ArgumentParser(description="Main script for SFT, DPO, merge, and inference.")
    parser.add_argument("--task", type=str, required=True, choices=["sft", "dpo", "merge", "inference"], help="Task to perform.")
    
    args, remaining_args = parser.parse_known_args()

    if args.task == "sft":
        sft_parser = argparse.ArgumentParser()
        sft_parser.add_argument("--dataset_file", type=str, required=True)
        sft_parser.add_argument("--model_name_or_path", type=str, required=True)
        sft_parser.add_argument("--checkpoint_dir", type=str, required=True)
        sft_parser.add_argument("--learning_rate", type=float, default=5e-5)
        sft_parser.add_argument("--adam_beta1", type=float, default=0.9)
        sft_parser.add_argument("--adam_beta2", type=float, default=0.999)
        sft_parser.add_argument("--weight_decay", type=float, default=0.01)
        sft_parser.add_argument("--warmup_ratio", type=float, default=0.1)
        sft_parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
        sft_parser.add_argument("--logging_steps", type=int, default=10)
        sft_parser.add_argument("--bf16", action="store_true")
        sft_parser.add_argument("--per_device_train_batch_size", type=int, default=8)
        sft_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        sft_parser.add_argument("--max_seq_length", type=int, default=2048)
        sft_parser.add_argument("--epochs", type=int, default=3)
        sft_parser.add_argument("--save_steps", type=int, default=100)
        sft_parser.add_argument("--save_strategy", type=str, default="steps")
        sft_parser.add_argument("--max_grad_norm", type=float, default=1.0)
        sft_parser.add_argument("--cache_dir", type=str, default=None)
        sft_args = sft_parser.parse_args(remaining_args)
        sft_train(sft_args)

    elif args.task == "dpo":
        # Remap eval_strategy to evaluation_strategy for TrlParser
        if "--eval_strategy" in remaining_args:
            eval_strategy_index = remaining_args.index("--eval_strategy")
            remaining_args[eval_strategy_index] = "--evaluation_strategy"

        dpo_parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
        script_args, training_args, model_args = dpo_parser.parse_args_and_config(args=remaining_args)
        dpo_train(script_args, training_args, model_args)

    elif args.task == "merge":
        merge_parser = argparse.ArgumentParser()
        merge_parser.add_argument("--base_model_path", type=str, required=True)
        merge_parser.add_argument("--lora_path", type=str, required=True)
        merge_parser.add_argument("--output_path", type=str, required=True)
        merge_args = merge_parser.parse_args(remaining_args)
        apply_lora(merge_args.base_model_path, merge_args.output_path, merge_args.lora_path)

    elif args.task == "inference":
        inference_parser = argparse.ArgumentParser()
        inference_parser.add_argument("--checkpoint_dir", type=str, required=True)
        inference_parser.add_argument("--prompt", type=str)
        inference_parser.add_argument("--prompt_file", type=str)
        inference_parser.add_argument("--max_completion_length", type=int, default=2048)
        inference_args = inference_parser.parse_args(remaining_args)
        infer(inference_args)

if __name__ == "__main__":
    main()
