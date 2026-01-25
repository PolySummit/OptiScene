from typing import Optional

from datasets import IterableDataset, load_dataset

# SYSTEM_PROMPT = """
# Respond in the following format:
# <answer>
# ...
# </answer>
# """

def get_bin_dataset(split="train", sft=False, sft_file=None, dpo_file=None, cache_dir=None) -> IterableDataset:
   
    if sft:
        data = load_dataset('json', data_files=sft_file, split=split)
        data = data.map(lambda x: {
            'messages': [
                {'role': 'user', 'content': x['instruction']+x['input']},
                {'role': 'assistant', 'content': x['output']},
            ]
        }, remove_columns=data.column_names)
    elif dpo_file:
        data = load_dataset('json', data_files=dpo_file, split=split)
        data = data.map(lambda x: {
            "prompt": [{"role": "user", "content":  x["combined_prompts"][0]['instruction']+x["combined_prompts"][0]['input']}],
            "chosen": [{"role": "assistant", "content": x["combined_prompts"][0]['output']}],
            "rejected": [{"role": "assistant", "content": x["combined_prompts"][1]['output']}],
        }, remove_columns=data.column_names)
    else:
        raise ValueError("Either `sft` must be True or `dpo_file` must be provided.")
        
    return data


