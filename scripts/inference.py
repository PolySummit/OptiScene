import argparse
from modelscope import AutoModelForCausalLM, AutoTokenizer

from prompt import floor_plan_prompts

def infer(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    prompt = floor_plan_prompts
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt += f.read()
    elif args.prompt:
        prompt += args.prompt
    else:
        with open("default_prompt.txt", "r", encoding="utf-8") as f:
            prompt += f.read()

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_completion_length
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(response)
    print(f"Assistant:\n{response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    args = parser.parse_args()
    infer(args)
