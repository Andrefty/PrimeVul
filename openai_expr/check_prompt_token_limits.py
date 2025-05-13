import argparse
import json
import os
import statistics # For mean, median, stdev

# Assuming utils.py is in the same directory or accessible
from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT

import tiktoken
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizerBase = None
    print("Warning: `transformers` library not installed. Qwen models will use tiktoken for token counting.")

# --- Copied from run_prompting.py ---
_tokenizer_cache = {}

def get_tokenizer_for_model(model_name):
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    tokenizer_obj = None
    if model_name.startswith("Qwen") and AutoTokenizer is not None:
        try:
            tokenizer_obj = AutoTokenizer.from_pretrained(model_name)
            print(f"Using Hugging Face AutoTokenizer for {model_name} for token counting.")
        except Exception as e:
            print(f"Warning: Failed to load AutoTokenizer for {model_name}. Error: {e}. Falling back to tiktoken cl100k_base.")
            tokenizer_obj = tiktoken.get_encoding("cl100k_base")
    else:
        if model_name.startswith("Qwen"): # Transformers not available or failed, print specific warning
            print(f"Using tiktoken for Qwen model {model_name} due to AutoTokenizer issue or missing 'transformers' library.")
        try:
            tokenizer_obj = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback for unknown models or if tiktoken doesn't have a direct mapping
            print(f"Warning: tiktoken model {model_name} not found. Using cl100k_base encoding.")
            tokenizer_obj = tiktoken.get_encoding("cl100k_base")

    _tokenizer_cache[model_name] = tokenizer_obj
    return tokenizer_obj

def construct_prompts(input_file, inst):
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    for sample in samples:
        key = sample.get("project", "unknown_project") + "_" + sample.get("commit_id", "unknown_commit")
        p = {"sample_key": key}
        # Ensure 'func' key exists, provide default if not (though data should have it)
        p["func"] = sample.get("func", "")
        p["target"] = sample.get("target", -1) # Assuming target is int or can be defaulted
        p["prompt"] = inst.format(func=p["func"])
        prompts.append(p)
    return prompts
# --- End of copied code ---

def calculate_prompt_tokens(messages, model_name):
    """Calculates the total tokens consumed by a list of messages for a given model."""
    tokenizer = get_tokenizer_for_model(model_name)
    is_hf_tokenizer = PreTrainedTokenizerBase is not None and isinstance(tokenizer, PreTrainedTokenizerBase)

    if model_name.startswith("Qwen"):
        tokens_per_message = 5  # As per Qwen's chat template structure
        num_tokens_consumed = 3   # Priming for assistant's reply: <|im_start|>assistant\n
    else: # OpenAI models (based on cookbook approximation)
        tokens_per_message = 3
        num_tokens_consumed = 3   # Every reply is primed with <|start|>assistant<|message|>

    for message in messages:
        num_tokens_consumed += tokens_per_message
        for key, value in message.items():
            if isinstance(value, str) and value: # Only tokenize non-empty strings
                if is_hf_tokenizer:
                    # For HF tokenizers, add_special_tokens=False is often desired for content parts
                    encoded_value = tokenizer.encode(value, add_special_tokens=False)
                else: # tiktoken
                    encoded_value = tokenizer.encode(value)
                num_tokens_consumed += len(encoded_value)
    return num_tokens_consumed

def get_total_model_capacity(model_name):
    """Returns the total context window size for a given model."""
    if model_name == "gpt-3.5-turbo-0125":
        return 16385
    elif model_name == "gpt-4-0125-preview":
        return 128000
    elif model_name.startswith("Qwen"): # e.g., Qwen/Qwen3-32B
        return 40960 # From Qwen3-32B config.json: max_position_embeddings
    else:
        print(f"Warning: Unknown model '{model_name}' for total capacity. Using a default of 4096.")
        return 4096 # A conservative default

def main():
    parser = argparse.ArgumentParser(description="Analyze prompt token lengths and available generation space.")
    parser.add_argument('--model', type=str, required=True,
                        choices=["gpt-3.5-turbo-0125", "gpt-4-0125-preview", "Qwen/Qwen3-32B", "Qwen/Qwen3-8B"],
                        help='Model name to analyze for.')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="std_cls",
                        help='Prompt strategy (std_cls or cot)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the .jsonl data file')
    parser.add_argument('--fewshot_eg', action="store_true",
                        help='Include few-shot examples in the prompts')
    args = parser.parse_args()

    if args.prompt_strategy == "std_cls":
        inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy selected.")

    prompts_data_list = construct_prompts(args.data_path, inst)
    total_model_capacity = get_total_model_capacity(args.model)
    available_for_generation_stats = []

    print(f"\nAnalyzing prompts for model: {args.model}")
    print(f"Total model context capacity: {total_model_capacity} tokens")
    print(f"Prompt strategy: {args.prompt_strategy}, Few-shot examples: {args.fewshot_eg}")
    print(f"Processing {len(prompts_data_list)} samples from {args.data_path}...\n")

    for p_data in prompts_data_list:
        if args.fewshot_eg:
            current_messages = [
                {"role": "system", "content": SYS_INST},
                {"role": "user", "content": ONESHOT_USER},
                {"role": "assistant", "content": ONESHOT_ASSISTANT},
                {"role": "user", "content": TWOSHOT_USER},
                {"role": "assistant", "content": TWOSHOT_ASSISTANT},
                {"role": "user", "content": p_data["prompt"]}
            ]
        else:
            current_messages = [
                {"role": "system", "content": SYS_INST},
                {"role": "user", "content": p_data["prompt"]}
            ]

        prompt_tokens = calculate_prompt_tokens(current_messages, args.model)
        available_for_gen = total_model_capacity - prompt_tokens

        available_for_generation_stats.append(available_for_gen)

        if available_for_gen < 0:
            print(f"  WARNING: Sample '{p_data['sample_key']}' - Prompt tokens: {prompt_tokens}, Exceeds capacity by {-available_for_gen} tokens.")

    if not available_for_generation_stats:
        print("No prompts were processed.")
        return

    print("\n--- Statistics for Available Generation Tokens ---")
    print(f"Total prompts analyzed: {len(available_for_generation_stats)}")
    min_avail = min(available_for_generation_stats)
    max_avail = max(available_for_generation_stats)
    mean_avail = statistics.mean(available_for_generation_stats)
    median_avail = statistics.median(available_for_generation_stats)

    print(f"Min available: {min_avail} tokens")
    print(f"Max available: {max_avail} tokens")
    print(f"Mean available: {mean_avail:.2f} tokens")
    print(f"Median available: {median_avail:.2f} tokens")
    if len(available_for_generation_stats) > 1:
        stdev_avail = statistics.stdev(available_for_generation_stats)
        print(f"Std Dev available: {stdev_avail:.2f} tokens")

    # Percentiles
    sorted_tokens = sorted(available_for_generation_stats)
    print("\nPercentiles for available generation tokens:")
    for p_val in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = int(len(sorted_tokens) * (p_val / 100.0))
        # Ensure index is within bounds, especially for small lists
        idx = max(0, min(idx, len(sorted_tokens) - 1))
        print(f"  {p_val}th percentile: {sorted_tokens[idx]} tokens")

    # Check against a typical max_gen_length for the model
    # Using the one from your eval_command.sh for Qwen, or typical values for others
    if args.model.startswith("Qwen"):
        target_max_gen_length = 32768
    elif args.model == "gpt-4-0125-preview":
        target_max_gen_length = 4096 # Common, though it can do more
    else: # gpt-3.5 or default
        target_max_gen_length = 4096


    supported_prompts = sum(1 for x in available_for_generation_stats if x >= target_max_gen_length)
    percentage_supported = (supported_prompts / len(available_for_generation_stats)) * 100 if available_for_generation_stats else 0
    print(f"\nPrompts that can support max_gen_length >= {target_max_gen_length} tokens: {supported_prompts} / {len(available_for_generation_stats)} ({percentage_supported:.2f}%)")

    prompts_exceeding_capacity = sum(1 for x in available_for_generation_stats if x < 0)
    if prompts_exceeding_capacity > 0:
        print(f"\nCRITICAL WARNING: {prompts_exceeding_capacity} prompts consume more tokens than the model's total capacity.")
        print("These prompts will likely cause errors or be heavily truncated by the API/server before generation even starts.")
        print("Review the 'WARNING' messages above for specific samples.")

if __name__ == "__main__":
    main()