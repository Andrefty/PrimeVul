import argparse
import os
import time
import json
from tqdm import tqdm

import openai
from openai import OpenAI
from openai._types import NOT_GIVEN

from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT

import tiktoken
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizerBase = None
    print("Warning: `transformers` library not installed. Qwen models will use tiktoken for token counting.")

# Tokenizer cache
_tokenizer_cache = {}

def get_tokenizer_for_model(model_name):
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    tokenizer_obj = None
    if model_name.startswith("Qwen") and AutoTokenizer is not None:
        try:
            tokenizer_obj = AutoTokenizer.from_pretrained(model_name)
            # print(f"Using Hugging Face AutoTokenizer for {model_name} for token counting.") # Less verbose
        except Exception as e:
            print(f"Warning: Failed to load AutoTokenizer for {model_name}. Error: {e}. Falling back to tiktoken cl100k_base.")
            tokenizer_obj = tiktoken.get_encoding("cl100k_base")
    else:
        if model_name.startswith("Qwen"):
            print(f"Using tiktoken for Qwen model {model_name} due to AutoTokenizer issue or missing 'transformers' library.")
        try:
            tokenizer_obj = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print(f"Warning: tiktoken model {model_name} not found. Using cl100k_base encoding.")
            tokenizer_obj = tiktoken.get_encoding("cl100k_base")

    _tokenizer_cache[model_name] = tokenizer_obj
    return tokenizer_obj

client = None

# --- Helper functions for token calculation and targeted truncation ---
def get_total_model_capacity(model_name):
    """Returns the total context window size for a given model."""
    if model_name == "gpt-3.5-turbo-0125":
        return 16385
    elif model_name == "gpt-4-0125-preview":
        return 128000
    elif model_name.startswith("Qwen"):
        return 40960  # Qwen3-32B max_position_embeddings
    else:
        print(f"Warning: Unknown model '{model_name}' for total capacity. Using a default of 4096.")
        return 4096

# calculate_tokens_for_message_list IS REMOVED as its logic is now integrated into get_openai_chat for precision.

def tokenize_and_truncate_string_if_needed(text_string, budget_tokens, tokenizer, is_hf_tokenizer):
    """Tokenizes a string and truncates it if it exceeds the token budget."""
    if not text_string:
        return "", 0
    if budget_tokens <= 0:
        print(f"Warning: Zero or negative budget ({budget_tokens}) for string, returning empty.")
        return "", 0

    if is_hf_tokenizer:
        token_ids = tokenizer.encode(text_string, add_special_tokens=False)
    else:
        token_ids = tokenizer.encode(text_string)

    original_token_count = len(token_ids)
    if original_token_count > budget_tokens:
        token_ids = token_ids[:budget_tokens]
        # print(f"Truncated string from {original_token_count} to {budget_tokens} tokens.")

    if is_hf_tokenizer:
        truncated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        truncated_text = tokenizer.decode(token_ids)
    return truncated_text, len(token_ids)
# --- End of helper functions ---


def get_openai_chat(prompt_data, args):
    global client # Ensure client is accessible
    
    MIN_REASONABLE_OUTPUT_SIZE = 4096 # Ensure model always has at least this much space to generate

    # Determine the prompt template (inst)
    if args.prompt_strategy == "std_cls":
        inst_template_str = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst_template_str = PROMPT_INST_COT
    else:
        raise ValueError(f"Invalid prompt strategy: {args.prompt_strategy}")

    tokenizer = get_tokenizer_for_model(args.model)
    is_hf_tokenizer = PreTrainedTokenizerBase is not None and isinstance(tokenizer, PreTrainedTokenizerBase)

    # 1. Assemble base messages (system, few-shot)
    base_messages = []
    if args.fewshot_eg:
        base_messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
        ]
    else:
        base_messages = [{"role": "system", "content": SYS_INST}]

    # 2. Calculate tokens for base messages (structure and content)
    # This helper counts tokens for a list of messages, excluding any final assistant priming.
    def _count_tokens_for_message_list_contents_and_structure(msg_list, model_name_local, tokenizer_local, is_hf_tokenizer_local):
        count = 0
        tokens_per_msg_structure = 5 if model_name_local.startswith("Qwen") else 3
        for m in msg_list:
            count += tokens_per_msg_structure
            content = m.get("content")
            if content and isinstance(content, str):
                if is_hf_tokenizer_local:
                    count += len(tokenizer_local.encode(content, add_special_tokens=False))
                else:
                    count += len(tokenizer_local.encode(content))
        return count
    
    tokens_for_base_messages_structure_and_content = _count_tokens_for_message_list_contents_and_structure(
        base_messages, args.model, tokenizer, is_hf_tokenizer
    )

    # 3. Determine assistant priming tokens (part of prompt budget, for the next turn)
    assistant_priming_tokens = 3  # Applies to both Qwen (for <|im_start|>assistant\n) and OpenAI

    # 4. Split prompt template and tokenize prefix/suffix
    if "{func}" not in inst_template_str:
        raise ValueError("Prompt template (inst_template_str) must contain '{func}' placeholder.")
    inst_parts = inst_template_str.split("{func}", 1)
    template_prefix_str = inst_parts[0]
    template_suffix_str = inst_parts[1] if len(inst_parts) > 1 else ""

    if is_hf_tokenizer:
        tokens_prefix_count = len(tokenizer.encode(template_prefix_str, add_special_tokens=False))
        tokens_suffix_count = len(tokenizer.encode(template_suffix_str, add_special_tokens=False))
    else:
        tokens_prefix_count = len(tokenizer.encode(template_prefix_str))
        tokens_suffix_count = len(tokenizer.encode(template_suffix_str))

    # 5. Original function code and its token count (untruncated)
    original_func_code = prompt_data["func"]
    if is_hf_tokenizer:
        tokens_for_original_func_code = len(tokenizer.encode(original_func_code, add_special_tokens=False))
    else:
        tokens_for_original_func_code = len(tokenizer.encode(original_func_code))
        
    # 6. Overhead for the final user message structure itself
    final_user_msg_structure_overhead = 5 if args.model.startswith("Qwen") else 3

    # 7. Calculate total tokens for the prompt if func code is NOT truncated
    prompt_tokens_if_func_untruncated = (
        tokens_for_base_messages_structure_and_content +
        assistant_priming_tokens +
        final_user_msg_structure_overhead +
        tokens_prefix_count +
        tokens_for_original_func_code +
        tokens_suffix_count
    )

    # 8. Determine effective max_gen_length for API call
    model_total_capacity = get_total_model_capacity(args.model)
    max_possible_gen_if_func_untruncated = model_total_capacity - prompt_tokens_if_func_untruncated
    
    desired_gen_length = min(args.max_gen_length, max_possible_gen_if_func_untruncated)
    effective_max_gen_length_for_api_call = max(MIN_REASONABLE_OUTPUT_SIZE, desired_gen_length)

    # Further cap: ensure effective_max_gen_length doesn't make prompt impossible
    minimal_prompt_tokens_without_func = (
        tokens_for_base_messages_structure_and_content +
        assistant_priming_tokens +
        final_user_msg_structure_overhead +
        tokens_prefix_count +
        tokens_suffix_count
    )
    # Ensure there's at least 1 token for the prompt after reserving for generation
    max_gen_allowed_by_minimal_prompt = model_total_capacity - (minimal_prompt_tokens_without_func + 1) 
    effective_max_gen_length_for_api_call = min(effective_max_gen_length_for_api_call, max_gen_allowed_by_minimal_prompt)
    effective_max_gen_length_for_api_call = max(1, effective_max_gen_length_for_api_call) # Must be at least 1

    # 9. Calculate the total budget allowed for the entire prompt based on effective generation length
    total_prompt_budget_allowed = model_total_capacity - effective_max_gen_length_for_api_call
    if total_prompt_budget_allowed <=0:
        print(f"CRITICAL: No budget for prompt for sample {prompt_data.get('sample_key', 'N/A')}. Capacity: {model_total_capacity}, Effective Gen: {effective_max_gen_length_for_api_call}")
        total_prompt_budget_allowed = 1 # Avoid negative, but expect issues

    # 10. Calculate budget for the func code tokens (content only)
    budget_for_func_code_tokens = (
        total_prompt_budget_allowed -
        (tokens_for_base_messages_structure_and_content +
         assistant_priming_tokens +
         final_user_msg_structure_overhead +
         tokens_prefix_count +
         tokens_suffix_count)
    )

    # 11. Truncate func code
    truncated_func_code_str, func_tokens_count = tokenize_and_truncate_string_if_needed(
        original_func_code,
        budget_for_func_code_tokens,
        tokenizer,
        is_hf_tokenizer
    )

    if budget_for_func_code_tokens < 0 and original_func_code:
         print(f"Warning: Negative budget ({budget_for_func_code_tokens}) for func code for sample '{prompt_data.get('sample_key', 'N/A')}'. Prompt template + base messages too long. Code will be empty.")
    elif original_func_code and not truncated_func_code_str and budget_for_func_code_tokens > 10 and len(original_func_code) > 0 : # Heuristic
        print(f"Warning: Code for sample '{prompt_data.get('sample_key', 'N/A')}' was fully truncated. Original char length: {len(original_func_code)}, Budget for code tokens: {budget_for_func_code_tokens} tokens.")
    elif func_tokens_count < tokens_for_original_func_code:
        print(f"Info: Code for sample '{prompt_data.get('sample_key', 'N/A')}' was truncated from {tokens_for_original_func_code} to {func_tokens_count} tokens. Budget: {budget_for_func_code_tokens}")


    # 12. Assemble final user message content
    final_user_content_str = template_prefix_str + truncated_func_code_str + template_suffix_str
    
    # 13. Construct final messages list for the API
    final_user_message = {"role": "user", "content": final_user_content_str}
    all_messages_for_api = base_messages + [final_user_message]
    
    extra_body_params = {}
    if args.model.startswith("Qwen"):
        if hasattr(args, 'enable_thinking'):
            extra_body_params["enable_thinking"] = args.enable_thinking
        if hasattr(args, 'top_k') and args.top_k is not None:
            extra_body_params["top_k"] = args.top_k
        if hasattr(args, 'min_p') and args.min_p is not None:
            extra_body_params["min_p"] = args.min_p

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=all_messages_for_api, 
            max_tokens=effective_max_gen_length_for_api_call, # Use the dynamically calculated value
            temperature=args.temperature,
            top_p=args.top_p if hasattr(args, 'top_p') and args.top_p is not None else NOT_GIVEN,
            seed=args.seed,
            logprobs=args.logprobs,
            top_logprobs=5 if args.logprobs else NOT_GIVEN,
            extra_body=extra_body_params if extra_body_params else NOT_GIVEN,
        )
        response_content = response.choices[0].message.content
        response_logprobs = response.choices[0].logprobs.content[0].top_logprobs if args.logprobs and response.choices[0].logprobs and response.choices[0].logprobs.content else None
        
        log_prob_mapping = {}
        if response_logprobs:
            for topl in response_logprobs:
                log_prob_mapping[topl.token] = topl.logprob
        
        return response_content, log_prob_mapping

    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as error:
        retry_time = error.retry_after if hasattr(error, "retry_after") and error.retry_after is not None else 5
        print(f"API Error: {type(error).__name__}. Sleeping for {retry_time} seconds ...")
        time.sleep(retry_time)
        return get_openai_chat(prompt_data, args) # Retry with the same prompt_data
    except openai.BadRequestError as error:
        print(f"Bad Request Error: {error} for sample '{prompt_data.get('sample_key', 'N/A')}'")
        # You might want to log the problematic `all_messages_for_api` here for debugging
        # print("Problematic messages:", json.dumps(all_messages_for_api, indent=2))
        return "ERROR_BAD_REQUEST", {} # Return a distinct error marker
    except Exception as e:
        print(f"An unexpected error occurred: {e} for sample '{prompt_data.get('sample_key', 'N/A')}'")
        return "ERROR_UNEXPECTED", {}


def construct_prompts(input_file, inst_template_str_unused): # inst_template_str is no longer used here directly for formatting
    # The formatting with {func} is now handled inside get_openai_chat after targeted truncation
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts_data = []
    for sample in samples:
        key = sample.get("project", "unknown_project") + "_" + sample.get("commit_id", "unknown_commit")
        p_data = {"sample_key": key}
        p_data["idx"] = sample.get("idx", -1) # Add this line to get 'idx'
        p_data["func"] = sample.get("func", "") # Store raw function code
        p_data["target"] = sample.get("target", -1)
        # "prompt" field is no longer pre-formatted here, as formatting happens after truncation
        prompts_data.append(p_data)
    return prompts_data


def main():
    global client
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-32B",
                        choices=["gpt-3.5-turbo-0125", "gpt-4-0125-preview", "Qwen/Qwen3-32B", "Qwen/Qwen3-8B"],
                        help='Model name')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="std_cls", help='Prompt strategy') # Corrected default
    parser.add_argument('--data_path', type=str, required=True, help='Data path')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling parameter')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter')
    parser.add_argument('--min_p', type=float, default=None, help='Min-p sampling parameter')
    parser.add_argument('--max_gen_length', type=int, default=32768, help='Max generation tokens') # Defaulted to Qwen recommendation
    parser.add_argument('--seed', type=int, default=1337, help='Seed for reproducibility') # Changed default
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    parser.add_argument('--enable_thinking', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='(For Qwen models) Enable thinking mode. Default: True')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.model.startswith("Qwen"):
        client = OpenAI(
            api_key="EMPTY", # SGLang server doesn't require a key
            base_url="http://127.0.0.1:30000/v1"
        )
    else:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI models.")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # The inst_template_str is determined in get_openai_chat, so not needed for construct_prompts
    prompts_data_list = construct_prompts(args.data_path, None) # Pass None for inst_template_str

    output_file_name = f"{args.model.replace('/', '_')}_{args.prompt_strategy}_temp{args.temperature}"
    if args.top_p is not None:
        output_file_name += f"_topp{args.top_p}"
    if args.model.startswith("Qwen"):
        if args.top_k is not None:
             output_file_name += f"_topk{args.top_k}"
        if args.min_p is not None:
             output_file_name += f"_minp{args.min_p}"
        output_file_name += f"_think{args.enable_thinking}"

    output_file_name += f"_logprobs{args.logprobs}_fewshot{args.fewshot_eg}_seed{args.seed}.jsonl"
    output_file = os.path.join(args.output_folder, output_file_name)
    
    print(f"Output will be saved to: {output_file}")
    print(f"Requesting {args.model} to respond to {len(prompts_data_list)} prompts from {args.data_path}...")
    
    with open(output_file, "w") as f:
        for p_data in tqdm(prompts_data_list):
            response_content, log_prob_mapping = get_openai_chat(p_data, args)
            
            # Store results in the original p_data dictionary
            # p_data already contains idx, func, target, sample_key
            if log_prob_mapping:
                p_data["logprobs"] = log_prob_mapping
            
            p_data["response"] = response_content if response_content is not None else "ERROR_NO_RESPONSE"
            
            f.write(json.dumps(p_data))
            f.write("\n")
            f.flush()

if __name__ == "__main__":
    main()