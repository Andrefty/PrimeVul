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
            print(f"Warning: tiktoken model {model_name} not found. Using cl100k_base encoding.")
            tokenizer_obj = tiktoken.get_encoding("cl100k_base")

    _tokenizer_cache[model_name] = tokenizer_obj
    return tokenizer_obj

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) # Will be initialized in main
client = None


def truncate_tokens_from_messages(messages, model, max_gen_length):
    """
    Count the number of tokens used by a list of messages, 
    and truncate the messages if the number of tokens exceeds the limit.
    Reference for OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    Qwen overheads derived from its tokenizer_config.json chat_template.
    """
    tokenizer = get_tokenizer_for_model(model)
    is_hf_tokenizer = PreTrainedTokenizerBase is not None and isinstance(tokenizer, PreTrainedTokenizerBase)

    if model == "gpt-3.5-turbo-0125":
        # Max context tokens for the model - max generation tokens requested
        max_prompt_tokens = 16385 - max_gen_length
    elif model == "gpt-4-0125-preview":
        max_prompt_tokens = 128000 - max_gen_length
    elif model.startswith("Qwen"):
        # Qwen3-32B has max_position_embeddings: 40960 (from its config.json, as per README)
        # This total capacity is split between prompt and generation.
        # README: "reserving 32,768 tokens for outputs and 8,192 tokens for typical prompts"
        # So, total_capacity = 40960
        total_qwen_capacity = 40960
        max_prompt_tokens = total_qwen_capacity - max_gen_length
        if max_prompt_tokens < 0: # Safety check
            print(f"Warning: max_gen_length ({max_gen_length}) for Qwen model exceeds or meets total capacity ({total_qwen_capacity}). Setting max_prompt_tokens to 0.")
            max_prompt_tokens = 0
    else:
        # Default for other models, assuming a smaller context like gpt-3.5-turbo if not specified
        max_prompt_tokens = 4096 - max_gen_length
    
    if model.startswith("Qwen"):
        # Qwen: Each message is <|im_start|>role\ncontent<|im_end|>\n
        # Overhead: <|im_start|> (1) + role (1) + \n (1) + <|im_end|> (1) + \n (1) = 5 tokens per message.
        tokens_per_message = 5
        # Qwen assistant reply priming: <|im_start|>assistant\n -> 3 tokens.
        num_total_tokens = 3
    else: # OpenAI models (based on cookbook approximation)
        tokens_per_message = 3  # OpenAI specific: every message follows <|start|>{role/name}\n{content}<|end|>\n
        num_total_tokens = 3  # OpenAI specific: every reply is primed with <|start|>assistant<|message|>
    
    processed_messages = []

    for message_idx, message_dict in enumerate(messages):
        # Check if there's space for the message overhead
        if num_total_tokens + tokens_per_message > max_prompt_tokens:
            print(f"Max tokens reached before message {message_idx}. Skipping remaining messages.")
            break
        
        num_total_tokens += tokens_per_message
        current_processed_message = {}

        for key, value in message_dict.items():
            if not isinstance(value, str) or not value: # Non-string or empty string values
                current_processed_message[key] = value
                continue

            # Check if there's any space left before encoding this value
            if num_total_tokens >= max_prompt_tokens:
                current_processed_message[key] = "" # No space, set to empty
                continue 

            if is_hf_tokenizer:
                encoded_value = tokenizer.encode(value, add_special_tokens=False)
            else: # tiktoken
                encoded_value = tokenizer.encode(value)
            
            value_token_count = len(encoded_value)

            if num_total_tokens + value_token_count > max_prompt_tokens:
                print(f"Truncating content for key '{key}' in message {message_idx}: '{value[:100]}...'")
                tokens_can_take_for_this_value = max_prompt_tokens - num_total_tokens
                
                if tokens_can_take_for_this_value <= 0:
                    current_processed_message[key] = ""
                else:
                    truncated_encoded_value = encoded_value[:tokens_can_take_for_this_value]
                    if is_hf_tokenizer:
                        current_processed_message[key] = tokenizer.decode(truncated_encoded_value, skip_special_tokens=True)
                    else: # tiktoken
                        current_processed_message[key] = tokenizer.decode(truncated_encoded_value)
                    num_total_tokens += len(truncated_encoded_value)
                # This field was truncated, so this message cannot contain subsequent fields.
                # Add any remaining keys from original message_dict with empty string values
                # to ensure the message structure is preserved if other code expects those keys.
                for original_key in message_dict:
                    if original_key not in current_processed_message:
                        current_processed_message[original_key] = ""
                break # Break from iterating over keys in the current message_dict
            else: # Value fits fully
                current_processed_message[key] = value # Use original value string
                num_total_tokens += value_token_count
        
        # Ensure all keys from original message are in current_processed_message,
        # even if they were not processed due to an earlier break (they'd be empty).
        # This is important if the `break` above was hit.
        for original_key in message_dict:
            if original_key not in current_processed_message:
                 # This case implies the key was after a truncated key, so it should be empty.
                current_processed_message[original_key] = ""


        processed_messages.append(current_processed_message)
        
        if num_total_tokens >= max_prompt_tokens: # Check after processing a message
            if message_idx < len(messages) -1 :
                 print(f"Max tokens reached after processing message {message_idx}. Skipping further messages.")
            break
            
    return processed_messages


# get completion from an OpenAI chat model
def get_openai_chat(
    prompt,
    args
):
    if args.fewshot_eg:
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
            {"role": "user", "content": prompt["prompt"]}
        ]
    else:
        # select the correct in-context learning prompt based on the task
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": prompt["prompt"]}
            ]
    
    # count the number of tokens in the prompt
    messages = truncate_tokens_from_messages(messages, args.model, args.max_gen_length)
    
    extra_body_params = {}
    if args.model.startswith("Qwen"):
        if hasattr(args, 'enable_thinking'):
            extra_body_params["enable_thinking"] = args.enable_thinking
        if hasattr(args, 'top_k') and args.top_k is not None: # Check if top_k is provided
            extra_body_params["top_k"] = args.top_k
        if hasattr(args, 'min_p') and args.min_p is not None: # Check if min_p is provided
            extra_body_params["min_p"] = args.min_p

    # get response from OpenAI
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_gen_length,
            temperature=args.temperature,
            top_p=args.top_p if hasattr(args, 'top_p') and args.top_p is not None else NOT_GIVEN, # Pass top_p
            seed=args.seed,
            logprobs=args.logprobs,
            top_logprobs=5 if args.logprobs else NOT_GIVEN,
            extra_body=extra_body_params if extra_body_params else NOT_GIVEN,
            )
        response_content = response.choices[0].message.content
        response_logprobs = response.choices[0].logprobs.content[0].top_logprobs if args.logprobs else None
        # map the token to the prob
        log_prob_mapping = {}
        if response_logprobs:
            for topl in response_logprobs:
                log_prob_mapping[topl.token] = topl.logprob
        # the below could be used to verify the system fingerprint and ensure the system is the same
        # system_fingrprint = response.system_fingerprint
        # print(system_fingrprint)
        
        # if the API is unstable, consider sleeping for a short period of time after each request
        # time.sleep(0.2)
        return response_content, log_prob_mapping

    # when encounter RateLimit or Connection Error, sleep for 5 or specified seconds and try again
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as error:
        retry_time = error.retry_after if hasattr(error, "retry_after") else 5
        print(f"Rate Limit or Connection Error. Sleeping for {retry_time} seconds ...")
        time.sleep(retry_time)
        return get_openai_chat(
            prompt,
            args,
        )
    # when encounter bad request errors, print the error message and return None
    except openai.BadRequestError as error:
        print(f"Bad Request Error: {error}")
        return None

def construct_prompts(input_file, inst):
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    for sample in samples:
        key = sample["project"] + "_" + sample["commit_id"]
        p = {"sample_key": key}
        p["func"] = sample["func"]
        p["target"] = sample["target"]
        p["prompt"] = inst.format(func=sample["func"])
        prompts.append(p)
    return prompts


def main():
    global client # Added to modify global client
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-32B", # Changed default
                        choices=["gpt-3.5-turbo-0125", "gpt-4-0125-preview", "Qwen/Qwen3-32B", "Qwen/Qwen3-8B"], # Added Qwen models
                        help='Model name')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="standard", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling parameter') # Added top_p
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter (for Qwen models via SGLang)') # Added top_k
    parser.add_argument('--min_p', type=float, default=None, help='Min-p sampling parameter (for Qwen models via SGLang)') # Added min_p
    parser.add_argument('--max_gen_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    parser.add_argument('--enable_thinking', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='(For Qwen models) Enable thinking mode. Default: True') # Added for Qwen
    args = parser.parse_args()

    # Initialize client based on model
    if args.model.startswith("Qwen"):
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://127.0.0.1:30000/v1" # User's SGLang server
        )
    else:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI models.")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


    output_file = os.path.join(args.output_folder, f"{args.model.replace('/', '_')}_{args.prompt_strategy}_logprobs{args.logprobs}_fewshoteg{args.fewshot_eg}.jsonl") # Replaced / in model name for filename
    if args.prompt_strategy == "std_cls":
            inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy")
    prompts = construct_prompts(args.data_path, inst)

    with open(output_file, "w") as f:
        print(f"Requesting {args.model} to respond to {len(prompts)} {args.data_path} prompts ...")
        for p in tqdm(prompts):
            response, logprobs = get_openai_chat(p, args)
            if logprobs:
                p["logprobs"] = logprobs
                print(logprobs)
            if response is None:
                response = "ERROR"
            p["response"] = response
            f.write(json.dumps(p))
            f.write("\n")
            f.flush()


if __name__ == "__main__":
    main()