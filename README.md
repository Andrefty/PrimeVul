# PrimeVul -- Vulnerability Detection with Code Language Models: How Far Are We?

## 📜 Introduction

PrimeVul is a new dataset for vulnerability detection, aiming to train and evaluate code language models in the realistic vulnerability detection settings.

### Better Data

* ✨ **Diverse and Sufficient Data**: ~7k vulnerable functions and ~229k benign functions from real-world C/C++ projects, covering 140+ CWEs.
* ✨ **Accurate Labels**: Novel labeling techniques achieve human-level labeling accuracy, up to 3 $\times$ more accurate than existing automatic labeling approaches.
* ✨ **Minimal Contamination**: Thorough data de-duplication and chronological data splits minimizes the data contamination. 

### Better Evaluation

* ✨ **Realistic Tradeoff w/ VD-Score**: Measure risks of missing security flaws when not overwhelming developers with false alarms.
* ✨ **Exposing Models' Weaknesses w/ Paired Samples**: Analyzing models' capabilities in capturing subtle vulnerable patterns and distinguishing the vulnerable code from its patch.

## 📕 PrimeVul

* ✨ **Dataset**: [Google Drive Link](https://drive.google.com/drive/folders/19iLaNDS0z99N8kB_jBRTmDLehwZBolMY?usp=sharing)


## 💻 Experiments

### Install Dependencies

```conda env create -f environment.yml```

### 📖 Open-source Code LMs (< 7B)

#### Standard Binary Classification

```sh
cd os_expr;

PROJECT="primevul_cls"
TYPE=<MODEL_TYPE>
MODEL=<HUGGINGFACE_MODEL>
TOKENIZER=<HUGGINGFACE_TOKENIZER>
OUTPUT_DIR=../output/
python run_ft.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_primevul_train.jsonl> \
    --eval_data_file=<PATH_TO_primevul_valid.jsonl> \
    --test_data_file=<PATH_TO_primevul_test.jsonl> \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
cd ..;
```

#### Binary Classification + Class Weights

```sh
cd os_expr;

WEIGHT=30
PROJECT="primevul_cls_weights_${WEIGHT}"
TYPE=<MODEL_TYPE>
MODEL=<HUGGINGFACE_MODEL>
TOKENIZER=<HUGGINGFACE_TOKENIZER>
OUTPUT_DIR=../output/
python run_ft.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_primevul_train.jsonl> \
    --eval_data_file=<PATH_TO_primevul_valid.jsonl> \
    --test_data_file=<PATH_TO_primevul_test.jsonl> \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --class_weight \
    --vul_weight $WEIGHT \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
cd ..;
```

#### Binary Classification + Contrastive Learning

```sh
cd os_expr;

PROJECT="primevul_cls_clr"
TYPE=<MODEL_TYPE>
MODEL=<HUGGINGFACE_MODEL>
TOKENIZER=<HUGGINGFACE_TOKENIZER>
OUTPUT_DIR=../output/
python run_ft.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    <--clr_mask> \ # Add this parameter to enable CA-CLR
    --clr_temp 1.0 \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_primevul_train.jsonl> \
    --eval_data_file=<PATH_TO_primevul_valid.jsonl> \
    --test_data_file=<PATH_TO_primevul_test.jsonl> \
    --epoch 10 \
    --block_size 512 \
    --group_size 32 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
cd ..;
```

### 📖 Open-source Code LMs (7B)

For 7B models, since it requires more expensive computation, we implement the training differently fromm < 7B models

#### Set up
- Install [Huggingface Accelerate](https://huggingface.co/docs/accelerate/en/index).
- To train CodeGen2.5, we need transformers==4.33.0. Otherwise, there will be an error while loading the tokenizer. See this [issue](https://github.com/salesforce/CodeGen/issues/82) for further details. 
- To train StarCoder2, please use the newest transformers.
- Cconfigure Accelerate. Run `accelerate config` and choose the configuration based on your need. The configuration we used is shown in ``os_expr/default_config.yaml``.

#### Standard Binary Classification with Parallel Training
```sh
PROJECT="parallel_primevul_cls"
TYPE=<MODEL_TYPE>
MODEL=<HUGGINGFACE_MODEL>
TOKENIZER=<HUGGINGFACE_TOKENIZER>
OUTPUT_DIR=../output/
accelerate launch run_with_accelerator.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_primevul_train.jsonl> \
    --eval_data_file=<PATH_TO_primevul_valid.jsonl> \
    --test_data_file=<PATH_TO_primevul_test.jsonl> \
    --gradient_accumulation_steps 4 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

### 🤖 OpenAI Models

#### Standard Binary Classification
```sh
cd openai_expr;
MODEL=gpt-3.5-turbo-0125 # gpt-4-0125-preview
PROMPT_STRATEGY="std_cls";
python run_prompting.py \
    --model $MODEL \
    --prompt_strategy $PROMPT_STRATEGY \
    --data_path <PATH_TO_primevul_test_paired.jsonl> \
    --output_folder ../output_dir \
    --temperature 0.0 \
    --max_gen_length 1024 \
    --seed 1337 \
    --logprobs \
    --fewshot_eg
cd ..;
```

#### Chain-of-thought
```sh
cd openai_expr;
MODEL=gpt-3.5-turbo-0125 # gpt-4-0125-preview
PROMPT_STRATEGY=cot;
python run_prompting.py \
    --model $MODEL \
    --prompt_strategy $PROMPT_STRATEGY \
    --data_path <PATH_TO_primevul_test_paired.jsonl> \
    --output_folder ../output_dir \
    --temperature 0.0 \
    --max_gen_length 1024 \
    --seed 1337
cd ..;
```

### ⏲️ Calculate Vulnerability Detection Score (VD-S)

```sh
python calc_vd_score.py \
    --pred_file <OUTPUT_DIR>/predictions.txt
    --test_file <PATH_TO_primevul_test.jsonl>
```