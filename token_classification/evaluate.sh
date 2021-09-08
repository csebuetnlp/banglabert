#!/bin/bash

# misc. settings
export seed=1234

# model settings
export model_name=<path/to/finetuned/model>

# input settings
# exactly one of `dataset_dir` or the test
# dataset file needs to be provided
input_settings=(
    # "--dataset_dir sample_inputs/"
    "--test_file sample_inputs/sample_test_without_tags.jsonl"
)

# output settings
export output_dir="outputs/"

# batch sizes
export PER_DEVICE_EVAL_BATCH_SIZE=8

# optional_arguments
optional_arguments=(
    "--cache_dir cache_dir/"
    "--overwrite_cache"
)

# optional for logging
# export WANDB_PROJECT="Token_classification_finetuning"
# export WANDB_WATCH=false
# export WANDB_MODE="dryrun"
export WANDB_DISABLED=true

python ./token_classification.py \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --max_seq_length $MAX_SEQUENCE_LENGTH \
    --seed $seed --overwrite_output_dir  --do_predict \
    $(echo -n ${input_settings[@]}) \
    $(echo ${optional_arguments[@]})
