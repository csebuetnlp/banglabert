#!/bin/bash

# model settings
export model_name=<path/to/finetuned/model>

# qa specific settings
export doc_stride=256
export n_best_size=30
export max_answer_length=30
# misc. settings
export seed=1234

# input settings
# exactly one of `dataset_dir` or the (train / validation)
# dataset files need to be provided
input_settings=(
    "--dataset_dir inputs/sample_inputs/"
    # "--train_file sample_inputs/train.json"
    # "--validation_file sample_inputs/validation.json"
)

# output settings
export output_dir="outputs/"

# batch / sequence sizes
export PER_DEVICE_EVAL_BATCH_SIZE=16
export MAX_SEQUENCE_LENGTH=512

# optional arguments
optional_arguments=(
    "--allow_null_ans"
    "--null_score_diff_threshold 0.0"
    "--overwrite_cache"
    "--cache_dir cache_dir/"
    "--fp16"
    "--fp16_backend auto"
)

# optional for logging
# export WANDB_PROJECT="Question_answering_finetuning"
# export WANDB_WATCH=false
# export WANDB_MODE="dryrun"
export WANDB_DISABLED=true

python ./question_answering.py \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --max_seq_length $MAX_SEQUENCE_LENGTH \
    --doc_stride $doc_stride --n_best_size $n_best_size --max_answer_length $max_answer_length \
    --seed $seed --overwrite_output_dir --do_predict \
    $(echo -n ${input_settings[@]}) \
    $(echo ${optional_arguments[@]})

