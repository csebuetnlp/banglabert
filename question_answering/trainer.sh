#!/bin/bash

# training settings
export num_train_epochs=10
export save_strategy="epoch"
export logging_strategy="epoch"

# validation settings
export evaluation_strategy="epoch" 

# model settings
export model_name="csebuetnlp/banglabert"

# optimization settings
export learning_rate=2e-5
export warmup_ratio=0.1
export gradient_accumulation_steps=16
export weight_decay=0.01
export lr_scheduler_type="linear"

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
    "--dataset_dir inputs/sample_inputs"
    # "--train_file sample_inputs/train.json"
    # "--validation_file sample_inputs/validation.json"
)

# output settings
export output_dir="outputs/"

# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export PER_DEVICE_EVAL_BATCH_SIZE=2
export MAX_SEQUENCE_LENGTH=512

# optional arguments
optional_arguments=(
    "--allow_null_ans"
    "--null_score_diff_threshold 0.0"
    "--metric_for_best_model f1"
    "--greater_is_better true" # this should be commented out if the reverse is required
    "--load_best_model_at_end"
    "--logging_first_step"
    "--overwrite_cache"
    "--cache_dir cache_dir/"
    "--fp16"
    "--fp16_backend auto"
    "--do_predict"
)

# optional for logging
# export WANDB_PROJECT="Question_answering_finetuning"
# export WANDB_WATCH=false
# export WANDB_MODE="dryrun"
export WANDB_DISABLED=true

python ./question_answering.py \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --learning_rate=$learning_rate --warmup_ratio $warmup_ratio --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type  \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --max_seq_length $MAX_SEQUENCE_LENGTH --logging_strategy $logging_strategy \
    --doc_stride $doc_stride --n_best_size $n_best_size --max_answer_length $max_answer_length \
    --seed $seed --overwrite_output_dir \
    --num_train_epochs=$num_train_epochs --save_strategy $save_strategy \
    --evaluation_strategy $evaluation_strategy --do_train --do_eval \
    $(echo -n ${input_settings[@]}) \
    $(echo ${optional_arguments[@]})

