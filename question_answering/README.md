## Data format

The finetuning script supports only `json` as input file format. The input file structure should be the same as standard QA datasets like [SQuAD v2.0](). 

## Training & Evaluation

To see list of all available options, do `python question_answering.py -h`. There are two ways to provide input data files to the script:

* with flag `--dataset_dir <path>` where `<path>` points to the directory containing files with prefix `train`, `validation` and `test`.
* with flags `--train_file <path>` / `--train_file <path>` / `--validation_file <path>` / `--test_file <path>`.

For the following commands, we are going to use the `--dataset_dir <path>` to provide input files.


### Finetuning
For finetuning on single GPU, a minimal example is as follows:

```bash
$ python ./question_answering.py \
    --model_name_or_path "csebuetnlp/banglabert" \
    --dataset_dir "sample_inputs/" \
    --output_dir "outputs/" \
    --learning_rate=2e-5 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 2 \
    --weight_decay 0.1 \
    --lr_scheduler_type "linear"  \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --max_seq_length 512 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --num_train_epochs=3 \ 
    --do_train --do_eval
```
For a detailed example, refer to **[trainer.sh](trainer.sh).**


### Evaluation
* To calculate metrics on test set / inference on raw data, use the following snippet:

```bash
$ python ./question_answering.py \
    --model_name_or_path <path/to/trained/model> \
    --dataset_dir "sample_inputs/" \
    --output_dir "outputs/" \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --do_predict
```
For a detailed example, refer to **[evaluate.sh](evaluate.sh).**
