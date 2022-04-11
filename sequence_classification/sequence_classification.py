# Adapted from huggingface transformers classificaton scripts

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import glob

import datasets
import numpy as np
from datasets import load_metric
from datasets.io.json import JsonDatasetReader
from datasets.io.csv import CsvDatasetReader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from normalizer import normalize

EXT2CONFIG = {
    "csv" : (CsvDatasetReader, {}),
    "tsv" : (CsvDatasetReader, {"sep": "\t"}),
    "jsonl": (JsonDatasetReader, {}),
    "json": (JsonDatasetReader, {})
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
   
    dataset_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Path to the directory containing the data files. (.csv / .tsv / .jsonl)"
            "File datatypes will be identified with their prefix names as follows: "
            "`train`- Training file(s) e.g. `train.csv`/ `train_part1.csv` etc. "
            "`validation`- Evaluation file(s) e.g. `validation.csv`/ `validation_part1.csv` etc. "
            "`test`- Test file(s) e.g. `test.csv`/ `test_part1.csv` etc. "
            "All files for must have the same extension."
        }
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv / tsv / jsonl file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv / tsv / jsonl file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv / tsv / jsonl file containing the test data."})
    do_normalize: Optional[bool] = field(default=True, metadata={"help": "Normalize text before feeding to the model."})
    unicode_norm: Optional[str] = field(default="NFKC", metadata={"help": "Type of unicode normalization"})
    remove_punct: Optional[bool] = field(
        default=False, metadata={
            "help": "Remove punctuation during normalization. To replace with custom token / selective replacement you should "
            "use this repo (https://github.com/abhik1505040/normalizer) before feeding the data to the script."
    })
    remove_emoji: Optional[bool] = field(
        default=False, metadata={
            "help": "Remove emojis during normalization. To replace with custom token / selective replacement you should "
            "use this repo (https://github.com/abhik1505040/normalizer) before feeding the data to the script."
    })
    remove_urls: Optional[bool] = field(
        default=False, metadata={
            "help": "Remove urls during normalization. To replace with custom token / selective replacement you should "
            "use this repo (https://github.com/abhik1505040/normalizer) before feeding the data to the script."
    })
    sentence1_key: Optional[str] = field(
        default="sentence1", metadata={"help": "Key / column name in the input file corresponding to the first input sequence"}
    )
    sentence2_key: Optional[str] = field(
        default="sentence2", metadata={"help": "Key / column name in the input file corresponding to the second input sequence"}
    )
    label_key: Optional[str] = field(
        default="label", metadata={"help": "Key / column name in the input file corresponding to the classification label"}
    )
    
    def __post_init__(self):
        if self.train_file is not None and self.validation_file is not None:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "jsonl", "tsv", "json"], "`train_file` should be a csv / tsv / jsonl file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension csv / tsv / jsonl as `train_file`."



@dataclass
class ModelArguments:
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    has_ext = lambda path: len(os.path.basename(path).split(".")) > 1
    get_ext = lambda path: os.path.basename(path).split(".")[-1]

    if data_args.dataset_dir is not None:
        data_files = {}
        all_files = glob.glob(
            os.path.join(
                data_args.dataset_dir,
                "*"
            )
        )
        all_exts = [get_ext(k) for k in all_files if has_ext(k)]
        if not all_exts:
            raise ValueError("The `dataset_dir` doesnt have any valid file.")
            
        selected_ext = max(set(all_exts), key=all_exts.count)
        for search_prefix in ["train", "validation", "test"]:
            found_files = glob.glob(
                os.path.join(
                    data_args.dataset_dir,
                    search_prefix + "*" + selected_ext
                )
            )
            if not found_files:
                continue

            data_files[search_prefix] = found_files
        
    else:
        data_files = {
            "train": data_args.train_file, 
            "validation": data_args.validation_file,
            "test": data_args.test_file
        }

        data_files = {k: v for k, v in data_files.items() if v is not None}
        
        if not data_files:
            raise ValueError("No valid input file found.")

        selected_ext = get_ext(list(data_files.values())[0])


    dataset_configs = EXT2CONFIG[selected_ext]
    raw_datasets = dataset_configs[0](
        data_files, 
        **dataset_configs[1]
    ).read()

    for data_type, ds in raw_datasets.items():
        assert data_args.sentence1_key in ds.features, f"Input files doesnt have the `{data_args.sentence1_key}` key"
        if data_type != "test":
            assert data_args.label_key in ds.features, f"Input files doesnt have the `{data_args.label_key}` key"
        
        ignored_columns = set(ds.column_names) - set([data_args.sentence1_key, data_args.sentence2_key, data_args.label_key])
        raw_datasets[data_type] = ds.remove_columns(ignored_columns)


    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    label_to_id = config.label2id if config.task_specific_params and config.task_specific_params.get("finetuned", False) else None
    if label_to_id is None:
        label_list = raw_datasets["train"].unique(data_args.label_key)
        label_list.sort()
        num_labels = len(label_list)
        label_to_id = {v: i for i, v in enumerate(label_list)}
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
        config.task_specific_params = {"finetuned": True}
    else:
        label_list = list(label_to_id.keys())
        num_labels = len(label_list)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.do_normalize:
        normalization_kwargs = {
            "unicode_norm": data_args.unicode_norm,
            "punct_replacement": " " if data_args.remove_punct else None,
            "url_replacement": " " if data_args.remove_urls else None,
            "emoji_replacement": " " if data_args.remove_emoji else None
        }

        def normalize_example(example):
            l = example[data_args.sentence1_key]
            example[data_args.sentence1_key] = normalize(l, **normalization_kwargs)

            if data_args.sentence2_key in example:
                l = example[data_args.sentence2_key]
                example[data_args.sentence2_key] = normalize(l, **normalization_kwargs)

            return example

        raw_datasets = raw_datasets.map(
            normalize_example,
            desc="Running normalization on dataset",
            load_from_cache_file=not data_args.overwrite_cache
        )

    
    def preprocess_function(examples):
        # Tokenize the texts   
        args = (
            (examples[data_args.sentence1_key],) if data_args.sentence2_key not in examples else (examples[data_args.sentence1_key], examples[data_args.sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        if label_to_id is not None and data_args.label_key in examples:
            result["label"] = [label_to_id[l] for l in examples[data_args.label_key]]

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
    required_metrics = [load_metric(k) for k in metric_names]
    average_required = metric_names[1:]

    def compute_metrics(p: EvalPrediction):
        results = {}
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        for m in required_metrics:
            kwargs = {"average": "macro"} if m.name in average_required else {} 
            r = m.compute(
                predictions=preds,
                references=p.label_ids,
                **kwargs
            )
            for k, v in r.items():
                results[k] = v

        return results

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=1)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        output_predict_file = os.path.join(training_args.output_dir, f"predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

    

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
