# BanglaBERT

This repository contains the official release of the model **"BanglaBERT"** and associated downstream finetuning code and datasets introduced in the paper titled [**"BanglaBERT: Language Model Pretraining and Benchmarks for
Low-Resource Language Understanding Evaluation in Bangla"**](https://aclanthology.org/2022.findings-naacl.98/) accpeted in *Findings of the Association for Computational Linguistics: NAACL 2022*.

## Table of Contents

- [BanglaBERT](#banglabert)
  - [Table of Contents](#table-of-contents)
  - [Models](#models)
  - [Datasets](#datasets)
  - [Setup](#setup)
  - [Training & Evaluation](#training--evaluation)
  - [Benchmarks](#benchmarks)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)
  - [Citation](#citation)

## Models

The pretrained model checkpoints are available at [Huggingface model hub](https://huggingface.co/csebuetnlp).

- [**BanglaBERT**](https://huggingface.co/csebuetnlp/banglabert)
- [**BanglishBERT**](https://huggingface.co/csebuetnlp/banglishbert)
  
To use these models for the supported downstream tasks in this repository see **[Training & Evaluation](#training--evaluation).**

***Note:*** These models were pretrained using a ***specific normalization pipeline*** available **[here](https://github.com/csebuetnlp/normalizer)**. All finetuning scripts in this repository uses this normalization by default. If you need to adapt the pretrained model for a different task make sure ***the text units are normalized using this pipeline before tokenizing*** to get best results. A basic example is available at the **[model page](https://huggingface.co/csebuetnlp/banglabert).**

## Datasets

We are also releasing the Bangla Natural Language Inference (NLI) and Bangla Question Answering (QA) datasets introduced in the paper. 
- [**NLI**](https://huggingface.co/datasets/csebuetnlp/xnli_bn)
- [**QA**](https://huggingface.co/datasets/csebuetnlp/squad_bn)

## Setup

For installing the necessary requirements, use the following bash snippet
```bash
$ git clone https://github.com/csebuetnlp/banglabert
$ cd banglabert/
$ conda create python==3.7.9 pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -p ./env
$ conda activate ./env # or source activate ./env (for older versions of anaconda)
$ bash setup.sh 
```
* Use the newly created environment for running the scripts in this repository.

## Training & Evaluation

To use the pretrained model for finetuning / inference on different downstream tasks see the following section:

* **[Sequence Classification](sequence_classification/).**
  - For single sequence classification such as
    - Document classification
    - Sentiment classification
    - Emotion classification etc.
  - For double sequence classification such as 
    - Natural Language Inference (NLI)
    - Paraphrase detection etc.
- **[Token Classification](token_classification/).**
  - For token tagging / classification tasks such as
    - Named Entity Recognition (NER)
    - Parts of Speech Tagging (PoS) etc.
- **[Question Answering](question_answering/).**
    - For tasks such as,
      - Extractive Question Answering
      - Open-domain Question Answering


## Benchmarks
 
* Zero-shot cross-lingual transfer-learning

|     Model          |   Params   |     SC (macro-F1)     |      NLI (accuracy)     |    NER  (micro-F1)   |   QA (EM/F1)   |   BangLUE score |
|----------------|-----------|-----------|-----------|-----------|-----------|-----------|
|[mBERT](https://huggingface.co/bert-base-multilingual-cased) | 180M  | 27.05 | 62.22 | 39.27 | 59.01/64.18 |  50.35 |
|[XLM-R (base)](https://huggingface.co/xlm-roberta-base) |  270M   | 42.03 | 72.18 | 45.37 | 55.03/61.83 |  55.29 |
|[XLM-R (large)](https://huggingface.co/xlm-roberta-large) | 550M  | 49.49 | 78.13 | 56.48 | 71.13/77.70 |  66.59 |
|[BanglishBERT](https://huggingface.co/csebuetnlp/banglishbert) | 110M | 48.39 | 75.26 | 55.56 | 72.87/78.63 | 66.14 |

* Supervised fine-tuning

|     Model          |   Params   |     SC (macro-F1)     |      NLI (accuracy)     |    NER  (micro-F1)   |   QA (EM/F1)   |   BangLUE score |
|----------------|-----------|-----------|-----------|-----------|-----------|-----------|
|[mBERT](https://huggingface.co/bert-base-multilingual-cased) | 180M  | 67.59 | 75.13 | 68.97 | 67.12/72.64 | 70.29 |
|[XLM-R (base)](https://huggingface.co/xlm-roberta-base) |  270M   | 69.54 | 78.46 | 73.32 | 68.09/74.27  | 72.82 |        
|[XLM-R (large)](https://huggingface.co/xlm-roberta-large) | 550M  | 70.97 | 82.40 | 78.39 | 73.15/79.06 | 76.79 |
|[sahajBERT](https://huggingface.co/neuropark/sahajBERT) | 18M | 71.12 | 76.92 | 70.94 | 65.48/70.69 | 71.03 |
|[BanglishBERT](https://huggingface.co/csebuetnlp/banglishbert) | 110M | 70.61 | 80.95 | 76.28 | 72.43/78.40 | 75.73 |
|[BanglaBERT](https://huggingface.co/csebuetnlp/banglabert) | 110M | 72.89 | 82.80 | 77.78 | 72.63/79.34 | **77.09** |


The benchmarking datasets are as follows:
* **SC:** **[Sentiment Classification](https://aclanthology.org/2021.findings-emnlp.278)**
* **NER:** **[Named Entity Recognition](https://multiconer.github.io/competition)**
* **NLI:** **[Natural Language Inference](#datasets)**
* **QA:** **[Question Answering](#datasets)**
  
## Acknowledgements

We would like to thank [Intelligent Machines](https://bd.linkedin.com/company/intelligentmachines) and [Google TFRC Program](https://sites.research.google/trc/) for providing cloud support for pretraining the models.


## License
Contents of this repository are restricted to non-commercial research purposes only under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

## Citation
If you use any of the datasets, models or code modules, please cite the following paper:
```
@inproceedings{bhattacharjee-etal-2022-banglabert,
    title = "{B}angla{BERT}: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in {B}angla",
    author = "Bhattacharjee, Abhik  and
      Hasan, Tahmid  and
      Ahmad, Wasi  and
      Mubasshir, Kazi Samin  and
      Islam, Md Saiful  and
      Iqbal, Anindya  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.98",
    pages = "1318--1327",
    abstract = "In this work, we introduce BanglaBERT, a BERT-based Natural Language Understanding (NLU) model pretrained in Bangla, a widely spoken yet low-resource language in the NLP literature. To pretrain BanglaBERT, we collect 27.5 GB of Bangla pretraining data (dubbed {`}Bangla2B+{'}) by crawling 110 popular Bangla sites. We introduce two downstream task datasets on natural language inference and question answering and benchmark on four diverse NLU tasks covering text classification, sequence labeling, and span prediction. In the process, we bring them under the first-ever Bangla Language Understanding Benchmark (BLUB). BanglaBERT achieves state-of-the-art results outperforming multilingual and monolingual models. We are making the models, datasets, and a leaderboard publicly available at \url{https://github.com/csebuetnlp/banglabert} to advance Bangla NLP.",
}
```
