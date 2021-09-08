# BanglaBERT

This repository contains the official release of the model **"BanglaBERT"** and associated downstream finetuning code and datasets introduced in the paper titled [**"BanglaBERT: Combating Embedding Barrier in Multilingual Models for Low-Resource Language Understanding"**](https://arxiv.org/abs/2101.00204).

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

We are releasing a slightly better checkpoint than the one reported in the paper, pretrained with 27.5 GB data, more code switched and code mixed texts, and pretrained further for 2.5M steps. The pretrained model checkpoint is available **[here](https://huggingface.co/csebuetnlp/banglabert)**. To use this model for the supported downstream tasks in this repository see **[Training & Evaluation](#training--evaluation).**


***Note:*** This model was pretrained using a ***specific normalization pipeline*** available **[here](https://github.com/csebuetnlp/normalizer)**. All finetuning scripts in this repository uses this normalization by default. If you need to adapt the pretrained model for a different task make sure ***the text units are normalized using this pipeline before tokenizing*** to get best results. A basic example is available at the **[model page](https://huggingface.co/csebuetnlp/banglabert).**

## Datasets

We are also releasing the Bangla Natural Language Inference (NLI) dataset introduced in the paper. The dataset can be found **[here](https://huggingface.co/datasets/csebuetnlp/xnli_bn)**.

## Setup

For installing the necessary requirements, use the following snippet
```bash
$ git clone https://https://github.com/csebuetnlp/banglabert
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

## Benchmarks
 
|             |   SC   |  EC   |  DC   |  NER     | NLI      |
|-------------|--------|-------|-------|----------|----------|
|`Metrics`      |   `Accuracy` | `F1*`  | `Accuracy` | `F1 (Entity)*`  | `Accuracy` |  
|[mBERT](https://huggingface.co/bert-base-multilingual-cased)        | 83.39  | 56.02 | 98.64 | 67.40    |  75.40   |
|[XLM-R](https://huggingface.co/xlm-roberta-base)        | 89.49  | 66.70 | 98.71 | 70.63    |   76.87  |    
|[sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base) |  87.30  |  61.51  |  98.79   |  70.97   |   70.48     |
[monsoon-nlp/bangla-electra](https://huggingface.co/monsoon-nlp/bangla-electra)  |  73.54  | 34.55  | 97.64     | 52.57   |   63.48   |
|***BanglaBERT***   | **92.18** | **74.27** | **99.07** | **72.18** | **82.94**|

`*` - Weighted Average

The benchmarking datasets are as follows:
* **SC:** **[Sentiment Classification](https://ieeexplore.ieee.org/document/8554396/)**
* **EC:** **[Emotion Classification](https://aclanthology.org/2021.naacl-srw.19/)**
* **DC:** **[Document Classification](https://arxiv.org/abs/2005.00085)**
* **NER:** **[Named Entity Recognition](https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs179349)**
* **NLI:** **[Natural Language Inference](#datasets)**

## Acknowledgements

We would like to thank [Intelligent Machines](https://bd.linkedin.com/company/intelligentmachines) and [Google TFRC Program](https://sites.research.google/trc/) for providing cloud support for pretraining the models.


## License
Contents of this repository are restricted to non-commercial research purposes only under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

## Citation
If you use any of the datasets, models or code modules, please cite the following paper:
```
@article{DBLP:journals/corr/abs-2101-00204,
  author    = {Abhik Bhattacharjee and
               Tahmid Hasan and
               Kazi Samin and
               M. Sohel Rahman and
               Anindya Iqbal and
               Rifat Shahriyar},
  title     = {BanglaBERT: Combating Embedding Barrier for Low-Resource Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/2101.00204},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.00204},
  eprinttype = {arXiv},
  eprint    = {2101.00204},
  timestamp = {Thu, 21 Jan 2021 14:42:30 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2101-00204.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
