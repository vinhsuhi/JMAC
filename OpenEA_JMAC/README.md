# [JMAC for DBPv1 dataset](https://www.pytorch.org)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://www.pytorch.org)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-red.svg?style=flat-square)](https://www.pytorch.org/)
[![Paper](https://img.shields.io/badge/EMNLP%202022-PDF-yellow.svg?style=flat-square)](https://www.pytorch.org)

> This is the implementation of JMAC for the widely used KG alignment dataset OpenEA_dataset_v1.1.


## Table of contents
1. [Overview](#overview)
2. [Code Tree Description](#code-tree-description)
3. [Dataset](#dataset)
4. [Dependencies](#dependencies)
5. [Usage](#usage)
6. [Citation](#citation)


## Overview

We use [Python](https://www.python.org/) and [Pytorch](https://www.pytorch.org/) to implement a Knowledge Graph Matching algorithm named **OpenEA_JMAC** for the KG alignment dataset DBPv1. 


## Code Tree Description

```
OpenEA_dataset_v1.1/: where we put our data, you must download data first and decompress it to achieve this folder
trainer/: where we put our trainer
models/:
├── neural/: where we implement our GNN (alignment-based channel)
├── trans/: where we implement our shallow model (transitivity-based channel)
modules/: for other purpose such as load data, compute alignment results...
```

## Dataset

The datasets used in this paper are introduced in [here](https://github.com/nju-websoft/OpenEA) can be downloaded from [Dropbox](https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0). Download Pretrained word vectors at [here](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip), unzip it and save to the datasets folder.

The directory structure of each dataset is listed as follows (an example for EN_FR_15K_V1 dataset):

```
OpenEA_dataset_v1.1/:
├── EN_FR_15K_V1/
│   ├── attr_triples_1: attribute triples in KG1
│   ├── attr_triples_2: attribute triples in KG2
│   ├── rel_triples_1: relation triples in KG1
│   ├── rel_triples_2: relation triples in KG2
│   ├── ent_links: entity alignment between KG1 and KG2
│   ├── 721_5fold/: entity alignment with test/train/valid (7:2:1) splits
│   │   ├── 1/: the first fold
│   │   │   ├── train_links
│   │   │   ├── valid_links
│   │   │   └── test_links
│   │   ├── 2/
│   │   ├── 3/
│   │   ├── 4/
│   │   ├── 5/
```


## Dependencies
* Python 3.x
* pytorch 
* pandas
* scikit-learn
* numpy
* scipy
* torch-scatter
* tqdm


## Usage

To reproduce our experiments, please use the following script:

```bash
python main_with_args.py --training_data datasets/D_W_15K_V1/ 
```

## Citation
If you find the our implementation useful, please kindly cite the following paper:

```
@article{tong2022jmac,
  title={Joint Multilingual Knowledge Graph Completion and Alignment},
  author={Tong, Vinh and Nguyen, Dat Quoc and Trung, Huynh Thanh and Nguyen, Thanh Tam and Nguyen, Quoc Viet Hung and Mathias, Niepert},
  journal={Findings of the Association for Computational Linguistics: EMNLP 2022},
  year={2022}
}
```
