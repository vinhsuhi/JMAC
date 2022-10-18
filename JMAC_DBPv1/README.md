# [JMAC for DBPv1 (OpenEA_dataset_v1.1) dataset](https://arxiv.org/abs/2210.08922)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://www.pytorch.org)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-red.svg?style=flat-square)](https://www.pytorch.org/)
[![Paper](https://img.shields.io/badge/EMNLP%202022-PDF-yellow.svg?style=flat-square)](https://www.pytorch.org)

> JMAC for the benchmark dataset OpenEA_dataset_v1.1.

## Dataset

The dataset is introduced [here](https://github.com/nju-websoft/OpenEA) and can be downloaded from [Dropbox](https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0). Download the pre-trained word vectors file [wiki-news-300d-1M.vec.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip), unzip it and save it to the datasets folder.

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
# w/ SI 
python main_with_args.py --training_data datasets/EN_FR_15K_V1/ 
# w/o SI
python main_with_args.py --training_data datasets/EN_FR_15K_V1/ --no_name_info --completion_dropout_rate 0.1
```




## Citation
Details of the model architecture and experimental results can be found in [our EMNLP 2022 Findings paper](https://arxiv.org/abs/2210.08922):

```
@inproceedings{tong2022jmac,
  title     = {{Joint Multilingual Knowledge Graph Completion and Alignment}},
  author    = {Tong, Vinh and Nguyen, Dat Quoc and Trung, Huynh Thanh and Nguyen, Thanh Tam and Nguyen, Quoc Viet Hung and Mathias, Niepert},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  year      = {2022}
}
```

**Please CITE** our paper whenever our JMAC is used to help produce published results or incorporated into other software.

