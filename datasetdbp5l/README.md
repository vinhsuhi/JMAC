The original dataset can be found here [https://github.com/stasl0217/KEnS/tree/main/data](https://github.com/stasl0217/KEnS/tree/main/data). The structure of the dataset is listed as follows:


```
datasetdbp5l/:
├── entities/
│   ├── el.tsv: entity names for language 'el'
├── kg/
│   ├── el-train.tsv: the train dataset for the completion task
│   ├── el-val.tsv: the train dataset for the completion task
│   ├── el-test.tsv: the train dataset for the completion task
├── seed_train_pairs/
│   ├── el-en.tsv: alignment training seeds
├── seed_train_pairs/
│   ├── el-en.tsv: alignment test seeds
├── relation.txt: set of relations
```

Before running the code, please download the pre-trained word embedder [wiki-news-300d-1M.vec.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) and extract it to 'JMAC/.'
If you use this dataset, please also cite this paper [https://arxiv.org/abs/2010.03158](https://arxiv.org/abs/2010.03158)
