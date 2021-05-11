# scDEC

[![DOI](https://zenodo.org/badge/286327774.svg)](https://zenodo.org/badge/latestdoi/286327774)

![model](https://github.com/kimmo1019/scDEC/blob/master/model.png)

scDEC is a computational tool for single cell ATAC-seq data analysis with deep generative neural networks. scDEC enables simultaneously learning the deep embedding and clustering of the cells in an unsupervised manner. scDEC is also applicable to multi-modal single cell data. We tested it on the PBMC paired data (scRNA-seq and scATAC-seq) from 10x Genomics (see Tutorials).

## Requirements
- TensorFlow==1.13.1
- Scikit-learn==0.19.0
- Python==2.7

## Installation
Download scDEC by
```shell
git clone https://github.com/kimmo1019/scDEC
```
Installation has been tested in a Linux platform with Python2.7. GPU is recommended for accelerating the training process.

## Instructions

This section provides instructions on how to run scDEC with scATAC-seq datasets. One can also refer to [Codeocean platform](https://codeocean.com/capsule/0746056) and click `Reproducible Run` on the right. The embedding and clustering results of several datasets will be shown on the right panel.

### Data preparation

Several scATAC-seq datasets have been prepared as the input of scDEC model. These datasets can be downloaded from the [zenode repository](https://zenodo.org/record/3984189#.XzDpJRNKhTY). Uncompress the `datasets.tar.gz` in `datasets` folder then each dataset will have its own subfolder. Each dataset will contain two major files, which denote raw read count matrix (`sc_mat.txt`) and cell label (`label.txt`), respectively. The first column of `sc_mat.txt` represents the peaks information.

### Model training

scDEC is an unsupervised learning model for analyzing scATAC-seq data. One can run 

```python
python main_clustering.py --data [dataset] --K [nb_of_clusters] --dx [x_dim] --dy [y_dim] --train [is_train]
[dataset]  -  the name of the dataset (e.g.,Splenocyte)
[nb_of_clusters]  -  the number of clusters (e.g., 6)
[x_dim]  -  the dimension of Gaussian distribution
[y_dim]  -  the dimension of PCA (defalt: 20)
[is_train] - indicate training from scratch or using pretrained model

```
For an example, one can run `CUDA_VISIBLE_DEVICES=0 python main_clustering.py  --data Splenocyte --K 12 --dx 8 --dy 20` to cluster the scATAC-seq data with pretrained model. Note that the dimension of the embedding should be `K+x_dim`

Or one can run `CUDA_VISIBLE_DEVICES=0 python main_clustering.py  --data Splenocyte --K 12 --dx 8 --dy 20 --train True` to train the model from scratch.

### Model evaluation

If the pretrained model was used, the clustering results in the last step will be directly saved in `results/[dataset]/data_pre.npz` where `dataset` is the name of the scATAC-seq dataset. Note that `data_pre.npz` or `data_at_xxx.npz` contains the predictions from the H network. The first part denotes the embeddings and the second part denotes the inferred one-hot label where one can use `np.argmax` function to get the cluster label.

Then one can run `python eval.py --data [dataset]` to analyze the clustering results. 
For an example, one can run `python eval.py --data Splenocyte`

The t-SNE visualization plot of latent features (`scDEC_embedding.png`), latent feature matrix (`scDEC_embedding.csv`), inferred cluster label (`scDEC_cluster.txt`) will be saved in the `results/[dataset]` folder.


If scDEC model was trained from scratch, the results will be marked by a unique timestamp YYYYMMDD_HHMMSS. This timestamp records the exact time when you run the script. The outputs from the training includes:

 1) `log` files and predicted assignmemnts `data_at_xxx.npz` (xxx denotes different epoch) can be found at folder `results/[dataset]/YYYYMMDD_HHMMSS_x_dim=8_y_dim=20_alpha=10.0_beta=10.0_ratio=0.2`.
 
 2) Model weights will be saved at folder `checkpoint/YYYYMMDD_HHMMSS_x_dim=8_y_dim=20_alpha=10.0_beta=10.0_ratio=0.2`. 
 
 3) The training loss curves were recorded at folder `graph/YYYYMMDD_HHMMSS_x_dim=8_y_dim=20_alpha=10.0_beta=10.0_ratio=0.2`, which can be visualized using TensorBoard.

 Next, one can run 
 
```python
python eval.py --data [dataset] --timestamp [timestamp] --epoch [epoch] --train [is_train]
[dataset]  -  the name of the dataset (e.g.,Splenocyte)
[timestamp]  -  the timestamp of the experiment you ran
[epoch]  -  specify to use the results of which epoch (it can be ignored)
[is_train] - indicate training from scratch 
```

E.g., `python eval.py --data All_blood --timestamp 20200910_143208 --train True`

The t-SNE visualization plot of latent features (`scDEC_embedding.png`), latent feature matrix (`scDEC_embedding.csv`), inferred cluster label (`scDEC_cluster.txt`) will be saved in the same `results` folder as 1).


### Analyzing scATAC dataset without label

One can also use scDEC to analyze custome scATAC-seq dataset, especially the label is unknown. First, the users should prepare raw read count matrix (`sc_mat.txt`) under the folder `datasets/[NAME]`. `[NAME]` denotes the dataset name. 

Second, one can run the following command:

```python
python main_clustering.py --data [dataset] --K [nb_of_clusters] --dx [x_dim] --dy [y_dim] --train [is_train] --no_label
[dataset]  -  the name of the dataset (e.g.,Mydataset)
[nb_of_clusters]  -  the number of clusters (e.g., 6)
[x_dim]  -  the dimension of latent space (continous part)
[y_dim]  -  the dimension of PCA (defalt: 20)
[is_train] - indicate training from scratch 
```

For an example, one can run `CUDA_VISIBLE_DEVICES=0 python main_clustering.py  --data Mydataset --K 10 --dx 5 --dy 20 --train True --no_label` to clustering custom dataset.

Then one can run `python eval.py --data Mydataset --timestamp YYYYMMDD_HHMMSS --epoch epoch --no_label`. Nota time the timestamp `YYYYMMDD_HHMMSS` (for training) and epoch/batch index `epoch` (the last training epoch/batch index is recommended) should be provided. The clustering results (cluster assignments) will be saved in the `results/Mydataset/YYYYMMDD_HHMMSS_xxx` folder.



## Tutorial

[Tutorial Splenocyte](https://github.com/kimmo1019/scDEC/wiki/Splenocyte) Run scDEC on Splenocyte dataset (3166 cells)

[Tutorial Full mouse atlas](https://github.com/kimmo1019/scDEC/wiki/Full-Mouse-atlas) Run scDEC on full Mouse atlas dataset (81173 cells)

[Tutorial PBMC10k paired data ](https://github.com/kimmo1019/scDEC/wiki/PBMC10k) Run scDEC on PBMC data, which contains around 10k cells with both scRNA-seq and scATAC-seq (labels were manually annotated from 10x Genomic R&D group)
 
## Contact

Also Feel free to open an issue in Github or contact `liu-q16@mails.tsinghua.edu.cn` if you have any problem in running scDEC.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
