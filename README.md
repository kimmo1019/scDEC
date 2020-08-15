# scDEC

![model](https://github.com/kimmo1019/scDEC/blob/master/model.png)

scDEC is a computational tool for single cell ATAC-seq data analysis with deep generative neural networks. scDEC enables simultaneously learning the deep embedding and clustering of the cells in an unsupervised manner.

## Requirements
- TensorFlow==1.13.1
- Scikit-learn==0.19.0

## Installation
Download scDEC by
```shell
git clone https://github.com/kimmo1019/scDEC
```
Installation has been tested in a Linux platform with Python2.7. GPU card is recommended for accelerating the training process.

## Instructions

This section provides instructions on how to run scDEC with scATAC-seq datasets.

### Data preparation

Several scATAC-seq datasets have been prepared as the input of scDEC model. These datasets can be downloaded from the [zenode repository](https://zenodo.org/record/3984189#.XzDpJRNKhTY). Uncompress the `datasets.tar.gz` in `datasets` folder then each dataset will have its own subfolder. Each dataset will contain two major files, which denote raw read count matrix (`sc_mat.txt`) and cell label (`label.txt`), respectively. The first column of `sc_mat.txt` represents the peaks information.

### Model training

scDEC is an unsupervised learning model which is easy to run. One can run 

```python
python main_clustering.py --data [dataset] --K [nb_of_clusters] --dx [x_dim] --dy [y_dim] 
[dataset]  -  the name of the dataset (e.g.,Splenocyte)
[nb_of_clusters]  -  the number of clusters (e.g., 10)
[x_dim]  -  the dimension of latent space (continous part)
[y_dim]  -  the dimension of PCA (defalt: 20)
```
For an example, one can run `CUDA_VISIBLE_DEVICES=0 python main_clustering.py --data InSilico --K 6 --dx 9 --dy 20`.

### Model evaluation

After model training, the results will be marked by a unique timestamp YYYYMMDD_HHMMSS. This timestamp records the exact time when you run the script.

 1) `log` files and predicted assignmemnts `data_at_xxx.npz` (xxx denotes different epoch) can be found at folder `results/dataset/YYYYMMDD_HHMMSS_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0`.
 
 2) Model weights will be saved at folder `checkpoint/YYYYMMDD_HHMMSS_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0`. 
 
 3) The training loss curves were recorded at folder `graph/YYYYMMDD_HHMMSS_x_dim=10_y_dim=20_alpha=10.0_beta=10.0_ratio=0.0`, which can be visualized using TensorBoard.

 Next, one can run `python eval.py --data --timestamp --latent --vis` for visulization the latent features and save the latent feature matrix. 

 
## Contact

Also Feel free to open an issue in Github or contact `liu-q16@mails.tsinghua.edu.cn` if you have any problem in Roundtrip.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details