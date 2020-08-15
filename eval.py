import argparse
import metric
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
import numpy as np
import random
import sys,os
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import pandas as pd 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

def plot_embedding(X, labels, classes=None, method='tSNE', cmap='tab20', figsize=(8, 8), markersize=15, dpi=600,marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=True, show_axis_label=True, **legend_params):
    if marker is not None:
        X = np.concatenate([X, marker], axis=0)
    N = len(labels)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    matplotlib.rcParams.update({'font.size': 22})
    if X.shape[1] != 2:
        if method == 'tSNE':
            from sklearn.manifold import TSNE
            X = TSNE(n_components=2, random_state=124).fit_transform(X)
        if method == 'PCA':
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=124).fit_transform(X)
        if method == 'UMAP':
            from umap import UMAP
            X = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(X)
    labels = np.array(labels)
    plt.figure(figsize=figsize)
    if classes is None:
        classes = np.unique(labels)
    #tab10, tab20, husl, hls
    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.husl_palette(len(classes), s=.8)
    #markersize = 80
    for i, c in enumerate(classes):
        plt.scatter(X[:N][labels==c, 0], X[:N][labels==c, 1], s=markersize, color=colors[i], label=c)
    if marker is not None:
        plt.scatter(X[N:, 0], X[N:, 1], s=10*markersize, color='black', marker='*')
    
    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 20,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method+' dim 1', fontsize=12)
        plt.ylabel(method+' dim 2', fontsize=12)

    if save:
        plt.savefig(save, format='png', bbox_inches='tight',dpi=dpi)

def cluster_eval(labels_true,labels_infer):
    purity = metric.compute_purity(labels_infer, labels_true)
    nmi = normalized_mutual_info_score(labels_true, labels_infer)
    ari = adjusted_rand_score(labels_true, labels_infer)
    homogeneity = homogeneity_score(labels_true, labels_infer)
    ami = adjusted_mutual_info_score(labels_true, labels_infer)
    #print('NMI = {}, ARI = {}, Purity = {},AMI = {}, Homogeneity = {}'.format(nmi,ari,purity,ami,homogeneity))
    return nmi,ari,homogeneity

def get_best_epoch(exp_dir, data, measurement='NMI'):
    results = []
    for each in os.listdir('results/%s/%s'%(data,exp_dir)):
        if each.startswith('data'):
            data = np.load('results/%s/%s/%s'%(data,exp_dir,each))
            data_x_onehot_,label_y = data['arr_1'],data['arr_2']
            label_infer = np.argmax(data_x_onehot_, axis=1)
            nmi,ari,homo = cluster_eval(label_y,label_infer)
            results.append([each,nmi,ari,homo])
    results.sort(key=lambda a:-a[1])
    return results[0][0]

def save_embedding(emb_feat,labels,save,sep='\t'):
    data_pd = pd.DataFrame(emb_feat,index = labels)
    data_pd.to_csv(save,sep=sep)


if __name__ == '__main__':

        parser = argparse.ArgumentParser(description='Simultaneous deep generative modeling and clustering of single cell genomic data')
        parser.add_argument('--data', '-d', type=str, help='which dataset')
        parser.add_argument('--timestamp', '-t', type=str, help='timestamp')
        parser.add_argument('--epoch', '-e', type=int, help='which epoch')
        parser.add_argument('--save', '-s', type=str, help='save latent visualization plot (e.g., t-SNE)')
        args = parser.parse_args()

        exp_dir = [item for item in os.listdir('results/%s'%args.data) if item.startswith(args.timestamp)][0]

        if args.epoch is None:
            epoch = get_best_epoch(exp_dir,args.data,'NMI')
        else:
            epoch = args.epoch
    
        data = np.load('results/%s/data_at_%d.npz'%(exp_dir,epoch))
        embedding, label_infered_onehot = data['arr_0'],data['arr_1'],data['arr_2']
        label_infered = np.argmax(label_infered_onehot, axis=1)
        label_true = [item.strip() for item  in open('datasets/scATAC/%s/label.txt'%args.data).readlines()]
        save_embedding(embedding,label_true,save='results/%s/scDEC_embedding.csv'%args.data,sep='\t')
        plot_embedding(embedding,label_true,save='results/%s/scDEC_embedding.png'%args.data,dpi=600)

