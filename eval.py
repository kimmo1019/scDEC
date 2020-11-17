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

def plot_embedding(X, labels, classes=None, method='tSNE', cmap='tab20', figsize=(8, 8), markersize=15, dpi=300,marker=None,
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
    print('NMI = {}, ARI = {}, Purity = {},AMI = {}, Homogeneity = {}'.format(nmi,ari,purity,ami,homogeneity))
    return nmi,ari,homogeneity

def get_best_epoch(exp_dir, dataset, measurement='NMI'):
    results = []
    for each in os.listdir('results/%s/%s'%(dataset,exp_dir)):
        if each.startswith('data'):
            #print('results/%s/%s/%s'%(dataset,exp_dir,each))
            data = np.load('results/%s/%s/%s'%(dataset,exp_dir,each))
            data_x_onehot_,label_y = data['arr_1'],data['arr_2']
            label_infer = np.argmax(data_x_onehot_, axis=1)
            nmi,ari,homo = cluster_eval(label_y,label_infer)
            results.append([each,nmi,ari,homo])
    if measurement == 'NMI':
        results.sort(key=lambda a:-a[1])
    elif measurement == 'ARI':
        results.sort(key=lambda a:-a[2])
    elif measurement == 'HOMO':
        results.sort(key=lambda a:-a[3])
    else:
        print('Wrong indicated metric')
        sys.exit()
    print('NMI = {}\tARI = {}\tHomogeneity = {}'.format(results[0][1],results[0][2],results[0][3]))
    return results[0][0]

def save_embedding(emb_feat,save,sep='\t'):
    index = ['cell%d'%(i+1) for i in range(emb_feat.shape[0])]
    columns = ['feat%d'%(i+1) for i in range(emb_feat.shape[1])]
    data_pd = pd.DataFrame(emb_feat,index = index,columns=columns)
    data_pd.to_csv(save,sep=sep)

def save_clustering(label,save):
    f = open(save,'w')
    res_list = ['cell%d\t%s'%(i,str(item)) for i,item in enumerate(label)]
    f.write('\n'.join(res_list))
    f.close()

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Simultaneous deep generative modeling and clustering of single cell genomic data')
        parser.add_argument('--data', '-d', type=str, help='which dataset')
        parser.add_argument('--timestamp', '-t', type=str, help='timestamp')
        parser.add_argument('--epoch', '-e', type=int, help='epoch or batch index')
        parser.add_argument('--train', type=bool, default=False)
        parser.add_argument('--save', '-s', type=str, help='save latent visualization plot (e.g., t-SNE)')
        parser.add_argument('--no_label', action='store_true',help='whether the dataset has label')
        args = parser.parse_args()
        has_label = not args.no_label
        if has_label:
            if args.train:
                exp_dir = [item for item in os.listdir('results/%s'%args.data) if item.startswith(args.timestamp)][0]
                if args.epoch is None:
                    epoch = get_best_epoch(exp_dir,args.data,'ARI')
                else:
                    epoch = args.epoch
                data = np.load('results/%s/%s/%s'%(args.data,exp_dir,epoch))
                embedding, label_infered_onehot = data['arr_0'],data['arr_1']
                embedding_before_softmax = embedding[:,-label_infered_onehot.shape[1]:]
                label_infered = np.argmax(label_infered_onehot, axis=1)
                label_true = [item.strip() for item  in open('datasets/%s/label.txt'%args.data).readlines()]
                save_clustering(label_infered,save='results/%s/%s/scDEC_cluster.txt'%(args.data,exp_dir))
                save_embedding(embedding,save='results/%s/%s/scDEC_embedding.csv'%(args.data,exp_dir),sep='\t')
                plot_embedding(embedding,label_true,save='results/%s/%s/scDEC_embedding.png'%(args.data,exp_dir))
            else:
                if args.data == 'PBMC10k':
                    data = np.load('results/%s/data_pre.npz'%args.data)
                    embedding, label_infered_onehot = data['arr_0'],data['arr_1']
                    embedding_before_softmax = embedding[:,-label_infered_onehot.shape[1]:]
                    label_infered = np.argmax(label_infered_onehot, axis=1)
                    barcode2label = {item.split('\t')[0]:item.split('\t')[1].strip() for item in open('datasets/%s/labels_annot.txt'%args.data).readlines()[1:]}
                    barcodes = [item.strip() for item in open('datasets/%s/barcodes.tsv'%args.data).readlines()]
                    labels_annot = [barcode2label[item] for i,item in enumerate(barcodes) if item in barcode2label.keys()]
                    select_idx = [i for i,item in enumerate(barcodes) if item in barcode2label.keys()]
                    embedding = embedding[select_idx,:] # only evaluated on cells with annotation labels
                    label_infered = label_infered[select_idx]
                    uniq_label = list(np.unique(labels_annot))
                    Y = np.array([uniq_label.index(item) for item in labels_annot])
                    cluster_eval(Y,label_infered)
                    save_clustering(label_infered,save='results/%s/scDEC_cluster.txt'%args.data)
                    save_embedding(embedding,save='results/%s/scDEC_embedding.csv'%args.data,sep='\t')
                    plot_embedding(embedding,labels_annot,save='results/%s/scDEC_embedding.png'%args.data)
                else:
                    data = np.load('results/%s/data_pre.npz'%args.data)
                    embedding, label_infered_onehot = data['arr_0'],data['arr_1']
                    embedding_before_softmax = embedding[:,-label_infered_onehot.shape[1]:]
                    label_infered = np.argmax(label_infered_onehot, axis=1)
                    label_true = [item.strip() for item  in open('datasets/%s/label.txt'%args.data).readlines()]
                    save_clustering(label_infered,save='results/%s/scDEC_cluster.txt'%args.data)
                    save_embedding(embedding,save='results/%s/scDEC_embedding.csv'%args.data,sep='\t')
                    plot_embedding(embedding,label_true,save='results/%s/scDEC_embedding.png'%args.data)
        else:
            if args.epoch is None:
                print('Provide the epoch or batch index to analyze')
                sys.exit()
            else:
                exp_dir = [item for item in os.listdir('results/%s'%args.data) if item.startswith(args.timestamp)][0]
                data = np.load('results/%s/%s/data_at_%s.npz'%(args.data,exp_dir,args.epoch))
                embedding, label_infered_onehot = data['arr_0'],data['arr_1']
                label_infered = np.argmax(label_infered_onehot, axis=1)
                save_clustering(label_infered,save='results/%s/%s/scDEC_cluster.txt'%(args.data,exp_dir))
                


    

