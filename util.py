from __future__ import division
import scipy.sparse
import scipy.io
import numpy as np
import copy
from scipy.special import polygamma
import scipy.special
from scipy import pi
from tqdm import tqdm
import sys
import pandas as pd
from os.path import join
from collections import Counter
import cPickle as pickle
import gzip
import hdf5storage
from scipy.io import loadmat
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
import metric
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 22})
np.random.seed(0)

def compute_inertia(a, X, norm=True):
    if norm:
        W = [np.sum(pairwise_distances(X[a == c, :]))/(2.*sum(a == c)) for c in np.unique(a)]
        return np.sum(W)
    else:
        W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
        return np.mean(W)

#gap statistic 
def compute_gap(clustering, data, k_max=10, n_references=100):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference_inertia = []
    for k in range(2, k_max+1):
        reference = np.random.rand(*data.shape)
        mins = np.min(data,axis=0)
        maxs = np.max(data,axis=0)
        reference = reference*(maxs-mins)+mins
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))
    
    ondata_inertia = []
    for k in range(2, k_max+1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))
        
    gap = np.log(reference_inertia)-np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)



#scATAC data
class scATAC_Sampler(object):
    def __init__(self,name='GMvsHL',dim=20):
        self.name = name
        self.dim = dim
        X = pd.read_csv('datasets/%s/sc_mat.txt'%name,sep='\t',header=0,index_col=[0]).values
        labels = [item.strip() for item in open('datasets/%s/label.txt'%name).readlines()]
        uniq_labels = list(np.unique(labels))
        Y = np.array([uniq_labels.index(item) for item in labels])
        #X,Y = self.filter_cells(X,Y,min_peaks=10)
        X,Y = self.filter_peaks(X,Y,ratio=0.03)
        #TF-IDF transformation
        nfreqs = 1.0 * X / np.tile(np.sum(X,axis=0), (X.shape[0],1))
        X  = nfreqs * np.tile(np.log(1 + 1.0 * X.shape[1] / np.sum(X,axis=1)).reshape(-1,1), (1,X.shape[1]))
        X = X.T #(cells, peaks)
        X = MinMaxScaler().fit_transform(X)
        #PCA transformation
        #pca = PCA(n_components=dim, random_state=100).fit(X)
        pca = PCA(n_components=dim, random_state=3456).fit(X)
        X = pca.transform(X)
        self.correlation(X,Y)
        self.X, self.Y = X, Y
        self.total_size = len(self.Y)


    def filter_peaks(self,X,Y,ratio):
        ind = np.sum(X>0,axis=1) > len(Y)*ratio
        return X[ind,:], Y
    def filter_cells(self,X,Y,min_peaks):
        ind = np.sum(X>0,axis=0) > min_peaks
        return X[:,ind], Y[ind]

    def correlation(self,X,Y,heatmap=False):
        nb_classes = len(set(Y))
        print(nb_classes)
        km = KMeans(n_clusters=nb_classes,random_state=0).fit(X)
        label_kmeans = km.labels_
        purity = metric.compute_purity(label_kmeans, Y)
        nmi = normalized_mutual_info_score(Y, label_kmeans)
        ari = adjusted_rand_score(Y, label_kmeans)
        homogeneity = homogeneity_score(Y, label_kmeans)
        ami = adjusted_mutual_info_score(Y, label_kmeans)
        print('NMI = {}, ARI = {}, Purity = {},AMI = {}, Homogeneity = {}'.format(nmi,ari,purity,ami,homogeneity))
 
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
         return self.X, self.Y


#sample continuous (Gaussian) and discrete (Catagory) latent variables together
class Mixture_sampler(object):
    def __init__(self, nb_classes, N, dim, sd, scale=1):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        self.sd = sd 
        self.scale = scale
        np.random.seed(1024)
        self.X_c = self.scale*np.random.normal(0, self.sd**2, (self.total_size,self.dim))
        #self.X_c = self.scale*np.random.uniform(-1, 1, (self.total_size,self.dim))
        self.label_idx = np.random.randint(low = 0 , high = self.nb_classes, size = self.total_size)
        self.X_d = np.eye(self.nb_classes)[self.label_idx]
        self.X = np.hstack((self.X_c,self.X_d))
    
    def train(self,batch_size,weights=None):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim))
        #X_batch_c = self.scale*np.random.uniform(-1, 1, (batch_size,self.dim))
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        X_batch_d = np.eye(self.nb_classes)[label_batch_idx]
        return X_batch_c, X_batch_d

    def load_all(self):
        return self.X_c, self.X_d

#sample continuous (Gaussian Mixture) and discrete (Catagory) latent variables together
class Mixture_sampler_v2(object):
    def __init__(self, nb_classes, N, dim, weights=None,sd=0.5):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        np.random.seed(1024)
        if nb_classes<=dim:
            self.mean = np.random.uniform(-5,5,size =(nb_classes, dim))
            #self.mean = np.zeros((nb_classes,dim))
            #self.mean[:,:nb_classes] = np.eye(nb_classes)
        else:
            if dim==2:
                self.mean = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
            else:
                self.mean = np.zeros((nb_classes,dim))
                self.mean[:,:2] = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
        self.cov = [sd**2*np.eye(dim) for item in range(nb_classes)]
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        self.Y = np.random.choice(self.nb_classes, size=N, replace=True, p=weights)
        self.X_c = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_d = np.eye(self.nb_classes)[self.Y]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X_c[indx, :], self.X_d[indx, :], self.Y[indx, :]
        else:
            return self.X_c[indx, :], self.X_d[indx, :]

    def get_batch(self,batch_size,weights=None):
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        return self.X_c[label_batch_idx, :], self.X_d[label_batch_idx, :]
    def predict_onepoint(self,array):#return component index with max likelyhood
        from scipy.stats import multivariate_normal
        assert len(array) == self.dim
        return np.argmax([multivariate_normal.pdf(array,self.mean[idx],self.cov[idx]) for idx in range(self.nb_classes)])

    def predict_multipoints(self,arrays):
        assert arrays.shape[-1] == self.dim
        return map(self.predict_onepoint,arrays)
    def load_all(self):
        return self.X_c, self.X_d, self.label_idx

#get a batch of data from previous 50 batches, add stochastic
class DataPool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.nb_batch = 0
        self.pool = []

    def __call__(self, data):
        if self.nb_batch < self.maxsize:
            self.pool.append(data)
            self.nb_batch += 1
            return data
        if np.random.rand() > 0.5:
            results=[]
            for i in range(len(data)):
                idx = int(np.random.rand()*self.maxsize)
                results.append(copy.copy(self.pool[idx])[i])
                self.pool[idx][i] = data[i]
            return results
        else:
            return data

