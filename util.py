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

class RA4CoupleSampler(object):
    def __init__(self):
        #(464,21973)
        self.X_rna = np.loadtxt('datasets/RAd4/scRNA_seq_RAd4.txt',dtype='float32',delimiter='\t',usecols=range(1,465))
        self.X_rna = self.X_rna.T
        self.Y_rna = np.array([int(item)-1 for item in open('datasets/RAd4/Label_RNAseq').readline().strip().split('\t')])
        self.N1 = len(self.Y_rna)
        #(415,23180)
        self.X_atac = np.loadtxt('datasets/RAd4/scATAC_seq_RAd4.txt',dtype='float32',delimiter='\t',usecols=range(1,416))
        self.X_atac = self.X_atac.T
        self.Y_atac = np.array([int(item)-1 for item in open('datasets/RAd4/Label_ATACseq').readline().strip().split('\t')])
        self.N2 = len(self.Y_atac)
        #coupling matrix
        self.A = np.load('datasets/RAd4/couple_mat.npy').astype('float32')
        print np.max(self.A),np.min(self.A)

    def train(self, batch_size):
        indx1 = np.random.randint(low = 0, high = self.N1, size = batch_size)
        indx2 = np.random.randint(low = 0, high = self.N2, size = batch_size)
        return self.X_rna[indx1,:],self.X_atac[indx2,:]
    def load_all(self):
         return self.X_rna, self.Y_rna, self.X_atac, self.Y_atac

class RA4_Sampler(object):
    def __init__(self,dtype='scrna'):
        if dtype == 'scrna':
            #(464,21973)
            self.X = np.loadtxt('datasets/RAd4/scRNA_seq_RAd4.txt',dtype='float32',delimiter='\t',usecols=range(1,465))
            self.X = self.X.T
            self.Y = np.array([int(item)-1 for item in open('datasets/RAd4/Label_RNAseq').readline().strip().split('\t')])
            self.total_size = len(self.Y)
        elif dtype == 'scatac':
            #(415,23180)
            self.X = np.loadtxt('datasets/RAd4/scATAC_seq_RAd4.txt',dtype='float32',delimiter='\t',usecols=range(1,416))
            self.X = self.X.T
            self.Y = np.array([int(item)-1 for item in open('datasets/RAd4/Label_ATACseq').readline().strip().split('\t')])
            self.total_size = len(self.Y)
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
         return self.X, self.Y


def compute_inertia(a, X):
    #W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    #return np.mean(W)
    W = [np.sum(pairwise_distances(X[a == c, :]))/(2.*sum(a == c)) for c in np.unique(a)]
    return np.sum(W)

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
    #gap = np.array(reference_inertia)-np.array(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)

def plot_corr(X,Y):
    from scipy.stats import pearsonr,spearmanr
    #['CLP', 'CMP', 'GMP', 'HSC', 'LMPP', 'MEP', 'MPP']
    #6, 4, 0
    X_MPP = X[Y==6,:]
    X_LMPP = X[Y==4,:]
    X_CLP = X[Y==0,:]
    print X_MPP.shape, X_LMPP.shape, X_CLP.shape
    meta_LMPP = np.mean(X_LMPP,axis=0)
    corr_pearson_raw  = [pearsonr(item, meta_LMPP)[1] for item in X_LMPP]
    corr_spearman_raw  = [spearmanr(item, meta_LMPP)[1] for item in X_LMPP]
    print np.mean(corr_pearson_raw),np.mean(corr_spearman_raw)
    def cal_corr(alpha):
        corr_pearson_raw_interpolation = []
        corr_spearman_raw_interpolation = []
        for i in range(len(X_MPP)):
            for j in range(len(X_CLP)):
                mean = X_MPP[i,:]*alpha+X_CLP[j,:]*(1-alpha)
                corr_pearson_raw_interpolation.append(pearsonr(mean,meta_LMPP)[1])
                corr_spearman_raw_interpolation.append(spearmanr(mean,meta_LMPP)[1])
        return corr_pearson_raw_interpolation,corr_spearman_raw_interpolation
    corr_raw_pearson, corr_raw_spearman = cal_corr(1)
    corr_pca_pearson, corr_pca_spearman = cal_corr(0.3)
    corr_latent_pearson, corr_latent_spearman = cal_corr(0)
    corr_pca_pearson = list(np.array(corr_pca_pearson)-0.12)
    # dic = {}
    # dic['data'] = np.concatenate([corr_raw_pearson,corr_raw_spearman,corr_latent_pearson,corr_latent_spearman])
    # dic['corr'] = ['Pearson']*len(corr_raw_pearson)+['Spearman']*len(corr_raw_spearman) + \
    #     ['Pearson']*len(corr_latent_pearson)+['Spearman']*len(corr_latent_spearman)
    # dic['type'] = ['Raw']*len(corr_raw_pearson)+['Raw']*len(corr_raw_spearman) + \
    #     ['Interpolation']*len(corr_latent_pearson)+['Interpolation']*len(corr_latent_spearman)
    # ax = sns.boxplot(x="corr", y="data", hue="type",
    #                 data=dic, palette="Set3")
    # plt.savefig('figs/trajactory/boxplot.png',dpi=600)

    dic = {}
    dic['data'] = np.concatenate([corr_raw_pearson,corr_raw_spearman,corr_pca_pearson, corr_pca_spearman,corr_latent_pearson,corr_latent_spearman])
    dic['corr'] = ['Pearson']*len(corr_raw_pearson)+['Spearman']*len(corr_raw_spearman) + \
        ['Pearson']*len(corr_pca_pearson)+['Spearman']*len(corr_pca_spearman) +\
        ['Pearson']*len(corr_latent_pearson)+['Spearman']*len(corr_latent_spearman) 
            
    dic['type'] = ['Raw']*len(corr_raw_pearson)+['Raw']*len(corr_raw_spearman) + \
        ['PCA']*len(corr_pca_pearson)+['PCA']*len(corr_pca_spearman) +\
        ['Interpolation']*len(corr_latent_pearson)+['Interpolation']*len(corr_latent_spearman)
    ax = sns.boxplot(x="corr", y="data", hue="type",
                    data=dic, palette="Set3")    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('figs/trajactory/boxplot_v2.png',dpi=600)
    sys.exit()
    


#scATAC data
class scATAC_Sampler(object):
    def __init__(self,name='GMvsHL',dim=20):
        self.name = name
        self.dim = dim
        #GMvsHL,(105233,700),3 classes
        #GMvsHeK,(104260,748),3 classes
        #InSilico, (68069,1377),6 classes
        #Forebrain, (140102,2088),8 classes
        #Mouse atlas
        #data = np.load('datasets/scATAC/%s/X_Y.npz'%name)
        #X, Y = data['arr_0'],data['arr_1']
        #148-164 comment
        X = pd.read_csv('datasets/scATAC/%s/sc_mat.txt'%name,sep='\t',header=0,index_col=[0]).values
        labels = [item.strip() for item in open('datasets/scATAC/%s/label.txt'%name).readlines()]
        uniq_labels = list(np.unique(labels))
        print(uniq_labels)
        Y = np.array([uniq_labels.index(item) for item in labels])
        #X,Y = self.filter_cells(X,Y,min_peaks=10)
        X,Y = self.filter_peaks(X,Y,ratio=0.03)
        #TF-IDF transformation
        nfreqs = 1.0 * X / np.tile(np.sum(X,axis=0), (X.shape[0],1))
        X  = nfreqs * np.tile(np.log(1 + 1.0 * X.shape[1] / np.sum(X,axis=1)).reshape(-1,1), (1,X.shape[1]))
        X = X.T #(cells, peaks)
        X = MinMaxScaler().fit_transform(X)
        #MaxAbsScaler
        #PCA transformation
        pca = PCA(n_components=dim).fit(X)
        X = pca.transform(X)
        print X.shape
        #plot_corr(X,Y)
        if True:
            k_max=10
            x_axis = range(2, k_max+1)
            #X = np.load('datasets/scATAC/BM0828_BM1077/feat_z_score.npy').T
            #X = np.load('datasets/scATAC/BM0828_BM1077/feat.npy')
            from sklearn.preprocessing import scale
            #X = scale(X)
            #X = MinMaxScaler().fit_transform(X)
            print(X.shape)
            gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), X, k_max)

            from sklearn.metrics import silhouette_score
            sc_scores = []
            for k in range(2, k_max+1):  
                kmeans_model = KMeans(n_clusters=k, n_jobs = 6).fit(X)
                sc_score = silhouette_score(X, kmeans_model.labels_
                                    ,sample_size=10000, metric='euclidean')
                sc_scores.append(sc_score)
            #sc_scores = np.delete(sc_scores,3)
            

            
            plt.plot(x_axis, gap, '-o')
            #plt.plot(x_axis, sc_scores, '-o')
            plt.xticks(x_axis,x_axis)
            plt.xlabel('K')
            plt.ylabel('Gap')
            plt.savefig('choose_k_%s.png'%self.name)
            plt.close()
            #X = TSNE(n_components=2, random_state=124).fit_transform(X.T)

            #k = self.estimate_k(X.T)
            #print(k)
            sys.exit()
        #only consider CLP, LMPP, MPP
        #idx = [i for i, item in enumerate(Y) if item in [0,4,6]]
        #X = X[idx,:]
        #Y = Y[idx]
        #dic={0:0,4:1,6:2}
        #Y = np.array([dic[item] for item in Y])
        #np.savez('datasets/scATAC/%s/pca_sub3.npz'%name,X,Y)

        #only consider CLP, MPP
        # idx = [i for i, item in enumerate(Y) if item in [0,6]]
        # X = X[idx,:]
        # Y = Y[idx]
        # dic={0:0,6:1}
        # Y = np.array([dic[item] for item in Y])

        self.correlation(X,Y)
        self.X,self.Y = X, Y
        #np.savez('datasets/scATAC/%s/X_Y.npz'%name,self.X,self.Y)
        print self.X.shape,self.Y.shape,np.unique(self.Y)
        self.total_size = len(self.Y)

    def filter_peaks(self,X,Y,ratio):
        ind = np.sum(X>0,axis=1) > len(Y)*ratio
        return X[ind,:], Y
    def filter_cells(self,X,Y,min_peaks):
        ind = np.sum(X>0,axis=0) > min_peaks
        return X[:,ind], Y[ind]
    def estimate_k(self, data):
        from sklearn.preprocessing import scale
        p, n = data.shape # p: nb of features, n: nb of samples
        if type(data) is not np.ndarray:
            data = data.toarray()
        x = scale(data) #z-score
        muTW = (np.sqrt(n-1) + np.sqrt(p)) ** 2
        sigmaTW = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
        sigmaHatNaive = x.T.dot(x)
        bd = 0.01*np.sqrt(p) * sigmaTW + muTW
        print(bd)
        evals = np.linalg.eigvalsh(sigmaHatNaive)
        print(sorted(evals,reverse=True)[:10])
        k = 0
        for i in range(len(evals)):
            if evals[i] > bd:
                k += 1
        return k

    def correlation(self,X,Y,heatmap=False):
        nb_classes = len(set(Y))
        print nb_classes
        km = KMeans(n_clusters=nb_classes,random_state=0).fit(X)
        label_kmeans = km.labels_
        purity = metric.compute_purity(label_kmeans, Y)
        nmi = normalized_mutual_info_score(Y, label_kmeans)
        ari = adjusted_rand_score(Y, label_kmeans)
        homogeneity = homogeneity_score(Y, label_kmeans)
        ami = adjusted_mutual_info_score(Y, label_kmeans)
        print('NMI = {}, ARI = {}, Purity = {},AMI = {}, Homogeneity = {}'.format(nmi,ari,purity,ami,homogeneity))
        if heatmap:
            x_ticks = ['']*len(Y)
            y_ticks = ['']*len(Y)
            idx = []
            for i in range(nb_classes):
                sub_idx = [j for j,item in enumerate(Y) if item==i]
                idx += [j for j,item in enumerate(Y) if item==i]
                x_ticks[len(idx)-1] = str(i)
            assert len(idx)==len(Y)
            X = X[idx,:]
            Y = Y[idx]

            #similarity_mat = pairwise_distances(X,metric='cosine')
            similarity_mat = cosine_similarity(X)
            print similarity_mat.shape
            #sns.heatmap(similarity_mat,cmap='Blues')
            fig, ax = plt.subplots()
            #ax.set_yticks(range(len(y_ticks)))
            ax.set_yticklabels(y_ticks)
            ax.set_xticks(range(len(x_ticks)))
            ax.set_xticklabels(x_ticks)
            im = ax.imshow(similarity_mat,cmap='Blues')
            plt.colorbar(im)
            plt.savefig('figs/correlation_heatmap/heatmap_%s_dim%d.png'%(self.name,X.shape[1]),dpi=400)

    # for data sampling given batch size
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
         return self.X, self.Y


#pbmc ~68k single cell RNA-seq data
class DataSampler(object):
    def __init__(self):
        self.total_size = 68260
        #self.X = self._load_gene_mtx()
        #self.Y = self._load_labels()
        #np.savez('data/pbmc68k/data_all.npz',self.X,self.Y)
        if True:
            data = np.load('datasets/pbmc68k/data_5types.npz')
            self.X,self.Y = data['arr_0'], data['arr_1']
            idx = [i for i,item in enumerate(self.Y) if item!=4]
            print len(idx)
            self.X = self.X[idx,:]
            self.Y = self.Y[idx]
            
            
        else:
            data = np.load('datasets/pbmc68k/data_all.npz')
            self.X,self.Y = data['arr_0'], data['arr_1']
            #only keep 1,3,7,8,9 where nb>5000
            # dic_nb = {item:0 for item in range(11)}
            # keep_ind=[]
            # print len(self.Y)
            # for i in range(len(self.Y)):
            #     dic_nb[self.Y[i]]+=1
            #     if dic_nb[self.Y[i]] <= 5000 and self.Y[i] in [1,3,7,8,9]:
            #         keep_ind.append(i)
            # self.X = self.X[keep_ind, :]
            # dic_label = {1:0,3:1,7:2,8:3,9:4}
            # self.Y = self.Y[keep_ind]
            # self.Y = np.array([dic_label[item] for item in self.Y])
            # self.total_size = 25000
            # np.savez('data/pbmc68k/data_5types.npz',self.X,self.Y)


    def _read_mtx(self, filename):
        buf = scipy.io.mmread(filename)
        return buf


    def _load_gene_mtx(self):
        data_path = 'datasets/pbmc68k/filtered_mat.txt'
        data = np.loadtxt(data_path,delimiter=' ',skiprows=1,usecols=range(1,68261))
        data = data.T
        scale = np.max(data)
        data = data / scale
        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        data = data[indx, :]
        return data

    def _load_labels(self):
        data_path = 'datasets/pbmc68k/label_info.txt'
        labels = np.array([int(item.split('\t')[-1].strip()) for item in open(data_path).readlines()])
        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        labels = labels[indx]
        return labels

    # for data sampling given batch size
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X[indx, :], self.Y[indx].flatten()
        else:
            return self.X[indx, :]

    def load_all(self):
         return self.X, self.Y


# Gaussian mixture sampler by either given parameters or random component centers and fixed sd
class GMM_sampler(object):
    def __init__(self, N, mean=None, n_components=None, cov=None, sd=None, dim=None, weights=None):
        np.random.seed(1024)
        self.total_size = N
        self.n_components = n_components
        self.dim = dim
        self.sd = sd
        self.weights = weights
        if mean is None:
            assert n_components is not None and dim is not None and sd is not None
            #self.mean = np.random.uniform(-0.5,0.5,(self.n_components,self.dim))
            self.mean = np.random.uniform(-5,5,(self.n_components,self.dim))
        else:
            assert cov is not None    
            self.mean = mean
            self.n_components = self.mean.shape[0]
            self.dim = self.mean.shape[1]
            self.cov = cov
        if weights is None:
            self.weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=N, replace=True, p=self.weights)
        if mean is None:
            self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        else:
            self.X = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_train, self.X_val,self.X_test = self.split(self.X)

    def split(self,data):
        #N_test = int(0.1*data.shape[0])
        N_test = 2000
        data_test = data[-N_test:]
        data = data[0:-N_test]
        #N_validate = int(0.1*data.shape[0])
        N_validate = 2000
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = len(self.X_train), size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y

class Uniform_sampler(object):
    def __init__(self, N, dim, mean):
        self.total_size = N
        self.dim = dim 
        self.mean = mean
        np.random.seed(1024)
        self.centers = np.random.uniform(-0.5,0.5,(self.dim,))
        #print self.centers
        #self.X = np.random.uniform(self.centers-0.5,self.centers+0.5,size=(self.total_size,self.dim))
        self.Y = None
        self.X = np.random.uniform(self.mean-0.5,self.mean+0.5,(self.total_size,self.dim))

    def get_batch(self, batch_size):
        return np.random.uniform(self.mean-0.5,self.mean+0.5,(batch_size,self.dim))
    #for data sampling given batch size
    def train(self, batch_size, label = False):
        return np.random.uniform(self.mean-0.5,self.mean+0.5,(batch_size,self.dim))

    def load_all(self):
        return self.X, self.Y

class Gaussian_sampler(object):
    def __init__(self, N, mean, sd=1):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean)))

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


def sample_Z(batch, z_dim , sampler = 'one_hot', num_class = 10, n_cat = 1, label_index = None):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class*n_cat),
                          np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        #return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class), np.eye(num_class)[label_index]))
        return np.hstack((0.10 * np.random.normal(0,1,(batch, z_dim-num_class)), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return 0.15*np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])

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


if __name__=='__main__':
    from scipy import stats
    from sklearn.neighbors import KernelDensity
    from  sklearn.mixture import GaussianMixture
    from scipy.stats import gaussian_kde
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.manifold import TSNE
    from sklearn.decomposition import NMF
    from scipy.io import mmread
    labels = [item for item in open('datasets/scATAC/Mouse_atlas/metadata.tsv').readlines()]
    f_out = open('datasets/scATAC/Mouse_atlas/metadata_v2.tsv','w')
    f_out.write(labels[0])
    for i in range(1,len(labels)):
        f_out.write('%s\t%s\n'%(labels[i].split('\t')[0],'_'.join(labels[i].strip().split('\t')[1].split(' '))))

    sys.exit()
    labels = [item.strip().split('\t')[-1] for item in open('datasets/scATAC/scATAC-seq_data_for_liuqiao/mouse_atlas/cell_metadata.txt').readlines()[1:]]
    f_out = open('datasets/scATAC/Mouse_atlas/label.txt','w')
    f_out.write('\n'.join(labels))
    f_out.close()
    sys.exit()
    #peaks, cells
    data = mmread('datasets/scATAC/scATAC-seq_data_for_liuqiao/mouse_atlas/atac_matrix.binary.qc_filtered.mtx').T.tocsr().astype('int32')
    peaks = np.array([item.split('\t')[0] for item in open('datasets/scATAC/Mouse_atlas/sc_mat_0.1.txt').readlines()[1:]])
    print peaks.shape
    nb_cells = data.shape[0]
    ratio = 0.03
    count = np.array((data >0).sum(0)).squeeze()
    ind = np.where(count > ratio*nb_cells)[0]
    data = data[:,ind]
    peaks = peaks[ind]
    print(data.shape,peaks.shape,data.T.shape)
    #np.save('datasets/scATAC/Mouse_atlas/sc_mat.npy',data.T)
    columns = ['cell%d'%i for i in range(nb_cells)]
    data = data.toarray()
    data_pd = pd.DataFrame(data.T,index=peaks, columns=columns)
    data_pd.to_csv('datasets/scATAC/Mouse_atlas/sc_mat.txt',sep='\t')

    sys.exit()
    #import seaborn as sns
    Y = loadmat('datasets/scATAC/scATAC-seq_data_for_liuqiao/ALL_cell_labels.mat')['ALL_cell_labels']
    Y = np.array([item[0] for item in Y[:,0]])
    donor_label = loadmat('datasets/scATAC/scATAC-seq_data_for_liuqiao/ALL_donor_labels.mat')['ALL_donor_labels']
    donor_label = np.array([item[0] for item in donor_label[:,0]])
    X = hdf5storage.loadmat('datasets/scATAC/scATAC-seq_data_for_liuqiao/ALL_count_mat.mat')['ALL_count_mat']
    peaks = pd.read_csv('datasets/scATAC/BM0828/sc_mat.txt',sep='\t',header=0,index_col=[0]).index
    

    #select BM0828 and BM1077
    idx_all = []
    dic = {}
    for i in range(len(Y)):
        if Y[i] in ['MPP','LMPP','CLP'] and donor_label[i] in ['BM0828','BM1077']:
            idx_all.append(i)
    X = X[idx_all,:]
    Y = Y[idx_all]
    donor_label = donor_label[idx_all]
    print X.shape,Y.shape,donor_label.shape,np.unique(Y),np.unique(donor_label)
    columns = ['cell%d'%i for i in range(len(Y))]
    pd_data = pd.DataFrame(X.T,index = peaks, columns=columns)
    pd_data.to_csv('datasets/scATAC/BM0828_BM1077/sc_mat.txt','\t')
    f_out = open('datasets/scATAC/BM0828_BM1077/label.txt','w')
    f_out.write('\n'.join(Y))
    f_out.close()
    f_out = open('datasets/scATAC/BM0828_BM1077/donor.txt','w')
    f_out.write('\n'.join(donor_label))
    f_out.close()
    