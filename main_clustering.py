from __future__ import division
import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
import numpy as np
import random
import copy
import math
import util
import metric
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score,homogeneity_score
import pandas as pd 


tf.set_random_seed(0)
tf.reset_default_graph()


'''
Instructions: scDEC model
    x,y - data drawn from base density (e.g., Gaussian) and observation data
    x_onehot - data drawn from caltegrory distribution
    y_  - Generated data where y_=G(x,x_onehot)
    x_latent_,x_onehot_  -  Embedding and inferred clustering label where x_latent_, x_onehot_=H(y)
    y__ - reconstructed distribution, y__ = G(H(y))
    x__ - reconstructed distribution, x__ = H(G(y))
    G(.)  - generator network for mapping latent space to data space
    H(.)  - generator network for mapping data space to latent space (embedding) and clustering, simultaneously
    Dx(.) - discriminator network in x space (latent space)
    Dy(.) - discriminator network in y space (observation space)
'''
class scDEC(object):
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, y_sampler, nb_classes, data, pool, batch_size, alpha, beta, is_train):
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net = dx_net
        self.dy_net = dy_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.pool = pool
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim


        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='x_onehot')
        self.x_combine = tf.concat([self.x,self.x_onehot],axis=1,name='x_combine')

        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.y_ = self.g_net(self.x_combine,reuse=False)

        self.x_latent_, self.x_onehot_ = self.h_net(self.y,reuse=False)#continuous + softmax + before_softmax
        self.x_ = self.x_latent_[:,:self.x_dim]
        self.x_logits_ = self.x_latent_[:,self.x_dim:]
        
        self.x_latent__, self.x_onehot__ = self.h_net(self.y_)
        self.x__ = self.x_latent__[:,:self.x_dim]
        self.x_logits__ = self.x_latent__[:,self.x_dim:]

        self.x_combine_ = tf.concat([self.x_, self.x_onehot_],axis=1)
        self.y__ = self.g_net(self.x_combine_)

        self.dy_ = self.dy_net(self.y_, reuse=False)
        self.dx_ = self.dx_net(self.x_, reuse=False)

        self.l2_loss_x = tf.reduce_mean((self.x - self.x__)**2)
        self.l2_loss_y = tf.reduce_mean((self.y - self.y__)**2)

        #self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.x_onehot, logits=self.x_logits__))
        self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.x_logits__,labels=self.x_onehot))
        
        self.g_loss_adv = -tf.reduce_mean(self.dy_)
        self.h_loss_adv = -tf.reduce_mean(self.dx_)
        

        self.g_loss = self.g_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.h_loss = self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*(self.l2_loss_x + self.l2_loss_y) + self.beta*self.CE_loss_x
       
        self.dx = self.dx_net(self.x)
        self.dy = self.dy_net(self.y)

        self.dx_loss = -tf.reduce_mean(self.dx) + tf.reduce_mean(self.dx_)
        self.dy_loss = -tf.reduce_mean(self.dy) + tf.reduce_mean(self.dy_)

        #gradient penalty for x
        epsilon_x = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon_x * self.x + (1 - epsilon_x) * self.x_
        dx_hat = self.dx_net(x_hat)
        grad_x = tf.gradients(dx_hat, x_hat)[0] #(bs,x_dim)
        grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,)
        self.gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))

        #gradient penalty for y
        epsilon_y = tf.random_uniform([], 0.0, 1.0)
        y_hat = epsilon_y * self.y + (1 - epsilon_y) * self.y_
        dy_hat = self.dy_net(y_hat)
        grad_y = tf.gradients(dy_hat, y_hat)[0] #(bs,x_dim)
        grad_norm_y = tf.sqrt(tf.reduce_sum(tf.square(grad_y), axis=1))#(bs,)
        self.gpy_loss = tf.reduce_mean(tf.square(grad_norm_y - 1.0))

        self.d_loss = self.dx_loss + self.dy_loss + 10*(self.gpx_loss + self.gpy_loss)

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        #self.d_optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr) \
        #        .minimize(self.d_loss, var_list=self.dx_net.vars+self.dy_net.vars)
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.dx_net.vars+self.dy_net.vars)

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        self.g_loss_adv_summary = tf.summary.scalar('g_loss_adv',self.g_loss_adv)
        self.h_loss_adv_summary = tf.summary.scalar('h_loss_adv',self.h_loss_adv)
        self.l2_loss_x_summary = tf.summary.scalar('l2_loss_x',self.l2_loss_x)
        self.l2_loss_y_summary = tf.summary.scalar('l2_loss_y',self.l2_loss_y)
        self.dx_loss_summary = tf.summary.scalar('dx_loss',self.dx_loss)
        self.dy_loss_summary = tf.summary.scalar('dy_loss',self.dy_loss)
        self.gpx_loss_summary = tf.summary.scalar('gpx_loss',self.gpx_loss)
        self.gpy_loss_summary = tf.summary.scalar('gpy_loss',self.gpy_loss)
        self.g_merged_summary = tf.summary.merge([self.g_loss_adv_summary, self.h_loss_adv_summary,\
            self.l2_loss_x_summary,self.l2_loss_y_summary,self.gpx_loss_summary,self.gpy_loss_summary])
        self.d_merged_summary = tf.summary.merge([self.dx_loss_summary,self.dy_loss_summary])

        #graph path for tensorboard visualization
        self.graph_dir = 'graph/{}/{}_x_dim={}_y_dim={}_alpha={}_beta={}_ratio={}'.format(self.data,self.timestamp,self.x_dim, self.y_dim, self.alpha, self.beta, ratio)
        if not os.path.exists(self.graph_dir) and is_train:
            os.makedirs(self.graph_dir)
        
        #save path for saving predicted data
        self.save_dir = 'results/{}/{}_x_dim={}_y_dim={}_alpha={}_beta={}_ratio={}'.format(self.data,self.timestamp,self.x_dim, self.y_dim, self.alpha, self.beta, ratio)
        if not os.path.exists(self.save_dir) and is_train:
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver(max_to_keep=5000)

        #run_config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=run_config)


    def train(self, nb_batches, patience):
        data_y_train = self.y_sampler.load_all()[0]
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer=tf.summary.FileWriter(self.graph_dir,graph=tf.get_default_graph())
        batches_per_eval = 100
        start_time = time.time()
        weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        last_weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        diff_history=[]
        for batch_idx in range(nb_batches):
            lr = 2e-4
            #update D
            for _ in range(5):
                bx, bx_onehot = self.x_sampler.train(self.batch_size,weights)
                by = random.sample(data_y_train,self.batch_size)
                d_summary,_ = self.sess.run([self.d_merged_summary, self.d_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
            self.summary_writer.add_summary(d_summary,batch_idx)

            bx, bx_onehot = self.x_sampler.train(self.batch_size,weights)
            by = random.sample(data_y_train,self.batch_size)

            #update G
            g_summary, _ = self.sess.run([self.g_merged_summary ,self.g_h_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
            self.summary_writer.add_summary(g_summary,batch_idx)
            #quick test on a random batch data
            if batch_idx % batches_per_eval == 0:
                g_loss_adv, h_loss_adv, CE_loss, l2_loss_x, l2_loss_y, g_loss, \
                    h_loss, g_h_loss, gpx_loss, gpy_loss = self.sess.run(
                    [self.g_loss_adv, self.h_loss_adv, self.CE_loss_x, self.l2_loss_x, self.l2_loss_y, \
                    self.g_loss, self.h_loss, self.g_h_loss, self.gpx_loss, self.gpy_loss],
                    feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by}
                )
                dx_loss, dy_loss, d_loss = self.sess.run([self.dx_loss, self.dy_loss, self.d_loss], \
                    feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by})

                print('Batch_idx [%d] Time [%.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] CE_loss [%.4f] gpx_loss [%.4f] gpy_loss [%.4f] \
                    l2_loss_x [%.4f] l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] dy_loss [%.4f] d_loss [%.4f]' %
                    (batch_idx, time.time() - start_time, g_loss_adv, h_loss_adv, CE_loss, gpx_loss, gpy_loss, l2_loss_x, l2_loss_y, \
                    g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))                 

            if (batch_idx+1) % batches_per_eval == 0:
                self.evaluate(timestamp,batch_idx)
                self.save(batch_idx)

                tol = 0.02
                estimated_weights = self.estimate_weights(use_kmeans=False)
                weights = ratio*weights + (1-ratio)*estimated_weights
                weights = weights/np.sum(weights)
                diff_weights = np.mean(np.abs(last_weights-weights))
                diff_history.append(diff_weights)
                if np.min(weights)<tol:
                    weights = self.adjust_tiny_weights(weights,tol)
                last_weights = copy.copy(weights)
            
            if len(diff_history)>100 and np.mean(diff_history[-10:]) < 5e-3 and batch_idx>30000:
                print('Reach a stable cluster')
                self.evaluate(timestamp,batch_idx)
                sys.exit()

    def adjust_tiny_weights(self,weights,tol):
        idx_less = np.where(weights<tol)[0]
        idx_greater = np.where(weights>=tol)[0]
        weights[idx_less] = np.array([np.random.uniform(2*tol,1./self.nb_classes) for item in idx_less])
        weights[idx_greater] = weights[idx_greater]*(1-np.sum(weights[idx_less]))/np.sum(weights[idx_greater])
        return weights    

    def estimate_weights(self,use_kmeans=False):
        data_y, _ = self.y_sampler.load_all()
        data_x_, data_x_onehot_ = self.predict_x(data_y)
        if use_kmeans:
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(np.concatenate([data_x_,data_x_onehot_],axis=1))
            label_infer = km.labels_
        else:
            label_infer = np.argmax(data_x_onehot_, axis=1)
        weights = np.empty(self.nb_classes, dtype=np.float32)
        for i in range(self.nb_classes):
            weights[i] = list(label_infer).count(i)  
        return weights/float(np.sum(weights)) 

    def evaluate(self,timestamp,batch_idx):
        data_y, label_y = self.y_sampler.load_all()
        N = data_y.shape[0]
        data_x_, data_x_onehot_ = self.predict_x(data_y)
        label_infer = np.argmax(data_x_onehot_, axis=1)
        purity = metric.compute_purity(label_infer, label_y)
        nmi = normalized_mutual_info_score(label_y, label_infer)
        ari = adjusted_rand_score(label_y, label_infer)
        homo = homogeneity_score(label_y,label_infer)
        print('scDEC: NMI = {}, ARI = {}, Homogeneity = {}'.format(nmi,ari,homo))
        if is_train:
            np.savez('{}/data_at_{}.npz'.format(self.save_dir, batch_idx+1),data_x_,data_x_onehot_,label_y)
            f = open('%s/log.txt'%self.save_dir,'a+')
            f.write('NMI = {}\tARI = {}\tHomogeneity = {}\t batch_idx = {}\n'.format(nmi,ari,homo,batch_idx))
            f.close()
        else:
            np.savez('results/{}/data_pre.npz'.format(self.data),data_x_,data_x_onehot_,label_y)

    #predict with y_=G(x)
    def predict_y(self, x, x_onehot, bs=256):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.y_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_x_onehot = x_onehot[ind, :]
            batch_y_ = self.sess.run(self.y_, feed_dict={self.x:batch_x, self.x_onehot:batch_x_onehot})
            y_pred[ind, :] = batch_y_
        return y_pred
    
    #predict with x_=H(y)
    def predict_x(self,y,bs=256):
        assert y.shape[-1] == self.y_dim
        N = y.shape[0]
        x_pred = np.zeros(shape=(N, self.x_dim+self.nb_classes)) 
        x_onehot = np.zeros(shape=(N, self.nb_classes)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y = y[ind, :]
            batch_x_,batch_x_onehot_ = self.sess.run([self.x_latent_, self.x_onehot_], feed_dict={self.y:batch_y})
            x_pred[ind, :] = batch_x_
            x_onehot[ind, :] = batch_x_onehot_
        return x_pred, x_onehot


    def save(self,batch_idx):

        checkpoint_dir = 'checkpoint/{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'),global_step=batch_idx)

    def load(self, pre_trained = False, timestamp='',batch_idx=999):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}'.format(self.data)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt-best'))
        else:
            if timestamp == '':
                print('Best Timestamp not provided.')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint/{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt-%d'%batch_idx))
                print('Restored model weights.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='Splenocyte',help='name of dataset')
    parser.add_argument('--model', type=str, default='model',help='file for definition of neural networks')
    parser.add_argument('--K', type=int, default=11)
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=20)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--nb_batches', type=int, default=50000,help='total number of training batches or the batch idx for loading pretrain model')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--ratio', type=float, default=0.2,help='parameter in updating Caltegory distribution')
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=bool, default=False)
    args = parser.parse_args()
    data = args.data
    model = importlib.import_module(args.model)
    nb_classes = args.K
    x_dim = args.dx
    y_dim = args.dy
    batch_size = args.bs
    nb_batches = args.nb_batches
    patience = args.patience
    alpha = args.alpha
    beta = args.beta
    ratio = args.ratio
    timestamp = args.timestamp
    is_train = args.train
    g_net = model.Generator(input_dim=x_dim,output_dim = y_dim,name='g_net',nb_layers=10,nb_units=512,concat_every_fcl=False)
    h_net = model.Encoder(input_dim=y_dim,output_dim = x_dim+nb_classes,feat_dim=x_dim,name='h_net',nb_layers=10,nb_units=256)
    dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=256)
    dy_net = model.Discriminator(input_dim=y_dim,name='dy_net',nb_layers=2,nb_units=256)
    pool = util.DataPool(10)

    xs = util.Mixture_sampler(nb_classes=nb_classes,N=10000,dim=x_dim,sd=1)
    ys = util.scATAC_Sampler(data,y_dim)

    model = scDEC(g_net, h_net, dx_net, dy_net, xs, ys, nb_classes, data, pool, batch_size, alpha, beta, is_train)

    if args.train:
        model.train(nb_batches=nb_batches, patience=patience)
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            model.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            model.load(pre_trained=False, timestamp = timestamp, batch_idx = nb_batches-1)
        model.evaluate(timestamp,nb_batches-1)
        
            
            
