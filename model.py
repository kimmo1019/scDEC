import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
#the default is relu function
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)
    #return tf.maximum(0.0, x)
    #return tf.nn.tanh(x)
    #return tf.nn.elu(x)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x , y*tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] ,tf.shape(y)[3]])], 3)

class Discriminator(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            #fc = tcl.batch_norm(fc)
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = tcl.batch_norm(fc)
                #fc = leaky_relu(fc)
                fc = tf.nn.tanh(fc)
            
            output = tcl.fully_connected(
                fc, 1, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256, concat_every_fcl=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.concat_every_fcl = concat_every_fcl
        
    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            y = z[:,self.input_dim:]
            fc = tcl.fully_connected(
                z, self.nb_units,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = True)
            fc = leaky_relu(fc)
            #fc = tf.nn.dropout(fc,0.1)
            if self.concat_every_fcl:
                fc = tf.concat([fc, y], 1)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tcl.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = True)
                
                fc = leaky_relu(fc)
                if self.concat_every_fcl:
                    fc = tf.concat([fc, y], 1)
            
            output = tcl.fully_connected(
                fc, self.output_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                #activation_fn=tf.sigmoid
                activation_fn=tf.identity
                )
            #output = tc.layers.batch_norm(output,decay=0.9,scale=True,updates_collections=None,is_training = True)
            #output = tf.nn.relu(output)
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class MutualNet(object):# only categrory variable
    def __init__(self, output_dim, name,nb_units=256):
        self.output_dim = output_dim
        self.name = name
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc = leaky_relu(fc)
            
            output = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            prob = tf.nn.softmax(output)
            return prob

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class MutualNet_v2(object):# catogory + continuous
    def __init__(self, latent_dim,nb_classes, name,nb_units=256):
        self.latent_dim = latent_dim
        self.nb_classes = nb_classes
        self.name = name
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc = leaky_relu(fc)
            
            output = tcl.fully_connected(
                fc, self.nb_classes+self.latent_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            prob = tf.nn.softmax(output[:,:self.nb_classes])
            return prob,output[:,self.nb_classes:self.nb_classes+self.latent_dim]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
class GeneratorCouple(object):
    def __init__(self, input_dim, feat_dim, output_dim1, output_dim2, name, nb_layers=2, nb_units=256, concat_every_fcl=True):
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.nb_classes = (input_dim-feat_dim)/2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.concat_every_fcl = concat_every_fcl
        
    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            y1 = z[:,-2*self.nb_classes:-self.nb_classes]
            y2 = z[:,-self.nb_classes:]
            fc = tcl.fully_connected(
                z, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            if self.concat_every_fcl:
                fc = tf.concat([fc, y1, y2], 1)
            for i in range(self.nb_layers-1):
                if i== self.nb_layers-2:
                    fc = tcl.fully_connected(
                        fc, 2*self.nb_units,
                        #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                        activation_fn=tf.identity
                    )
                else:
                    fc = tcl.fully_connected(
                        fc, self.nb_units,
                        #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                        activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)
                if self.concat_every_fcl:
                    fc = tf.concat([fc, y1, y2], 1)
            
            output1 = tcl.fully_connected(
                fc[:,:self.nb_units], self.output_dim1,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.nn.relu
                )
            output2 = tcl.fully_connected(
                fc[:,self.nb_units:], self.output_dim2,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.nn.relu
                )
            return output1,output2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator_resnet(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def residual_block(self, x, dim):
        e = tcl.fully_connected(x, self.nb_units, activation_fn=tf.identity)
        e = leaky_relu(e)
        e = tcl.fully_connected(x, dim, activation_fn=tf.identity)
        e = leaky_relu(e)
        return x+e
        
    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                z, self.nb_units/2,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            fc = tcl.fully_connected(
                z, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = self.residual_block(fc,self.nb_units)

            fc = tcl.fully_connected(
                z, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)   
            fc = tcl.fully_connected(
                z, self.nb_units/2,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc) 

            output = tcl.fully_connected(
                fc, self.output_dim,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator_res(object):#skip connection
    def __init__(self, input_dim, label_dim, output_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:   
        z_latent = z[:,:self.input_dim]    
        z_label = z[:,self.input_dim:]    
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                z, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tf.concat([fc,z_label],axis=1)
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)
            #fc = tf.concat([fc,z_label],axis=1)
            output = tcl.fully_connected(
                fc, self.output_dim,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                #weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator_Bayes(object):#y1,y2 = f(x1,x2) where p(y1|x1,x2) = p(y1|x1) 
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2, name, nb_layers=2,nb_units=256,constrain=False):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.constrain = constrain

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            z1 = z[:,:self.input_dim1]
            z2 = z[:,self.input_dim1:]

            fc1 = tcl.fully_connected(
                z1, self.nb_units,
                activation_fn=tf.identity,
                scope='z1_0' 
                )
            fc1 = leaky_relu(fc1)

            fc2 = tcl.fully_connected(
                z, self.nb_units,
                activation_fn=tf.identity,
                scope='z2_0'
                )
            fc2 = leaky_relu(fc2)     

            for i in range(self.nb_layers-1):
                z = fc1
                fc1 = tcl.fully_connected(
                    fc1, self.nb_units,
                    activation_fn=tf.identity,
                    scope='z1_%d'%(i+1)
                    )
                fc1 = leaky_relu(fc1)

                fc2 = tf.concat([z,fc2],axis=1)
                fc2 = tcl.fully_connected(
                    fc2, self.nb_units,
                    activation_fn=tf.identity,
                    scope='z2_%d'%(i+1)
                    )
                fc2 = leaky_relu(fc2)
            
            output1 = tcl.fully_connected(
                fc1, self.output_dim1,
                activation_fn=tf.identity,
                scope='z1_last'
                )
            fc2 = tf.concat([fc1,fc2],axis=1)
            output2 = tcl.fully_connected(
                fc2, self.output_dim2,
                activation_fn=tf.identity,
                scope='z2_last'
                )
            if self.constrain:
                output2_phi = output2[:,1:2]
                output2_sigma2 = output2[:,2:3]
                output2_nu = output2[:,3:4]
                output2_phi = tf.tanh(output2_phi)
                output2_sigma2 = tf.abs(output2_sigma2)
                #output2_nu = tf.abs(output2_nu)
                output2 = tf.concat([output2[:,0:1],output2_phi,output2_sigma2,output2_nu,output2[:,-2:]],axis=1)              
            return [output1,output2]

    @property
    def vars(self):
        vars_z1 = [var for var in tf.global_variables() if self.name+'/z1' in var.name]
        vars_z2 = [var for var in tf.global_variables() if self.name+'/z2' in var.name]
        all_vars = [var for var in tf.global_variables() if self.name in var.name]
        return [vars_z1,vars_z2,all_vars]

class Generator_PCN(object):#partially connected network, z1<--f1(z1), z2<--f2(z1,z2)
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2, name, nb_layers=2,nb_units=256):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            z1 = z[:,:self.input_dim1]
            z2 = z[:,self.input_dim1:]

            fc1 = tcl.fully_connected(
                z1, self.nb_units,
                activation_fn=tf.identity,
                scope='z1_0' 
                )
            fc1 = leaky_relu(fc1)
            #cross connections
            fc_cross = tcl.fully_connected(
                z1, self.nb_units,
                weights_initializer=tf.zeros_initializer(),
                biases_initializer=None,
                activation_fn=tf.identity,
                scope='zc_0'
                )
            fc_cross = leaky_relu(fc_cross)     

            fc2 = tcl.fully_connected(
                z2, self.nb_units,
                activation_fn=tf.identity,
                scope='z2_0'
                )
            fc2 = leaky_relu(fc2)              
            fc2 = tf.add(fc2,fc_cross)
            #fc2 = tf.concat([fc2,fc_cross],axis=1)
            
            for i in range(self.nb_layers-1):
                z = fc1
                fc1 = tcl.fully_connected(
                    fc1, self.nb_units,
                    activation_fn=tf.identity,
                    scope='z1_%d'%(i+1)
                    )
                fc1 = leaky_relu(fc1)

                #cross connection
                fc_cross = tcl.fully_connected(
                    z, self.nb_units,
                    activation_fn=tf.identity,
                    weights_initializer=tf.zeros_initializer(),
                    biases_initializer=None,
                    scope='zc_%d'%(i+1)
                    )
                fc_cross = leaky_relu(fc_cross)

                fc2 = tcl.fully_connected(
                    fc2, self.nb_units,
                    activation_fn=tf.identity,
                    scope='z2_%d'%(i+1)
                    )
                fc2 = leaky_relu(fc2)
                fc2 = tf.add(fc2,fc_cross)
                #fc2 = tf.concat([fc2,fc_cross],axis=1)

            output1 = tcl.fully_connected(
                fc1, self.output_dim1,
                activation_fn=tf.identity,
                scope='z1_last'
                )
            #cross connection
            output_cross = tcl.fully_connected(
                fc1, self.output_dim2,
                activation_fn=tf.identity,
                weights_initializer=tf.zeros_initializer(),
                scope='zc_last'
                )           
            
            output2 = tcl.fully_connected(
                fc2, self.output_dim2,
                activation_fn=tf.identity,
                scope='z2_last'
                )
            output2 = tf.add(output2,output_cross)        
            return [output1,output2]

    @property
    def vars(self):
        vars_z1 = [var for var in tf.global_variables() if self.name+'/z1' in var.name]
        vars_z2 = [var for var in tf.global_variables() if self.name+'/z2' in var.name]
        vars_zc = [var for var in tf.global_variables() if self.name+'/zc' in var.name]
        return [vars_z1,vars_z2,vars_zc]

class Encoder(object):
    def __init__(self, input_dim, output_dim, feat_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)

            output = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )               
            logits = output[:, self.feat_dim:]
            y = tf.nn.softmax(logits)
            #return output[:, 0:self.feat_dim], y, logits
            return output, y

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder_1dcnn(object):
    def __init__(self, input_dim, output_dim, feat_dim, name):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.name = name

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            x = tf.expand_dims(x, -1)
            fc = tf.layers.conv1d(x,filters=64,kernel_size=10,strides=1,padding='valid')
            print fc
            fc = tf.layers.max_pooling1d(fc,pool_size=5,strides=3,padding='valid')
            print fc
            fc = tf.layers.conv1d(fc,filters=64,kernel_size=10,strides=1,padding='valid')
            print fc
            fc = tf.layers.max_pooling1d(fc,pool_size=5,strides=3,padding='valid')
            print fc
            fc = tf.layers.conv1d(fc,filters=64,kernel_size=10,strides=1,padding='valid')
            print fc
            fc = tf.layers.max_pooling1d(fc,pool_size=5,strides=3,padding='valid')
            print fc
            fc = tf.layers.conv1d(fc,filters=64,kernel_size=10,strides=1,padding='valid')
            print fc
            fc = tf.layers.max_pooling1d(fc,pool_size=5,strides=3,padding='valid')
            print fc
            fc = tf.keras.layers.GlobalAveragePooling1D()(fc)
            print fc
            output = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )               
            logits = output[:, self.feat_dim:]
            y = tf.nn.softmax(logits)
            return output[:, 0:self.feat_dim], y, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class EncoderCouple(object):
    def __init__(self, input_dim1, input_dim2, output_dim, feat_dim, name, nb_layers=2, nb_units=256):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.nb_classes = (output_dim-feat_dim)/2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)

            output = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )               
            logits1 = output[:, -2*self.nb_classes:-self.nb_classes]
            y1 = tf.nn.softmax(logits1)
            logits2 = output[:, -self.nb_classes:]
            y2 = tf.nn.softmax(logits2)
            return output[:, :self.feat_dim], y1, logits1, y2, logits2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Discriminator_img(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256,dataset='mnist'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]

            if self.dataset=="mnist":
                z = tf.reshape(z, [bs, 28, 28, 1])
            elif self.dataset=="cifar10":
                z = tf.reshape(z, [bs, 32, 32, 3])
            conv = tcl.convolution2d(z, 64, [4,4],[2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            #(bs, 14, 14, 32)
            conv = leaky_relu(conv)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, 128, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                #conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
            #(bs, 7, 7, 32)
            #fc = tf.reshape(conv, [bs, -1])
            fc = tcl.flatten(conv)
            #(bs, 1568)
            fc = tcl.fully_connected(
                fc, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None)
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, 1, 
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


#generator for images, G()
class Generator_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',is_training=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.is_training = is_training

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            y = z[:,-10:]
            #yb = tf.reshape(y, shape=[bs, 1, 1, 10])
            fc = tcl.fully_connected(
                z, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = tf.concat([fc, y], 1)

            if self.dataset=='mnist':
                fc = tcl.fully_connected(
                    fc, 7*7*128,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                fc = tf.reshape(fc, tf.stack([bs, 7, 7, 128]))
            elif self.dataset=='cifar10':
                fc = tcl.fully_connected(
                    fc, 8*8*128,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                fc = tf.reshape(fc, tf.stack([bs, 8, 8, 128]))
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = conv_cond_concat(fc,yb)
            conv = tcl.convolution2d_transpose(
                fc, 64, [4,4], [2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #(bs,14,14,64)
            conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            conv = tf.nn.relu(conv)
            if self.dataset=='mnist':
                output = tcl.convolution2d_transpose(
                    conv, 1, [4, 4], [2, 2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.nn.sigmoid
                )
                output = tf.reshape(output, [bs, -1])
            elif self.dataset=='cifar10':
                output = tcl.convolution2d_transpose(
                    conv, 3, [4, 4], [2, 2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.nn.sigmoid
                )
                output = tf.reshape(output, [bs, -1])
            #(0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#encoder for images, H()
class Encoder_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',cond=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.cond = cond

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            if self.dataset=="mnist":
                x = tf.reshape(x, [bs, 28, 28, 1])
            elif self.dataset=="cifar10":
                x = tf.reshape(x, [bs, 32, 32, 3])
            conv = tcl.convolution2d(x,64,[4,4],[2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            conv = leaky_relu(conv)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, self.nb_units, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
            conv = tcl.flatten(conv)
            fc = tcl.fully_connected(conv, 1024, 
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity)
            
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, self.output_dim, 
                activation_fn=tf.identity
                )        
            logits = output[:, -self.nb_classes:]
            y = tf.nn.softmax(logits)
            return output[:, :-self.nb_classes], y, logits        

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#####for pathology imgs#######3

class Discriminator_pathology_img(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256,dataset='pathology'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            z = tf.reshape(z, [bs, 256, 256, 3])
            conv = tcl.convolution2d(z, 32, [4,4],[2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            conv = tcl.max_pool2d(conv,[2,2])
            #(bs, 64, 64, 32)
            conv = leaky_relu(conv)
            for _ in range(2):
                conv = tcl.convolution2d(conv, 64, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                conv = tcl.max_pool2d(conv,[2,2])
                #conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
            #(bs, 4, 4, 64)
            #fc = tf.reshape(conv, [bs, -1])
            fc = tcl.flatten(conv)
            #(bs, 1568)
            fc = tcl.fully_connected(
                fc, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None)
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, 1, 
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


#generator for images, G()
class Generator_pathology_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',is_training=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.is_training = is_training

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            y = z[:,-10:]
            #yb = tf.reshape(y, shape=[bs, 1, 1, 10])
            fc = tcl.fully_connected(
                z, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = tf.concat([fc, y], 1)
            fc = tcl.fully_connected(
                fc, 8*8*128,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = tf.reshape(fc, tf.stack([bs, 8, 8, 128]))
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = conv_cond_concat(fc,yb)
            conv = tcl.convolution2d_transpose(
                fc, 64, [4,4], [2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #(bs,16,16,64)
            conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            conv = tf.nn.relu(conv)
            for _ in range(3):
                conv = tcl.convolution2d_transpose(
                    conv, 64, [4,4], [2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
                conv = tf.nn.relu(conv)
            #(bs,128,128,64)

            output = tcl.convolution2d_transpose(
                conv, 3, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.nn.sigmoid
            )
            output = tf.reshape(output, [bs, -1])
            #(0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Encoder_pathology_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',cond=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.cond = cond

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 256, 256, 3])
            conv = tcl.convolution2d(x,64,[4,4],[2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            conv = leaky_relu(conv)
            conv = tcl.max_pool2d(conv,[2,2])
            #(bs,64,64,64)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, self.nb_units, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
            conv = tcl.max_pool2d(conv,[4,4],stride=4)
            #(bs,8,8,64)
            conv = tcl.flatten(conv)
            fc = tcl.fully_connected(conv, 1024, 
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity)
            
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, self.output_dim, 
                activation_fn=tf.identity
                )        
            logits = output[:, -self.nb_classes:]
            y = tf.nn.softmax(logits)
            return output[:, :-self.nb_classes], y, logits        

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


if __name__=='__main__':
    import numpy as np
    import time
    from tensorflow.python.ops.parallel_for.gradients import jacobian,batch_jacobian
    b=np.random.normal(size=(2,3)).astype('float32')
    a = tf.convert_to_tensor(b)
    b=2*tf.ones_like(a)
    c=a*b 

    one_hot_a = 1-tf.sign(tf.reduce_max(a,axis=1,keepdims=True)-a)
    grad = tf.gradients(one_hot_a, a)[0]
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    sess.run(tf.global_variables_initializer())
    print sess.run([a,b,c])
    sys.exit()
    #W = tf.Variable(tf.zeros([3,5]))
    thred = 20
    wij2 = tf.matmul(W,tf.transpose(W))
    c = tf.constant([1,3], tf.int32)
    wi2 = tf.tile(tf.reshape(tf.reduce_sum(tf.square(W),axis=1),[3,1]),c)
    c_t = tf.constant([3,1], tf.int32)
    wj2 = tf.tile(tf.reshape(tf.reduce_sum(tf.square(W),axis=1),[1,3]),c_t)
    diff_w  =wi2 - 2*wij2 + wj2
    diff_w  = tf.nn.relu(thred - (wi2 - 2*wij2 + wj2))
    #loss_w = (tf.reduce_sum(diff_w)-tf.trace(diff_w))/2
    print wij2, wi2, wj2,diff_w, loss_w

    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    sess.run(tf.global_variables_initializer())
    print sess.run(diff_w)
    sys.exit()
    y1 = tf.placeholder(tf.float32, [3, 100, 1], name='y1')
    y2 = tf.placeholder(tf.float32, [64, 16415], name='y2')
    h_net = Encoder_1dcnn(input_dim=20000, output_dim=32+10, feat_dim=32,name='h_net')
    pre = h_net(y2,reuse=False)

    #(3,4,5)
    print y2, pre
    sys.exit()
    
    l2_loss = tf.reduce_mean((y1-y2)**2)
    A = np.ones((2,2),dtype='float32')
    diag = tf.linalg.tensor_diag_part(tf.matmul(tf.matmul(y1, A),tf.transpose(y2)))
    x_onehot1 = tf.placeholder(tf.float32, [3, 5], name='onehot1')
    x_onehot2 = tf.placeholder(tf.float32, [3, 5], name='onehot2')
    couple_loss = -(tf.linalg.trace(tf.matmul(tf.matmul(y1, A),tf.transpose(y2))))*1.0
    
    x = tf.placeholder(tf.float32, [16, 10], name='x')
    dx_net  = Discriminator(input_dim=10,name='dx_net',nb_layers=2,nb_units=16)
    dx = dx_net(x,reuse=False)
    grad = tf.gradients(dx, x)[0] #(16,10)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))#(16,)
    ddx = tf.square(grad_norm - 1.0)

    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    sess.run(tf.global_variables_initializer())
    a=np.ones((3,2),dtype='float32')
    b=3*np.ones((3,2),dtype='float32')
    result = sess.run(l2_loss,feed_dict={y1:a,y2:b})
    print result
    sys.exit()

    print 1./(2*np.pi)**2 * 0.2 * np.exp(-0.8)
    print 1./((2*np.pi)**2) * 0.2 * 2**2 *np.exp(-0.8)
    x_dim=2
    y_dim=2
    N=2
    sd_y=1.0
    y_points = np.ones((2,2))
    y_points[1] = 2*np.ones(2)
    x_points_ = np.ones((2,2))
    x_points_[1] = 2*np.ones(2)
    y_points__ = np.ones((2,2))
    y_points__[1] = 4*np.ones(2)
    rt_error = np.sum((y_points-y_points__)**2,axis=1)
    #get jocobian matrix with shape (N, y_dim, x_dim)
    #jacob_mat = np.random.normal(size=(N,y_dim,x_dim))
    jacob_mat = np.zeros((2,2,2))
    jacob_mat[0] = 2*np.eye(2)
    jacob_mat[1] = 4*np.eye(2)
    jacob_mat_transpose = jacob_mat.transpose((0,2,1))
    #matrix A = G^T(x_)*G(x_) with shape (N, x_dim, x_dim)
    A = map(lambda x, y: np.dot(x,y), jacob_mat_transpose, jacob_mat)
    #vector b = grad_^T(G(x_))*(y-y__) with shape (N, x_dim)
    b = map(lambda x, y: np.dot(x,y), jacob_mat_transpose, y_points-y_points__)
    #covariant matrix in constructed multivariate Gaussian with shape (N, x_dim, x_dim)
    Sigma = map(lambda x: np.linalg.inv(np.eye(x_dim)+x/sd_y**2),A)
    Sigma_inv = map(lambda x: np.eye(x_dim)+x/sd_y**2,A)
    #mean vector in constructed multivariate Gaussian with shape (N, x_dim)
    mu = map(lambda x,y,z: x.dot(y/sd_y**2-z),Sigma,b,x_points_)
    #constant c(y) in the integral c(y) = l2_norm(x_)^2 + l2_norm(y-y__)^2/sigma**2-mu^T*Sigma*mu
    c_y = map(lambda x,y,z,w: np.sum(x**2)+y/sd_y**2-z.T.dot(w).dot(z), x_points_, rt_error, mu, Sigma_inv)
    py_est = map(lambda x,y: 1./(np.sqrt(2*np.pi)*sd_y)**y_dim * np.sqrt(np.linalg.det(x)) * np.exp(-0.5*y), Sigma, c_y)
    print len(py_est),py_est
    print rt_error[0]
    print A[0]
    print b[0]
    print mu[0]
    print Sigma[0]
    print Sigma_inv[0]
    print c_y[0]
    sys.exit()
    g_net = Generator_resnet(input_dim=5,output_dim = 10,name='g_net',nb_layers=3,nb_units=10)
    #g_net = Generator_PCN(3,2,10,2,'g_net',nb_layers=3,nb_units=64)
    x = tf.placeholder(tf.float32, [1, 2], name='x')
    #y_ = g_net(x,reuse=False)
    y_  = x**2
    t = tf.reduce_mean((x - y_)**2)
    J = jacobian(y_,x)
    J2 = batch_jacobian(y_,x)
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    sess.run(tf.global_variables_initializer())
    N=2
    a=np.array([[1,2,3,4,5,6]])
    b=np.tile(a,(N,1)).astype('float32')
    t_ = sess.run(t,feed_dict={x:np.array([[2,3]])})
    print t_
    sys.exit()
    #print len(g_net.vars[0]),g_net.vars[0]
    #print len(g_net.vars[1]),g_net.vars[1]
    #print len(g_net.vars[2]),g_net.vars[2]
    g_net_vars_z1 = g_net.vars[0]
    g_net_vars_z2 = g_net.vars[1]
    g_net_vars_zc = g_net.vars[2]
    print len(g_net_vars_z2),len(g_net_vars_zc)
    w_pretrain = sess.run(g_net_vars_z2)
    loss_w = tf.add_n([2 * tf.nn.l2_loss(v[0]-v[1]) for v in zip(g_net_vars_z2,w_pretrain)])
    loss_c = tf.add_n([2 * tf.nn.l2_loss(v) for v in g_net_vars_zc])
    adam_g = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5, beta2=0.9)
    g_optim = adam_g.minimize(loss_w+loss_c, var_list=g_net_vars_z2+g_net_vars_zc)
    sess.run(tf.variables_initializer(adam_g.variables()))
    #print len(g_net.vars[0]),g_net.vars[0]
    #print len(g_net.vars[1]),g_net.vars[1]
    #print len(g_net.vars[2]),g_net.vars[2]
    sess.run(tf.variables_initializer(adam_g.variables()))
    w_pretrain = sess.run(g_net_vars_z2)
    print len(g_net_vars_z2),len(g_net_vars_zc)
    loss_w = tf.add_n([2 * tf.nn.l2_loss(v[0]-v[1]) for v in zip(g_net_vars_z2,w_pretrain)])
    loss_c = tf.add_n([2 * tf.nn.l2_loss(v) for v in g_net_vars_zc])
    adam_g = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5, beta2=0.9)
    g_optim = adam_g.minimize(loss_w+loss_c, var_list=g_net_vars_z2+g_net_vars_zc)
    sess.run(tf.variables_initializer(adam_g.variables()))
    # print len(g_net.vars[0]),g_net.vars[0]
    # print len(g_net.vars[1]),g_net.vars[1]
    # print len(g_net.vars[2]),g_net.vars[2]
    sess.run(tf.variables_initializer(adam_g.variables()))
    print len(g_net_vars_z2),len(g_net_vars_zc)
    sys.exit()
    c = tf.add_n([2 * tf.nn.l2_loss(v) for v in g_net.vars[1]])
    a = [sess.run(v) for v in g_net.vars[1]]
    print len(g_net.vars[1])
    sess.run(tf.global_variables_initializer())
    print len(g_net.vars[1])
    b = tf.reduce_mean([ tf.nn.l2_loss(v[0]-v[1]) for v in zip(g_net.vars[1],a)])
    print sess.run(b)
    print sess.run(c)