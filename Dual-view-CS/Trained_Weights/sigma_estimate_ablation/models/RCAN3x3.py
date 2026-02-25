import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

n_resgroups = 2
w_init = tf.random_normal_initializer(stddev=0.02)
b_init = tf.constant_initializer(value=0.0)
g_init = tf.random_normal_initializer(1., 0.02)


def RCAN(t_image, is_train=False, reuse=False):     # multi level non-useful
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """


    with tf.variable_scope("RCAN", reuse=reuse):

        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv1')
        nn = n

        for i in range(n_resgroups):
            with tf.variable_scope("RG%s"%(i)):
                nn = ResidualGroup(nn)

        nn = Conv2d(nn,  64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv2')
        n = ElementwiseLayer([n, nn], tf.add, name='add')
        n = Conv2d(n,  1, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv3')

        return n



def ResidualGroup(n):

    nn = n
    for i in range(8):
        if i < 4:
            rat = i+1
        else:
            rat = 8-i
        #rat = 1
        with tf.variable_scope("RCAB%s"%(i)):
            nn = RCAB(nn, rat)

    nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv')
    n = ElementwiseLayer([n, nn], tf.add, name='add')

    return n


def RCAB(n, rat):
    
    nn = n
    nn = AtrousConv2dLayer(nn, n_filter = 64, filter_size = (3, 3), rate = rat, act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1')
    nn = AtrousConv2dLayer(nn, n_filter = 64, filter_size = (3, 3), rate = rat, act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv2') #tf.identity means no act
    with tf.variable_scope("CA"):
        nn = CALayer(nn)
    n = ElementwiseLayer([n, nn], tf.add, name='add')  

    return n     


def CALayer(n):

    nn = n
    nn = GlobalMeanPool2d(nn, name = 'reduce_mean')
    nn.outputs = tf.reshape(nn.outputs,[-1,1,1,64])
    nn = Conv2d(nn, 4, (1, 1), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1')
    nn = Conv2d(nn, 64, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init, b_init=b_init, name='conv2')
    n = ElementwiseLayer([n, nn], tf.multiply, name='multiply')
    
    return n  

