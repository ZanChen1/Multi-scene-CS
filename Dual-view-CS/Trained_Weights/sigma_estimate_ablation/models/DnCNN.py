#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


def DnCNN(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    
    with tf.variable_scope("DnCNN", reuse=reuse) as vs:

        n = InputLayer(t_image, name='input')
        n_input = n
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv1')
        #temp1=n
        # B residual blocks
        for i in range(18):
            with tf.variable_scope("B%s"%i, reuse=False):
                n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv')
                n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='bnorm')

        n = Conv2d(n,  1, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv2')
        n = ElementwiseLayer([n, n_input], tf.add, name='add')

        return n