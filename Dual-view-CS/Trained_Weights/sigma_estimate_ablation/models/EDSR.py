#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py

def EDSR(t_image, is_train=False, reuse=False):   #看参照模型图示来看此网络结构
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    

    with tf.variable_scope("EDSR", reuse=reuse):
        n = InputLayer(t_image, name='input')                       #将输入转化为layer结构，本身tensor数据存入了layer中的outputs
        n_input = n
        n = Conv2d(n, 64, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv1')  #卷积层操作，参数依次为输入层类，输出通道数，卷积核大小，步长，激活函数选择，padding以及权重初始化

        # 9 residual blocks
        for i in range(9):
            with tf.variable_scope("RB%s"%i, reuse=False):
                nn = Conv2d(n, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1')
                nn = Conv2d(nn, 64, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv2')
                nn = ElementwiseLayer([n, nn], tf.add, name='add')     #元素相加操作
                n = nn
        # 9 residual blacks end

        n = Conv2d(n,  1, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv2')
        n = ElementwiseLayer([n, n_input], tf.add, name='add')
        
        return n
