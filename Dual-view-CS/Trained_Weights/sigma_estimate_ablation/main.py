#! /usr/bin/python
# -*- coding: utf8 -*-

import random
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import scipy.io as sio
import tensorflow as tf
import tensorlayer as tl
from models.SigCNN import SigCNN                       #导入模型SigCNN
from utilities.utils import *                              #导入utils中所有函数
from config import config, log_config
import os
import time
import random
from functools import reduce
###====================== HYPER-PARAMETERS ===========================###
## Adam
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    #
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          #设定显卡，若有多张卡需要选定时进行设置，模型显卡选择为0
batch_size = config.TRAIN.batch_size              #从config文件中导入设置的训练batch数目，学习率，优化器参数
lr_init = config.TRAIN.lr_init                    #学习率
beta1 = config.TRAIN.beta1                        #优化器参数
n_epoch_half = config.TRAIN.n_epoch_half
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init          #初始化训练epoch数目

def train():
    ## create folders to save result images and trained model
 
    checkpoint_dir = "checkpoints"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)      #检查目录checkpiont是否存在，若不存在，则创建该目录
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))    #将文件夹中所有png文件名存入变量，形成一个list
    eval_hr_img_list = sorted(tl.files.load_file_list(path=config.EVAL.hr_img_path, regx='.*.bmp', printable=False))    #将文件夹中所有png文件名存入变量，形成一个list
	
    ####========================== tensorflow图构建 ==========================###
    Input_image = tf.placeholder('float32', [batch_size, 192, 192, 1], name='Input_images')          #设定一个占位变量，tensor格式，定义它的数据类型，尺寸大小以及名称，占位变量一般用来导入数据
    Target_sigma = tf.placeholder('float32', [batch_size, 1], name='Target_sigma')        #设定一个占位变量，tensor格式，定义它的数据类型，尺寸大小以及名称
    nn_net = SigCNN(Input_image, is_train=True, reuse=False)

    ####========================== DEFINE TRAIN OPS ==========================###	
    mse_loss = tl.cost.mean_squared_error(nn_net.outputs, Target_sigma, is_mean=True)   #定义训练损失函数，这里的nn_net为tensorlayer结构体，真正的输出为其中的outputs变量。
    nn_vars = tl.layers.get_variables_with_name('SigCNN', True, True)                    #获取所有名称中包含’net‘的参数，将其保存入nn_vars

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)                                     #初始化学习率，学习率是不经过训练的，所以设定为false

    ## Pretrain
    nn_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=nn_vars)    #初始化优化器，设定其学习率，bata值，优化目标以及需要更新的参数
	
	
	####========================== 正式训练 ==========================###	
	
    #config1 = tf.ConfigProto()
    #config1.gpu_options.per_process_gpu_memory_fraction = 0.49      
    #sess = tf.Session(config=config1)           #设定显卡占用率，若默认，tensorflow将占用显卡所有显存，看情况应用
    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)  #初始化网络所有参数
    #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/SigCNN.npz', network=nn_net)   #若需要在某训练好的模型上重新训练，在此导入模型。

    sess.run(tf.assign(lr_v, (lr_init)))         #学习率赋值
    print(" ** fixed learning rate: %f (for init G)" % (lr_init))
    for epoch in range(0,n_epoch_init+1):        #整个训练epoch大循环
        random.shuffle(train_hr_img_list)        #打乱读入的所有文件名顺序
        epoch_time = time.time()                 #时间获取，计时用，若无需计时查看，则可删除相关部分代码
        total_mse_loss, n_iter = 0, 0            #统计loss变量
        if epoch != 0 and (epoch % n_epoch_half == 0):     #设置学习率更新，此模型与对应的训练集中，按经验划分每20epoch学习率下降为原学习率的三分之一
            sess.run(tf.assign(lr_v, lr_init/(2**(epoch//n_epoch_half))))
            log = " ** new learning rate: %f (for GAN)" % (lr_init/(2**(epoch//n_epoch_half)))
            print(log)
        for idx in range(0, len(train_hr_img_list)//batch_size*batch_size, batch_size):    #单个epoch中batch_size训练小循环
            step_time = time.time()
            try:                    #try与except的应用，避免训练过程中出现的一些不可预知的错误，比如图像大小不合适等出现的中断，若调节合适，可去除try语句
                train_hr_imgs = tl.visualize.read_images(train_hr_img_list[idx:idx + batch_size], path=config.TRAIN.hr_img_path, n_threads = 8, printable=False)
                train_hr_imgs = tl.prepro.threading_data(train_hr_imgs, fn=crop, wrg=192, hrg=192, is_random=True)             #读取的数据进行随机划块               
                Th_out = tl.prepro.threading_data(train_hr_imgs, fn=add_noise, sl=config.noise_sl, sh=config.noise_sh, wrg=192, hrg=192, with_noise = True, normal_way = 0)             #读取的数据进行加噪，噪声方差范围90~100，函数见utils
                n_imgs = np.array(Th_out[:,0].tolist())
                b_sigma = np.array(Th_out[:,1].tolist())
            except:
                continue
            errM, _ = sess.run([mse_loss, nn_optim_init], feed_dict={Input_image: n_imgs, Target_sigma: b_sigma})       #将真实图像预噪声图像分别送入占位变量中，sess.run之后按图中逻辑运行，其中[]中代表需要得到的变量，sess.run的返回值将是[]中的变量，其次更新网络参数必须进行sess.run优化器
            total_mse_loss += errM
            n_iter += 1
            if n_iter%10 ==0:
                try:                    #try与except的应用，避免训练过程中出现的一些不可预知的错误，比如图像大小不合适等出现的中断，若调节合适，可去除try语句
                    eval_hr_imgs = tl.visualize.read_images(eval_hr_img_list[0:batch_size], path=config.EVAL.hr_img_path, n_threads = 8, printable=False)
                    eval_hr_imgs = tl.prepro.threading_data(eval_hr_imgs, fn=crop, wrg=192, hrg=192, is_random=True)             #读取的数据进行随机划块               
                    eval_out = tl.prepro.threading_data(eval_hr_imgs, fn=add_noise, sl=config.noise_sl, sh=config.noise_sh, wrg=192, hrg=192, with_noise = True, normal_way = 0)             #读取的数据进行加噪，噪声方差范围90~100，函数见utils
                    eval_imgs = np.array(eval_out[:,0].tolist())
                    eval_sigma = np.array(eval_out[:,1].tolist())
                except:
                    continue
                outputs = sess.run([nn_net.outputs], feed_dict={Input_image: eval_imgs}) 
                print(outputs[0][1], eval_sigma[1])
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, total_mse_loss / n_iter))  #打印相关计算的mse
                

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)#打印均值mse
		
        tl.files.save_npz(nn_net.all_params, name=checkpoint_dir + '/SigCNN.npz', sess=sess)#保存模型，这里是每个epoch保存一次，若名称固定则会将上一次模型覆盖。根据需要自己调整。


    ###========================= train GAN (SRGAN) =========================###

def evaluate_old():             #测试函数
    checkpoint_dir = "checkpoints"
    eval_hr_img_list = sorted(tl.files.load_file_list(path=config.EVAL.hr_img_path, regx='.*.tif', printable=False)) 
    ###========================== DEFINE MODEL ============================###
    Input_image = tf.placeholder('float32', [1,192,192,1], name='input_image') # the old version of TL need to specify the image size
    nn_net = SigCNN(Input_image, is_train=False, reuse=False) 
    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/SigCNN.npz'.format(tl.global_flag['mode']), network=nn_net)#导入模型

    for i in range(len(eval_hr_img_list)):
        eval_hr_imgs = tl.visualize.read_images(eval_hr_img_list[i:i+1], path=config.EVAL.hr_img_path, n_threads = 8, printable=False)
        eval_hr_imgs = tl.prepro.threading_data(eval_hr_imgs, fn=crop, wrg=192, hrg=192, is_random=True)             #读取的数据进行随机划块               
        eval_out = tl.prepro.threading_data(eval_hr_imgs, fn=add_noise, sl=config.noise_sl, sh=config.noise_sh, wrg=192, hrg=192, with_noise = True, normal_way = 0)             #读取的数据进行加噪，噪声方差范围90~100，函数见utils
        eval_imgs = np.array(eval_out[:,0].tolist())
        eval_sigma = np.array(eval_out[:,1].tolist())
        outputs = sess.run([nn_net.outputs], {Input_image:eval_imgs})
        print(outputs[0][0], eval_sigma[0])


def evaluate():             #测试函数
    checkpoint_dir = "checkpoints"
    eval_hr_img_list = sorted(tl.files.load_file_list(path=config.EVAL.hr_img_path, regx='.*.mat', printable=False))
    ###========================== DEFINE MODEL ============================###
    t_image = tf.placeholder('float32', [1,256,256,1], name='input_image') # the old version of TL need to specify the image size
    nn_net, end_points = SigCNN(t_image, is_train=False, reuse=False) 
    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/SigCNN.npz'.format(tl.global_flag['mode']), network=nn_net)#导入模型


    for i in range(len(eval_hr_img_list)):
        #eval_hr_imgs = tl.prepro.threading_data(eval_hr_img_list[i:i + 1], fn=get_imgs_fn1, path=config.EVAL.hr_img_path)
        eval_hr_imgs = get_imgs_fn1(eval_hr_img_list[i:i + 1], config.EVAL.hr_img_path)
        tinput = np.array(eval_hr_imgs)
        out1 = tinput
        out1 = out1.astype(np.float)
        tinput = tinput /(255./2.)    
        tinput = tinput - 1.0          #由于在训练过程中将数据归到-1-1之间，在此入网络前也应做相关操作
        outputs, outputs_temp = sess.run([nn_net.outputs, end_points], {t_image:tinput})
        sio.savemat('test.mat',{'data':outputs_temp})
        print(outputs[0]*255/2)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')         #python main.py --mode srgan  python main.py --mode evaluate 默认参数--mode 为 srgan 

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
