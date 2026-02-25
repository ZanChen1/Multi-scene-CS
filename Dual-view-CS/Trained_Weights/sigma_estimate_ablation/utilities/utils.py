import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path
import scipy.io as sio
import scipy
import numpy as np

def get_imgs_fn(file_name, path):#tensorflow输入为B H W C
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    img = scipy.misc.imread(path + file_name, mode='L')  #按灰度图进行读图操作
    img = np.array(img)
    w,h = img.shape
    if w<h:
        img = img.reshape([h,w])
    return img

def get_imgs_fn1(file_name, path):#tensorflow输入为B H W C
    """ Input an image path and name, return an image array """
    data1 = sio.loadmat(path + file_name[0])
    data1 = data1['data'] 
    if len(data1.shape)>2:
        data1 = data1.reshape((data1.shape[0],data1.shape[1],data1.shape[2]))
    else:
        data1 = data1.reshape((1,data1.shape[0],data1.shape[1],1))
    data1 = data1.astype(float)
    return data1



def add_noise(x,sl=0,sh=5,wrg=192, hrg=192, with_noise = False, normal_way = 0):

    if with_noise:
        sigma = np.random.randint(sl,sh-1)+np.random.rand(1)  #产生范围内噪声标准差
        x = x.astype(np.float64) + np.random.normal(0,sigma,[wrg,hrg])#产生噪声标准差的高斯噪声

    if normal_way == 1:
        x = x.reshape([wrg,hrg,1])
        x = x / 255.
        x = x.astype(np.float32)
        
    else:
        x = x.reshape([wrg,hrg,1])
        x = x / (255. / 2.)-1.
        x = x.astype(np.float32)
        sigma = sigma/(255/2)


    return x, sigma





    

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x
