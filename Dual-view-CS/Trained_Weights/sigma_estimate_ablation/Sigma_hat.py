import os
from math import log10
from torch import nn
import torch.optim as optim
import torch.utils.data
## import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
import time
import scipy.io as sio
import scipy
from data_utils import *
#from config import config
from models.SigCNN_pt import SigCNNhalf
from models.SigCNN_all import SigCNN

import array

def sigma(noisy):
    path = 'D:\Chenzan\CS_image\Trained_Weights\sigma_estimate_ablation\modelSigALL.pth'
    # noisy = np.ones((1, 256 * 256))
    imsize = 256
    noisy = np.array(noisy)
    noisy = torch.from_numpy(noisy)
    noisy = noisy.cuda()
    noisy = noisy / (255. / 2.)
    noisy = noisy - 1.
    noisy = torch.reshape(noisy, (1, 1, imsize, imsize))
    noisy = noisy.float()
    net = SigCNN(imsize).cuda()
    # net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(path))
    net.eval()
    # for k, v in net.named_parameters():
    #     v.requires_grad = False
    x_hat = net(noisy)
    x_hat = x_hat.double()
    # x_hat = x_hat * 255
    # x_hat = torch.reshape(x_hat, (imsize * imsize, 1))
    x_hat = x_hat.cpu()
    x_hat = x_hat.detach().numpy()
    x_hat = array.array('d', x_hat)

    return x_hat

def sigmahalf(noisy):
    path = 'D:\Chenzan\CS_image\Trained_Weights\sigma_estimate_ablation\modelSighalf.pth'
    # noisy = np.ones((1, 256 * 256))
    imsize = 256
    noisy = np.array(noisy)
    noisy = torch.from_numpy(noisy)
    noisy = noisy.cuda()
    noisy = noisy / (255. / 2.)
    noisy = noisy - 1.
    noisy = torch.reshape(noisy, (1, 1, imsize, imsize))
    noisy = noisy.float()
    net = SigCNNhalf(imsize).cuda()
    # net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(path))
    net.eval()

    x_hat = net(noisy)
    x_hat = x_hat.double()

    x_hat = x_hat.cpu()
    x_hat = x_hat.detach().numpy()
    x_hat = array.array('d', x_hat)

    return x_hat

