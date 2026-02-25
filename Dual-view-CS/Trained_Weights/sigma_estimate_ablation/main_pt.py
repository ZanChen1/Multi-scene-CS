import os
from math import log10
from torch import nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
import time
import scipy.io as sio
import scipy
from data_utils import *
from models.SigCNN_all import SigCNN


import scipy.io as scio
#仅含训练过程，测试仅需导入模型与参量即可
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"              #选择显卡


batch_size = 128              #设定batch大小
lr_init =5*1e-5               #设定初始学习率
n_epoch_init =100             #设定学习总epoch数
src_path =  'E:\Task\RCAN_denoising_2\Training_Data/'                #设定训练集路径
noise_sl = 0                   #设定噪声电平
noise_sh = 100
patch_size = 256            #设定训练图像划分小块
half_epoch =10              #设定学习率减半epoch数
train_hr_img_list = sorted(load_file_list(path=src_path, regx='.*.png', printable=False))
netG = SigCNN(patch_size)                 #网络模型，测试时，采取netG = RCAN.eval()
mse_criterion = nn.MSELoss()   #loss设定
netG.cuda()                    #模型载入显卡

# if args.finetune:#微调模型，包括继续训练及不同结构微调
#     if args.diff_model:
#         pretrained_dict=torch.load(args.per_model)
#         netG_dict=netG.state_dict()
#         new_dict = {k: v for k, v in pretrained_dict.items() if k in netG_dict}
#         netG_dict.update(new_dict)
#         netG.load_state_dict(netG_dict)
    # else:
# netG.load_state_dict(torch.load('modelSig.pth'))#若有预训练模型则可导入

# for name,parameters in netG.named_parameters():
#     parameters.requires_grad = True
#     print(name)
mse_criterion.cuda()        #loss函数载入显卡

optimizerG = optim.Adam(netG.parameters(),betas=(0.9, 0.999),lr=lr_init)#设定优化器需要更新的参数与学习率

for epoch in range(0,n_epoch_init+1):        #epoch训练大循环
    #print(epoch
    netG.train()                             #训练设定
    random.shuffle(train_hr_img_list)        #打乱训练文件顺序
    epoch_time = time.time()                 #计时
    total_mse_loss1,total_mse_loss2,n_iter = 0,0,0
    if epoch > 2 and (epoch % half_epoch == 0):
        lr = lr_init/(2**((epoch-0)//half_epoch))
        log = " ** new learning rate: %f " % (lr_init/(2**((epoch-0)//half_epoch)))
        print(log)
        for param_group in optimizerG.param_groups:
            param_group['lr'] = lr                            #每过10个epoch更新一次学习率（epoch>20），学习率缩减三倍
    for idx in range(0,len(train_hr_img_list)//batch_size*batch_size, batch_size):      #epoch中batchsize小训练循环
            step_time = time.time()
            try:
                train_hr_imgs = threading_data(train_hr_img_list[idx:idx + batch_size], fn=get_imgs_fn,path=src_path)#读取图片数据
                real_sigma = np.random.randint(noise_sl, noise_sh) + np.random.rand(1)
                train_batch_data= threading_data(train_hr_imgs,fn=crop_sub_imgs_fn,sig = real_sigma,is_random=True,p_size = patch_size)#图片数据截取与加噪
                # [n_imgs_384,sigma]  = np.split(train_batch_data,2,axis=1)#噪声图与原图分离

                #print(b_imgs_384.shape)
                # real_img = Variable(torch.from_numpy(b_imgs_384))  #将numpy转为tensor格式
                z = Variable(torch.from_numpy(train_batch_data))
                if torch.cuda.is_available():
                    # real_img = real_img.cuda()   #数据入显卡
                    z = z.cuda()
                netG.zero_grad()    #网络梯度清零，每次更新前需要清零操作
                real_sigma = np.expand_dims(real_sigma, 0).repeat(128, axis=0)
                real_sigma = torch.Tensor(real_sigma)
                real_sigma = real_sigma.cuda()
                fake_sigma = netG(z)  #反馈网络结果
                g_loss = mse_criterion(fake_sigma, real_sigma)  #计算loss
                g_loss.backward()  #反馈loss计算梯度
                optimizerG.step()   #更新参数
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, n_iter, time.time() - step_time, g_loss.item()))#打印loss
                total_mse_loss1 += g_loss.item()
                n_iter += 1
            except:
                continue
    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f,size [%3d,%3d]" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss1 / (1+n_iter),patch_size,patch_size)#打印loss
    print(log)
    with open( 'loss.txt', 'a+') as f:
        f.writelines('{0}\n'.format(total_mse_loss1 / (1+n_iter)))
    zz = str(epoch)+'modelSigALL.pth'
    # zz = 'AAFall400_'+str(args.mode)+'_sigma_'+str(args.sigma)+'.pth'
    #print(zz)
    torch.save(netG.state_dict(),zz )#保存模型