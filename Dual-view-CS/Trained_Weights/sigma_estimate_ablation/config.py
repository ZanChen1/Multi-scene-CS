from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.EVAL = edict()

## Adam
config.batch_size = 24
config.lr_init = 1e-4
config.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
## train set location
config.TRAIN.hr_img_path = 'E:\Task\RCAN_denoising_2\Training_Data/'
## eval set location
# config.EVAL.hr_img_path = '../../Test_Images/Waterloo_crop/'
##  train half period
config.TRAIN.n_epoch_half = 5

config.VALID = edict()
## test set location

config.noise_sl = 0
config.noise_sh = 100


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
