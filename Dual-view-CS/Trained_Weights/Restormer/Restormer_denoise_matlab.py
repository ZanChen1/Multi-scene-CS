import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.restormer_arch import Restormer
import array
import yaml
import gc  # 引入垃圾回收模块

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def denoiser(img_L, img_weight, img_height, noise_level_model):
    sigma_test = noise_level_model
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 初始化变量，防止 finally 中报错
    model_restoration = None
    input_ = None
    img_E = None

    try:
        # -----------------------------------------------model parameter setting
        parser = argparse.ArgumentParser(description='Gasussian Grayscale Denoising using Restormer')
        default_weights = os.path.join(current_dir, 'pretrained_models', 'gaussian_gray_denoising')
        parser.add_argument('--weights', default=default_weights, type=str, help='Path to weights')
        parser.add_argument('--model_type', default='blind', type=str,
                            help='blind: single model to handle various noise levels. non_blind: separate model for each noise level.')

        # 避免 MATLAB 参数冲突
        args = parser.parse_args([])

        ####### Load yaml #######
        if args.model_type == 'blind':
            yaml_file = os.path.join(current_dir, 'Options', 'GaussianGrayDenoising_Restormer.yml')
        else:
            yaml_file = os.path.join(current_dir, 'Options', f'GaussianGrayDenoising_RestormerSigma{args.sigmas}.yml')

        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
        s = x['network_g'].pop('type')

        # --------------------------------------------------model injection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_restoration = Restormer(**x['network_g'])

        if args.model_type == 'blind':
            weights_path = args.weights + '_blind.pth'
        else:
            weights_path = args.weights + '_sigma' + str(sigma_test) + '.pth'

        checkpoint = torch.load(weights_path)
        model_restoration.load_state_dict(checkpoint['params'])
        model_restoration = model_restoration.to(device)
        model_restoration = nn.DataParallel(model_restoration)
        model_restoration.eval()

        # --------------------------------------------------data preprocess
        img_L = np.array(img_L)
        img_L = np.reshape(img_L, (int(img_weight), int(img_height)))
        if img_L.ndim == 2:
            img_L = np.expand_dims(img_L, axis=2)

        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
        input_ = img_L.to(device)

        ##########################
        factor = 8
        # --------------------------------------------------model evaluation
        with torch.no_grad():
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            # 推理
            img_E_tensor = model_restoration(input_)

            # Unpad images to original dimensions
            img_E_tensor = img_E_tensor[:, :, :h, :w]

            # 转回 CPU numpy
            img_E = torch.clamp(img_E_tensor, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            # 准备返回数据
            img_E = img_E * 255.0
            img_E = np.reshape(img_E, (int(img_weight) * int(img_height), 1))
            img_E = array.array('d', img_E)

    except Exception as e:
        print(f"Python Error: {e}")
        # 如果出错，返回空或者抛出异常，但在抛出前会执行 finally
        raise e

    finally:
        # ==================================================
        # 无论成功还是报错，这里都会执行，确保显存释放！！！
        # ==================================================
        if 'model_restoration' in locals(): del model_restoration
        if 'input_' in locals(): del input_
        if 'img_E_tensor' in locals(): del img_E_tensor

        gc.collect()
        torch.cuda.empty_cache()
        # print("GPU Memory Cleared.")

    return img_E