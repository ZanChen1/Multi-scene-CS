import os
import torch
from MWCNN_Nested_2group import MWCNN
import cv2
import numpy as np
import array

# ---------- path helpers ----------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def _wpath(*parts):
    """Build absolute path relative to this file."""
    return os.path.join(_THIS_DIR, *parts)

def _needs_dataparallel(weight_path: str) -> bool:
    """
    Keep original behavior: for a few specific weights, use DataParallel.
    Original code compared absolute paths; here we compare basenames instead.
    """
    name = os.path.basename(weight_path)
    return name in {
        "MWCNN_90_100.pth",
        "MWCNN_95_100.pth",
        "MWCNN_90_95.pth",
    }

# ---------- original functions with minimal edits ----------

def denoise_MWCNN(noisy, path):
    if _needs_dataparallel(path):
        net = MWCNN()
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(path))
        net.eval()
        for k, v in net.named_parameters():
            v.requires_grad = False
        x_hat = net(noisy)
    else:
        net = MWCNN()
        net.load_state_dict(torch.load(path))
        net.cuda()
        net.eval()
        for k, v in net.named_parameters():
            v.requires_grad = False
        x_hat = net(noisy)
    return x_hat


def denoise17(noisy, highth, width, sigma_hat):
    mwcnn_path = _wpath("model")

    if sigma_hat > 500:
        path = os.path.join(mwcnn_path, 'MWCNN_500_1000.pth')
    elif sigma_hat > 300:
        path = os.path.join(mwcnn_path, 'MWCNN_300_500.pth')
    elif sigma_hat > 150:
        path = os.path.join(mwcnn_path, 'MWCNN_150_300.pth')
    elif sigma_hat > 125:
        path = os.path.join(mwcnn_path, 'MWCNN_125_150.pth')
    elif sigma_hat > 100:
        path = os.path.join(mwcnn_path, 'MWCNN_100_125.pth')
    elif sigma_hat > 90:
        path = os.path.join(mwcnn_path, 'MWCNN_90_100.pth')
    elif sigma_hat > 80:
        path = os.path.join(mwcnn_path, 'MWCNN_80_90.pth')
    elif sigma_hat > 70:
        path = os.path.join(mwcnn_path, 'MWCNN_70_80.pth')
    elif sigma_hat > 60:
        path = os.path.join(mwcnn_path, 'MWCNN_60_70.pth')
    elif sigma_hat > 50:
        path = os.path.join(mwcnn_path, 'MWCNN_50_60.pth')
    elif sigma_hat > 40:
        path = os.path.join(mwcnn_path, 'MWCNN_40_50.pth')
    elif sigma_hat > 30:
        path = os.path.join(mwcnn_path, 'MWCNN_30_40.pth')
    elif sigma_hat > 20:
        path = os.path.join(mwcnn_path, 'MWCNN_20_30.pth')
    elif sigma_hat > 15:
        path = os.path.join(mwcnn_path, 'MWCNN_15_20.pth')
    elif sigma_hat > 10:
        path = os.path.join(mwcnn_path, 'MWCNN_10_15.pth')
    elif sigma_hat > 5:
        path = os.path.join(mwcnn_path, 'MWCNN_5_10.pth')
    else:
        path = os.path.join(mwcnn_path, 'MWCNN_0_5.pth')

    noisy = np.array(noisy)
    noisy = torch.from_numpy(noisy)
    noisy = noisy.cuda()
    noisy = noisy / 255
    noisy = torch.reshape(noisy, (1, 1, int(highth), int(width)))
    noisy = noisy.float()
    x_hat = denoise_MWCNN(noisy, path)
    x_hat = x_hat.double()
    x_hat = x_hat * 255
    x_hat = torch.reshape(x_hat, (int(highth) * int(width), 1))

    x_hat = x_hat.cpu().numpy()
    x_hat = array.array('d', x_hat)
    return x_hat


def denoise_MWCNN_24(noisy, path):
    if _needs_dataparallel(path):
        net = MWCNN()
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(path))
        net.eval()
        for k, v in net.named_parameters():
            v.requires_grad = False
        x_hat = net(noisy)
    else:
        net = MWCNN()
        net.load_state_dict(torch.load(path))
        net.cuda()
        net.eval()
        for k, v in net.named_parameters():
            v.requires_grad = False
        x_hat = net(noisy)
    return x_hat


def denoise24(noisy, highth, width, sigma_hat):
    mwcnn_path = _wpath("model_packages")

    if sigma_hat > 500:
        path = os.path.join(mwcnn_path, 'MWCNN_500_1000.pth')
    elif sigma_hat > 300:
        path = os.path.join(mwcnn_path, 'MWCNN_300_500.pth')
    elif sigma_hat > 150:
        path = os.path.join(mwcnn_path, 'MWCNN_150_300.pth')
    elif sigma_hat > 125:
        path = os.path.join(mwcnn_path, 'MWCNN_125_150.pth')
    elif sigma_hat > 100:
        path = os.path.join(mwcnn_path, 'MWCNN_100_125_newdata_24.pth')
    elif sigma_hat > 95:
        path = os.path.join(mwcnn_path, 'MWCNN_90_100.pth')
    elif sigma_hat > 90:
        path = os.path.join(mwcnn_path, 'MWCNN_90_100.pth')
    elif sigma_hat > 85:
        path = os.path.join(mwcnn_path, 'MWCNN_80_90.pth')
    elif sigma_hat > 80:
        path = os.path.join(mwcnn_path, 'MWCNN_80_90.pth')
    elif sigma_hat > 75:
        path = os.path.join(mwcnn_path, 'MWCNN_75_80.pth')
    elif sigma_hat > 70:
        path = os.path.join(mwcnn_path, 'MWCNN_70_80.pth')
    elif sigma_hat > 65:
        path = os.path.join(mwcnn_path, 'MWCNN_65_70.pth')
    elif sigma_hat > 60:
        path = os.path.join(mwcnn_path, 'MWCNN_60_70.pth')
    elif sigma_hat > 55:
        path = os.path.join(mwcnn_path, 'MWCNN_55_60.pth')
    elif sigma_hat > 50:
        path = os.path.join(mwcnn_path, 'MWCNN_50_55.pth')
    elif sigma_hat > 45:
        path = os.path.join(mwcnn_path, 'MWCNN_45_50.pth')
    elif sigma_hat > 40:
        path = os.path.join(mwcnn_path, 'MWCNN_40_45.pth')
    elif sigma_hat > 35:
        path = os.path.join(mwcnn_path, 'MWCNN_35_40.pth')
    elif sigma_hat > 30:
        path = os.path.join(mwcnn_path, 'MWCNN_30_35.pth')
    elif sigma_hat > 25:
        path = os.path.join(mwcnn_path, 'MWCNN_20_30.pth')
    elif sigma_hat > 20:
        path = os.path.join(mwcnn_path, 'MWCNN_20_30.pth')
    elif sigma_hat > 15:
        path = os.path.join(mwcnn_path, 'MWCNN_10_15.pth')  # keep your original final assignment
    elif sigma_hat > 5:
        path = os.path.join(mwcnn_path, 'MWCNN_5_10.pth')
    else:
        path = os.path.join(mwcnn_path, 'MWCNN_0_5.pth')

    noisy = np.array(noisy)
    noisy = torch.from_numpy(noisy)
    noisy = noisy.cuda()
    noisy = noisy / 255
    noisy = torch.reshape(noisy, (1, 1, int(highth), int(width)))
    noisy = noisy.float()
    x_hat = denoise_MWCNN_24(noisy, path)
    x_hat = x_hat.double()
    x_hat = x_hat * 255
    x_hat = torch.reshape(x_hat, (int(highth) * int(width), 1))

    x_hat = x_hat.cpu().numpy()
    x_hat = array.array('d', x_hat)
    return x_hat