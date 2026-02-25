import torch
from torch import nn


class SigCNNhalf(nn.Module):
    def __init__(self,size):
        super(SigCNNhalf, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        # self.conva1 = nn.Conv2d(1, 64, 5, 1, 2, bias=True)
        # self.conva2 = nn.Conv2d(64, 64, 5, 1, 2, bias=True)
        # self.conva3 = nn.Conv2d(64, 64, 5, 1, 2, bias=True)
        # self.conva4 = nn.Conv2d(64, 64, 5, 1, 2, bias=True)
        # self.conva5 = nn.Conv2d(64, 64, 5, 1, 2, bias=True)
        self.conv1 = nn.Conv2d(1, 64, 5, 2, 2, bias=True)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2, bias=True)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1, bias=True)
        self.conv5 = nn.Conv2d(256, 256, 3, 2, 1, bias=True)
        self.conv6 = nn.Conv2d(256, 256, 3, 2, 1, bias=True)

        self.conv7 = nn.Conv2d(256, 1, 1, 1, 0, bias=True)

    def forward(self, x):
        # a1 = self.conva1(x)
        # a2 = self.relu(self.conva2(a1))
        # a3 = self.conva3(a2)
        # a3_1 = a3 + a1
        # a4 = self.relu(self.conva4(a3_1))
        # a5 = self.conva5(a4)
        # a5_1 = a5 + a3_1
        # a5_2 = a1 - a5_1

        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        c3 = self.relu(self.conv3(c2))
        c4 = self.relu(self.conv4(c3))
        c5 = self.relu(self.conv5(c4))
        c6 = self.relu(self.conv6(c5))
        c7 = torch.nn.functional.adaptive_avg_pool2d(c6, (1, 1)) #全局平均池化
        c8 = self.conv7(c7)
        output = torch.squeeze(c8,dim =2)
        output = torch.squeeze(output, dim=2)
        return output


if __name__ == '__main__':
    # 是否使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Model ###')
    x = torch.rand(4, 1, 256, 256).to(device)
    model = SigCNNhalf(256).to(device)

    y = model(x)
    print(y)
