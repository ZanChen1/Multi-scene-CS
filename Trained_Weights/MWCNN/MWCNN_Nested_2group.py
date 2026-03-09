import sys

sys.path.append("..")
import common
import CBAM
import torch
import torch.nn as nn


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        output = self.activation(x)

        return output


class RDB_Conv(nn.Module):  # RDB内部卷积
    def __init__(self, inChannels, growRate, kSize=3):  # growth rate，每�?�? block 里面的卷积层送到下一个卷积层的特征的数量
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):  # RDB块G0为初始输入特征�?�道数，G为每个卷积块为下�?个卷积块提供的特征�?�道�?;由于dense连接，后面的卷积块的输入会�?�渐增加G
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)  # �?1×1卷积进行�?部特征融�?

    def forward(self, x):
        return self.LFF(self.convs(x)) + x  # RDB�?部残差连�?


def make_model(parent=False):
    return MWCNN()


class MWCNNgroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act=nn.ReLU(True)):
        super(MWCNNgroup, self).__init__()

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        # Nested中间网络
        self.d_l0_nested_0_1 = conv_block_nested(n_feats + n_feats // 2, n_feats)
        self.d_l0_nested_0_2 = conv_block_nested(n_feats * 2 + n_feats // 2, n_feats)
        self.d_l1_nested_1_1 = conv_block_nested(n_feats * 2 + n_feats, n_feats * 2)

        # Nested解码端融合网�?
        self.d_l1_fuse = nn.Conv2d(n_feats * 4, n_feats * 2, kernel_size=1, bias=True)
        self.d_l0_fuse = nn.Conv2d(n_feats * 3, n_feats, kernel_size=1, bias=True)

        # d_l0 = [common.BBlock(conv, n_feats, n_feats, kernel_size, act=act)]
        # d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))
        d_l0 = []
        d_l0.append(RDB(n_feats, n_feats // 2, 2))  # 64 32 2

        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]
        # d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)) #d = 2 , 1
        d_l1.append(RDB(n_feats * 2, n_feats, 2))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))
        # d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(RDB(n_feats * 4, n_feats * 2, 2))

        pro_l3 = []
        pro_l3.append(common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        # pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))#d = 2 , 3
        pro_l3.append(RDB(n_feats * 8, n_feats * 4, 2))

        # pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))#d = 3 , 2
        pro_l3.append(RDB(n_feats * 8, n_feats * 4, 2))
        pro_l3.append(common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        # i_l2 = [common.DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)] #d = 2 , 1
        i_l2 = [RDB(n_feats * 4, n_feats * 2, 2)]
        i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False))

        # i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1 = [RDB(n_feats * 2, n_feats, 2)]
        i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False))

        # i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]
        i_l0 = [RDB(n_feats, n_feats // 2, 2)]

        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)

    def forward(self, x):
        x0_0 = self.d_l0(x)  # 输出64
        x1_0 = self.d_l1(self.DWT(x0_0))  # 输出128
        x0_1 = self.d_l0_nested_0_1(torch.cat([x0_0, self.IWT(x1_0)], 1))  # x0_0+x1_0上采�? 输入64+32 输出64

        x2_0 = self.d_l2(self.DWT(x1_0))  # 输出256
        x1_1 = self.d_l1_nested_1_1(torch.cat([x1_0, self.IWT(x2_0)], 1))  # x1_0+x2_0上采�? 输入128+64 输出128
        x0_2 = self.d_l0_nested_0_2(torch.cat([x0_0, x0_1, self.IWT(x1_1)], 1))  # x0_0+x0_1+x1_1上采�? 输入64+64+32 输出64

        x2_1 = self.IWT(self.pro_l3(self.DWT(x2_0))) + x2_0  # 输入256 输出256
        x1_2 = self.d_l1_fuse(torch.cat([x1_1, self.IWT(self.i_l2(x2_1))], 1)) + x1_0  # x1_1 + x2_1上采�? 输入128+128 输出128
        x0_3 = self.d_l0_fuse(
            torch.cat([x0_1, x0_2, self.IWT(self.i_l1(x1_2))], 1)) + x0_0  # x0_1 + x0_2 + x1_2上采�? 输入64+64+64 输出64
        x = self.i_l0(x0_3) + x  # �?外层残差

        return x


class MWCNN(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(MWCNN, self).__init__()
        n_resblocks = 20  # args.n_resblocks
        n_feats = 64  # args.n_feats
        kernel_size = 3  # 3
        self.scale_idx = 0
        nColor = 1  # args.n_colors
        act = nn.ReLU(True)

        MWCNN_head = [conv(nColor, n_feats, kernel_size)]

        MWCNN_group1 = [MWCNNgroup(conv, n_feats, kernel_size)]
        MWCNN_group2 = [MWCNNgroup(conv, n_feats, kernel_size)]

        MWCNN_fuse = [conv(2 * n_feats, n_feats, kernel_size=1)]

        MWCNN_CBAM = [CBAM.CBAM(n_feats)]
        MWCNN_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*MWCNN_head)
        self.group1 = nn.Sequential(*MWCNN_group1)
        self.group2 = nn.Sequential(*MWCNN_group2)
        self.CBAM = nn.Sequential(*MWCNN_CBAM)
        self.fuse = nn.Sequential(*MWCNN_fuse)
        self.tail = nn.Sequential(*MWCNN_tail)

    def forward(self, x):
        x_head = self.head(x)
        x_res1 = self.group1(x_head)
        x_res2 = self.group2(x_res1)
        x_res = torch.cat((x_res1, x_res2), 1)  # 128
        x_res = self.fuse(x_res)  # 64

        x_res = self.CBAM(x_res)
        x_res = x_res + x_head
        x_res = self.tail(x_res)

        x_clean = x + x_res  # �?外层残差

        return x_clean

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


if __name__ == '__main__':
    from thop import profile
    inputdata = torch.randn(1, 1, 256, 256).to('cuda:0')
    model = MWCNN().cuda()
    flops, params = profile(model, inputs=(inputdata,))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
