import torch.nn as nn
import torch
import torch.nn.functional as F
from INN import *
from GGSFM import *
from contourlet_torch import ContourDec
from MI import mutual_information
from resnet18 import BasicBlk


class gmfnet(nn.Module):  # 19 (1, 1) 20 (3, 3)
    def __init__(self, block, num_blocks, args, max_channel=64):
        super(gmfnet, self).__init__()
        self.args = args

        self.nlevs = 2
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2, cat=1)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2, cat=1)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2, cat=1)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2, cat=1)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2, cat=1)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2, cat=1)

        self.GRU_l = GGSFM(in_dim=4, h_dim=1, k_size=(1, 1))
        self.GRU_s = GGSFM(in_dim=2 ** self.nlevs, h_dim=4 * 2 ** self.nlevs, k_size=(1, 1))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        self.SA_p = nn.ModuleList()
        self.SA_m = nn.ModuleList()
        for i in range(3):
            self.inn.append(RevNetModel(num_channels=max_channel * 2 ** i + 4, kernel_size=3, num_layers=2))
            self.SA_p.append(SpatialAttention())
            self.SA_m.append(SpatialAttention())
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride, cat=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if cat == 1 and stride == strides[0]:
                layers.append(block(self.in_planes + 4, planes, stride))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def IBi(self, p, m, b, l, i):
        p_out = torch.mul(p, self.SA_p[i](l[i]))
        p_out = torch.concat((p_out, l[i]), dim=1)
        m_out = torch.mul(m, self.SA_m[i](b[i]))
        m_out = torch.concat((m_out, b[i]), dim=1)
        out = self.inn[2 - i](torch.cat((p_out, m_out), dim=1))
        return torch.chunk(out, 2, dim=1)

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)

        ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)

        # GGSFM module
        out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))

        p = self.conv_p(pan)
        m = self.conv_m(ms)
        p_out = self.p_layer1(p)
        m_out = self.m_layer1(m)
        p_out, m_out = self.IBi(p_out, m_out, out_s, out_l, 2)

        p_out = self.p_layer2(p_out)
        m_out = self.m_layer2(m_out)
        p_out, m_out = self.IBi(p_out, m_out, out_s, out_l, 1)

        p_out = self.p_layer3(p_out)
        m_out = self.m_layer3(m_out)
        p_out, m_out = self.IBi(p_out, m_out, out_s, out_l, 0)

        p_out = self.p_layer4(p_out)
        m_out = self.m_layer4(m_out)

        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])

        mi = 0
        for i in range(len(state_l)):
            mi += mutual_information(state_l[i][-1], state_s[i][-1])

        return out, mi * 0.1


def Net(args):
    return gmfnet(BasicBlk, [2, 2, 2, 2], args)


def test_net():
    ms  = torch.randn([1, 4, 16, 16]).to(device)
    pan = torch.randn([1, 1, 64, 64]).to(device)
    cfg = {
        'Categories_Number': 8
    }
    net = Net(cfg).to(device)
    y, _ = net(ms, pan)
    print(y.shape)


if __name__ == '__main__':
    test_net()