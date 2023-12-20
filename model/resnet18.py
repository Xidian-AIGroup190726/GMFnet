import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 kernel
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# get BasicBlock which layers < 50(18, 34)
class BasicBlk(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlk, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.expansion * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, self.expansion * out_ch,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_ch)
            )
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:  # is not None
            x = self.downsample(x)  # resize the channel
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(Resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(4, 64)
        self.conv2 = conv3x3(5, 64)
        self.conv3 = conv3x3(1, 64)
        self.BN = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=0, mode='2'):
        if mode == '2':
            out = torch.concat([x, F.interpolate(y, size=(16, 16))], dim=1)
            out = self.relu(self.BN(self.conv2(out)))  # torch.Size([20, 64, 16, 16])
        else:
            out = self.relu(self.BN(self.conv1(x)))  # torch.Size([20, 64, 16, 16])
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


def Net(args):
    return Resnet(BasicBlk, [2, 2, 2, 2], args['Categories_Number'])


def test_net():
    ms  = torch.randn([1, 4, 16, 16]).to(device)
    pan = torch.randn([1, 1, 64, 64]).to(device)
    cfg = {
        'Categories_Number': 8
    }
    net = Net(cfg).to(device)
    y = net(ms, pan)
    print(y.shape)


if __name__ == '__main__':
    test_net()
