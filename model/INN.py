import torch
import torch.nn as nn


class RevNet(nn.Module):
    def __init__(self, num_channels, kernel_size=3, padding=1):
        super(RevNet, self).__init__()
        self.num_channels = num_channels

        self.layer1 = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.layer2 = nn.BatchNorm2d(num_channels)
        self.layer3 = nn.ReLU(inplace=True)
        self.layer4 = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.layer5 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x1)))))
        y2 = x2 + y1
        return torch.cat((y2, x1), dim=1)

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        x1 = y2 - self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(y1)))))
        return torch.cat((x1, y1), dim=1)


class RevNetModel(nn.Module):
    def __init__(self, num_channels=512, kernel_size=3, num_layers=10):
        super(RevNetModel, self).__init__()

        self.num_channels = num_channels
        self.num_layers = num_layers
        self.layers = nn.ModuleList([RevNet(num_channels, kernel_size, kernel_size//2) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x

    def inverse(self, y):
        for i in reversed(range(self.num_layers)):
            y = self.layers[i].inverse(y)
        return y


if __name__ == '__main__':
    img1 = torch.randn([20, 1024, 4, 4])
    model = RevNetModel()
    res = model.forward(img1)
    print(res.shape)
    res = model.inverse(res)
    print(res.shape)
