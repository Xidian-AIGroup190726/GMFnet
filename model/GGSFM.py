import torch.nn as nn
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GGSFM(nn.Module):
    def __init__(self, in_dim, h_dim, k_size, mode=''):
        super(GGSFM, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=h_dim,
                              kernel_size=(1, 1), stride=1)
        self.cell1 = ConvGRUCell(in_dim, h_dim, (1, 1))
        self.cell2 = ConvGRUCell(in_dim, h_dim, (1, 1))
        self.cell3 = ConvGRUCell(in_dim, h_dim, (1, 1))
        self.h_cell1 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell2 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell3 = H_ConvGRU(in_dim, h_dim, k_size)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, in_tensor1, in_tensor2):

        out1, cur_state1 = self.cell1(in_tensor1[2], in_tensor2[2])
        out_h1, h_state1 = self.h_cell1(out1, cur_state1, cur_state1, mode='1')

        out2, cur_state2 = self.cell2(in_tensor1[1], in_tensor2[1])
        out_h2, h_state2 = self.h_cell2(out2, cur_state2, h_state1, mode='2')

        out3, cur_state3 = self.cell3(in_tensor1[0], in_tensor2[0])
        out_h3, h_state3 = self.h_cell3(out3, cur_state3, h_state2, mode='2')

        return (out_h1, out_h2, out_h3), (h_state1, h_state2, h_state3)


class H_ConvGRU(nn.Module):
    def __init__(self, in_dim, h_dim, k_size, name=''):
        super(H_ConvGRU, self).__init__()
        self.cell1 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell2 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell3 = ConvGRUCell(in_dim, h_dim, k_size)
        # self.cell4 = ConvGRUCell(in_dim, h_dim, k_size)
        self.SACA = SACA(h_dim, name)

    def init_hidden(self, tensor):
        batch_size = tensor.shape[0]
        height, width = tensor.shape[2], tensor.shape[3]
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, in_tensor, pre_state, state, mode):
        if mode == '1':
            out, cur_state1 = self.cell1(in_tensor, self.SACA(pre_state))
            out, cur_state2 = self.cell2(out+in_tensor, self.SACA(cur_state1))
            out, cur_state3 = self.cell3(out+in_tensor, self.SACA(cur_state2))
            # out, cur_state4 = self.cell4(out + in_tensor, self.SACA(cur_state3))
        else:
            if state[0].shape[3] != in_tensor.shape[3]:
                sample = nn.AdaptiveAvgPool2d((in_tensor.shape[2], in_tensor.shape[3]))
                out, cur_state1 = self.cell1(in_tensor, self.SACA(pre_state) + sample(state[0]))
                out, cur_state2 = self.cell2(out+in_tensor, self.SACA(cur_state1)+sample(state[1]))
                out, cur_state3 = self.cell3(out+in_tensor, self.SACA(cur_state2)+sample(state[2]))
                # out, cur_state4 = self.cell4(out + in_tensor, cur_state3 + self.SACA(self.upsample(state[3])))
            else:
                out, cur_state1 = self.cell1(in_tensor, pre_state + self.SACA(state[0]))
                out, cur_state2 = self.cell2(out+in_tensor, cur_state1+self.SACA(state[1]))
                out, cur_state3 = self.cell3(out+in_tensor, cur_state2+self.SACA(state[2]))
                # out, cur_state4 = self.cell4(out + in_tensor, cur_state3 + self.SACA(state[3]))
        return out, (cur_state1, cur_state2, cur_state3)


class ConvGRUCell(nn.Module):
    def __init__(self, in_dim, h_dim, k_size):
        super(ConvGRUCell, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.padding = k_size[0]//2
        self.bias = False

        self.z = nn.Conv2d(in_channels=in_dim+h_dim, out_channels=h_dim*2,
                           kernel_size=k_size, padding=self.padding,
                           bias=self.bias)
        self.r = nn.Conv2d(in_channels=in_dim+h_dim, out_channels=h_dim*2,
                           kernel_size=k_size, padding=self.padding,
                           bias=self.bias)
        self.h = nn.Conv2d(in_channels=in_dim+h_dim, out_channels=h_dim*2,
                           kernel_size=k_size, padding=self.padding,
                           bias=self.bias)
        self.conv_out = nn.Conv2d(in_channels=h_dim, out_channels=in_dim,
                                  kernel_size=k_size, padding=self.padding,
                                  bias=self.bias)
        self.BN = nn.BatchNorm2d(in_dim)

    def forward(self, in_tensor, cur_state):
        z_com = torch.cat([in_tensor, cur_state], dim=1)
        z_out = self.z(z_com)
        xz, hz = torch.split(z_out, [self.h_dim, self.h_dim], dim=1)
        z_t = torch.sigmoid(xz + hz)

        r_com = torch.cat([in_tensor, cur_state], dim=1)
        r_out = self.r(r_com)
        xr, hr = torch.split(r_out, [self.h_dim, self.h_dim], dim=1)
        r_t = torch.sigmoid(xr + hr)

        h_com = torch.cat([in_tensor, torch.mul(r_t, cur_state)], dim=1)
        h_out = self.h(h_com)
        xh, hh = torch.split(h_out, [self.h_dim, self.h_dim], dim=1)
        h_hat_t = torch.tanh(xh + hh)
        h_t = torch.mul((1 - z_t), cur_state) + torch.mul(z_t, h_hat_t)
        out = torch.sigmoid(self.BN(self.conv_out(h_t)))
        return out, h_t


class SACA(nn.Module):
    def __init__(self, inplanes, name=''):
        super(SACA, self).__init__()
        self.name = name
        self.SA = SpatialAttention()
        self.CA = ChannelAttention(inplanes)

    def forward(self, tensor):
        if self.name == 'CA':
            out = torch.mul(tensor, self.CA(tensor))
        elif self.name == 'SA':
            out = torch.mul(tensor, self.SA(tensor))
        else:
            out = torch.mul(tensor, self.CA(tensor))
            out = torch.mul(out, self.SA(tensor))
        return out


class ChannelAttention(nn.Module):
    def __init__(self, inplanes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
        # 通道注意力，即两个全连接层连接
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=max(inplanes // ratio, 1), kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=max(inplanes // ratio, 1), out_channels=inplanes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        # 最后输出的注意力应该为非负
        out = self.sigmoid(max_out + avg_out)
        return out


# torch.size([batch_size, 1, width, height])
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)  # torch.size([batch_size, 1, width, height])
        return out