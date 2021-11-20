import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy




class SEBlk(nn.Module):
    def __init__(self, ch_in, reduction=12):
        super(SEBlk, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)

        return x * y


class ConvBlk(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=20, stride=3):
        super(ConvBlk, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, stride=stride, ),
            nn.ReLU(),
            nn.BatchNorm1d(ch_out),
        )

    def forward(self, x):
        y = self.conv(x)

        return y


class GRU(nn.Module):
    def __init__(self, input_size=130, hidden_size=32, num_layers=2):
        super(GRU, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  #（time_step,batch,input）时是Ture
        )
        self.out1 = nn.Linear(32, 10)
        self.out2 = nn.Linear(10, 7)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # x (batch,time_step,input_size)
        out = self.out1(r_out[:, -1, :])  # (batch,time_step,input)
        out = self.out2(out)
        return out


class SENet_GRU(nn.Module):
    def __init__(self):
        super(SENet_GRU, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=3)
        self.se1 = SEBlk(120, 12)
        self.se2 = SEBlk(32,  12)
        # self.se3 = SEBlk(32, 12)
        self.conv1 = ConvBlk(1, 120, 20, 3)
        self.conv2 = ConvBlk(120, 32, 8, 1)
        # self.conv3 = ConvBlk(32, 32, 3)
        # self.GRU = nn.GRU()
        self.gru = GRU(130, 32, 2) # # x_dim,h_dim,layer_num

    def forward(self, x):
        '''

        :param x: [32, 1, 3600]
        :return:
        '''
        # [b, 1, 3600] => [b, 120, 398]
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.se1(x)
        # [b, 120, 398] => [b, 32, 130]
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.se2(x)
        x = self.gru(x)
        return x

# (out_len - 1) * stride + kernel_size = in_len + 2 * padding


# net = SENet_GRU()
# input = torch.randn(32, 1, 3600)
# output = net(input)
# print(output.shape)

# 画图
# a = output.detach().numpy()
# # plt.plot(range(391), a[0][0])
# plt.show()



