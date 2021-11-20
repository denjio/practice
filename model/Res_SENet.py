from torch import nn
from torch.nn import functional as F
import torch


class SEBlk(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SEBlk, self).__init__()
        # from one channel pooling one value
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
        # print(y.shape)  # ([32, 32])
        y = self.fc(y).view(b, c, 1)
        # print(y.shape)  # ([32, 32, 1])
        return x * y


class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, l] => [ b, ch_out, l]
            self.extra = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=2, stride=stride),
                nn.BatchNorm1d(ch_out)
            )
    def forward(self, x):
        """

        :param x: [b, ch ,l]
        :return:
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.shape, x.shape)
        # Short cut
        # extra module: [b, ch_in, l] with [b, ch_out, l]
        # element-wise add
        # print(self.extra(x).shape)
        out = self.extra(x) + out

        return out


class Res_SENet(nn.Module):

    def __init__(self):
        super(Res_SENet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(32)

        )
        # followed 4 blocks
        # [b, 64, l] => [b, 128, l]
        self.blk1 = ResBlk(32, 64, stride=2)
        # [b, 128, l] => [b, 256, l]
        self.blk2 = ResBlk(64, 128, stride=3)
        # [b, 256, l] => [b, 512, l]
        self.blk3 = ResBlk(128, 256, stride=2)
        # [b, 512, l] => [b, 1024, l]
        self.blk4 = ResBlk(256, 512, stride=2)
        self.blk5 = ResBlk(512, 512, stride=1)
        self.se1 = SEBlk(64, 16)
        self.se2 = SEBlk(128, 16)
        self.se3 = SEBlk(256, 16)
        self.se4 = SEBlk(512, 16)
        self.se5 = SEBlk(512, 16)
        self.outlayer1 = nn.Linear(512*10, 512)
        self.outlayer2 = nn.Linear(512, 7)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        # [b, 64, l] => [b, 1024, l]
        x = self.blk1(x)
        x = self.se1(x)
        x = self.blk2(x)
        x = self.se2(x)
        x = self.blk3(x)
        x = self.se3(x)
        x = self.blk4(x)
        x = self.se4(x)
        x = self.blk5(x)
        x = self.se5(x)

        # print('after conv', x.shape) # [b, 512, 50]
        # [b, 512, l] => [b, 512, l/5]
        x = F.adaptive_max_pool1d(x, (int(len(x[0][0]) / 5)))
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.outlayer1(x)
        # print(x.shape)
        x = self.outlayer2(x)
        # print(x.shape)
        return x


# def main():
#
#     # tmp = torch.randn(32, 32, 3600)
#     # out = SEBlk(32)
#     # out = out(tmp)
#     # print(out.shape)
#
#     x = torch.randn(32, 1, 3600)
#     model = Res_SENet()
#     out = model(x)
#
#
# if __name__ == '__main__':
#     main()

