from torch import nn
from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


class Conv1(nn.Module):
    """
    resnet block
    """
    def __init__(self,):
        """

        :param ch_in:
        :param ch_out:
        """
        super(Conv1, self).__init__()

        self.conv = nn.Conv1d(1, 32, kernel_size=3, stride=3, padding=0)
        self.bn = nn.BatchNorm1d(32)
        # 1
        self.conv1 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
        )

        # 2
        self.conv2 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, stride=3, padding=1),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
        )
        # 3
        self.conv3 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256)
        )
        # 4
        self.conv4 = nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(512),
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
        )
        # 5
        self.conv5 = nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
        )
        self.outlayer1 = nn.Linear(512*10, 512)
        self.outlayer2 = nn.Linear(512, 7)


    def forward(self, x):
        """

        :param x: [b, ch ,l]
        :return:
        """
        x = F.relu(self.conv(x))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(out.shape) # ([32, 512, 50])
        x = F.adaptive_max_pool1d(x, (int(len(x[0][0]) / 5)))
        x = x.view(x.size(0), -1)
        x = self.outlayer1(x)
        out = self.outlayer2(x)
        return out


# def main():
#
#     x = torch.randn(32, 1, 3600)
#     model = Conv1()
#     out = model(x)
#     plt.subplot(1,2,1)
#     plt.plot(range(20), range(20), label='loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.grid()
#     plt.subplot(1,2,2)
#     plt.plot(range(20), range(20), label='acc')
#     plt.ylabel('acc')
#     plt.grid()
#     plt.show()
#     # Creates two subplots and unpacks the output array immediately
#     fig = plt.figure()
#     ax1, ax2 = fig.subplots(2, 1, sharey=True)
#     ax1.set_xlabel('epoch')
#     ax1.set_ylabel('loss')
#     ax1.plot(range(1,21), range(20))
#     ax1.set_xticks = list(range(0, 20,1))
#     ax1.xaxis.set_major_locator(MultipleLocator(1))
#     # 把x轴的主刻度设置为1的
#     ax1.set_title('Sharing Y axis')
#     ax2.scatter(range(20), range(0, 20))
#     ax2.set_ylim(0, 1)
#     ax2.set_xlabel('epoch')
#     ax2.set_ylabel('loss')
#     plt.show()
# if __name__ == '__main__':
#     main()
#
