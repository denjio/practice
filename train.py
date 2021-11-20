# -*- coding: UTF-8 -*-
# ====================================================== 导入需要的包==================================
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.signal import medfilt
from model.senet_gru import SENet_GRU
from model.Resnet import ResNet
from model.Conv1 import Conv1
from model.swinTRM import SwinTransformer
from model.Res_SENet import Res_SENet
from model.Res_SE_GRUNet import Res_SE_GRUNet
# def fix_baseline_wander(data, fs=500):
#     data = np.array(data)
#     winsize = int(round(0.2 * fs))
#     # delayBLR = round((winsize-1)/2)
#     if winsize % 2 == 0:
#         winsize += 1
#     baseline_estimate = medfilt(data, kernel_size=winsize)
#     winsize = int(round(0.6 * fs))
#     # delayBLR = delayBLR + round((winsize-1)/2)
#     if winsize % 2 == 0:
#         winsize += 1
#     baseline_estimate = medfilt(baseline_estimate, kernel_size=winsize)
#
#     ecg_blr = data - baseline_estimate
#     return ecg_blr


# =================================================== 读取数据================================================
print("load data...")

X_list = [r'data/mit-bih-arrhythmia_X__MLII.csv']  # , r'data/european-st-t_X__MLII.csv']#, r'data/mit-bih-atrial-fibrillation_X__MLII.csv']

Y_list = [r'data/mit-bih-arrhythmia_Y__MLII.csv']  # , r'data/european-st-t_Y__MLII.csv']#, r'data/mit-bih-atrial-fibrillation_X__MLII.csv']

X = np.loadtxt('data/X.csv', delimiter=',', skiprows=1).astype('float32')  # [choose_index]
Y = np.loadtxt('data/Y.csv', dtype="str", delimiter=',', skiprows=1)  # [choose_index]
# 合并数据集
print("begin concatenating...")
for database in X_list:
    X = np.concatenate((X, (np.loadtxt(database, dtype="str", delimiter=',', skiprows=1).astype(np.float))))
for database in Y_list:
    Y = np.concatenate((Y, (np.loadtxt(database, dtype="str", delimiter=',', skiprows=1))))
print(" concatenating finished .")

AAMI = ['N', 'L', 'R', 'V', 'A', '|', 'B']
# N:Normal
# L:Left bundle branch block beat
# R:Right bundle branch block beat
# V:Premature ventricular contraction
# A:Atrial premature contraction
# |:Isolated QRS-like artifact
# B:Left or right bundle branch block
delete_list = []
for i in range(len(Y)):
    if Y[i] not in AAMI:  # 删除不在AAMI中标签的数据
        delete_list.append(i)
X = np.delete(X, delete_list, 0)
Y = np.delete(Y, delete_list, 0)

# 数据标准化：
print("begin standard scaler...")
ss = StandardScaler()
std_data = ss.fit_transform(X)
# print(X.shape) （6480， 3600）=> (6480, 1, 3600)
X = np.expand_dims(X, axis=1)  # 在中间加了一维

# print(Y.shape)

# 画图
# y = np.zeros(1200)
#
# plt.plot(range(1200), X[0][0][:1200], 'b', label='befor')
#
# y = fix_baseline_wander(X[0][0][:1200], 500)
#
# plt.plot(range(1200), y, 'r', label='after')
# plt.grid()
# plt.legend()
# plt.show()

# 把标签编码
le = preprocessing.LabelEncoder()
le = le.fit(AAMI)
Y = le.transform(Y)

# Y.shape = (6480,) ,X.shape = (6480, 3600, 1)

print("the label before encoding:", le.inverse_transform([0, 1, 2, 3, 4, 5, 6]))
# 分层抽样
print("begin StratifiedShuffleSplit...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9, random_state=0)
sss.get_n_splits(X, Y)
for train_index, test_index in sss.split(X, Y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    # y_train = [3 5 3 ... 3 3 3]
    # y_train = np.eye(7)[y_train]
    # y_test = np.eye(7)[y_test]  # 转换为对角矩阵
print(X_train.shape, y_train)  # (5832, 1, 3600) (5832,)
print(X_test.shape, y_test)  # (5832, 1, 3600) (5832,)
X_train1 = X_train[:, :, : 3584]
X_test1 = X_test[:, :, : 3584]

# ===================================================模型训练==================================================

batch_size = 32

# X_train = torch.from_numpy(X_train)
X_train = torch.from_numpy(X_train1)

y_train = torch.from_numpy(y_train)

# X_test = torch.from_numpy(X_test)
X_test = torch.from_numpy(X_test1)

y_test = torch.from_numpy(y_test)
train_dataset = TensorDataset(X_train, y_train, )
test_dataset = TensorDataset(X_test, y_test)
train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda')
# model = SENet_GRU().to(device)
# model = ResNet().to(device)
# model = Conv1().to(device)
# model = Res_SENet().to(device)
model = SwinTransformer().to(device)
criteon = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)
# d_loss = []
# d_acc = []
for epoch in range(20):
    model.train()  # train model
    for batchidx, (x, label) in enumerate(train):
        # x: [b, 1, 3600]
        # label: [b]
        x, label = x.to(device), label.to(device)
        x = x.type(torch.cuda.FloatTensor)
        logits = model(x)
        # logits:[b, 7]
        # label: [b]
        # loss: tensor scalar
        loss = criteon(logits, label.long())

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()  # test model
    with torch.no_grad():  # 不需要backprop
        # test
        total_correct = 0
        total_num = 0
        for x, label in test:
            # [b, 1, 3600]
            # [b]
            x, label = x.to(device), label.to(device)
            x = x.type(torch.cuda.FloatTensor)
            # [b, 7]
            logits = model(x)
            # [b]
            pred = logits.argmax(dim=1)  # argmax是最大的索引，max取得是值 这里需要可能性最大的索引就是标签
            # [b] vs [b] => scalar tensor
            total_correct += torch.eq(pred, label).float().sum()
            total_num += x.size(0)

        acc = total_correct / total_num
        # d_loss.append(loss.cpu().numpy())
        # d_acc.append(acc.cpu().numpy())
        # print(d_acc, d_loss)
        print('epoch:', epoch, 'loss:', loss.item(), 'acc:', acc)

# 画图

# fig = plt.figure()
# ax1, ax2 = fig.subplots(2, 1, sharey=True)
# ax1.set_xlabel('epoch')
# ax1.set_ylabel('loss')
# ax1.plot(range(20), d_loss)
# ax1.xaxis.set_major_locator(MultipleLocator(1))
#
# ax2.plot(range(20), d_acc)
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('acc')
# ax2.xaxis.set_major_locator(MultipleLocator(1))
# plt.show()
