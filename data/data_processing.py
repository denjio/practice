import scipy
import wfdb  # 导入wfdb包读取数据文件
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# 返回加载数据的文件名，对应使用的导联方式
def load_data(Lead_way, Using_lead_way, sig_name, load_name, name):
    i = len(Lead_way) - 1
    while (i >= 0):
        if Lead_way[i] in sig_name:
            print('Accuracy data uses method of lead_way:', Lead_way[i])
            return load_name.append(name), Using_lead_way.append(Lead_way[i])
        i -=  1


type = []
rootdir = 'european-st-t-database-1.0.0'  # 共同导联方式V4
# rootdir = 'sudden-cardiac-death-holter-database-1.0.0'        #心脏性猝死数据库
# rootdir = 'mit-bih-st-change-database-1.0.0'     #欧盟st-t数据库 导联方式ECG
# rootdir = 'mit-bih-arrhythmia-database-1.0.0'  # 共同导联方式MLII
# rootdir = 'mit-bih-atrial-fibrillation-database-1.0.0'  # 导联方式ECG1,ECG2

files = os.listdir(rootdir)  # 列出文件夹下所有
flg = 0
name_list = []  # name_list=[100,101,...234]
Lead_way = ['MLI', 'MLII', 'MLIII', 'V1', 'V2', 'V3', 'V4', 'V5', 'D3', 'ECG1']
Using_lead_way = []  # 用一种导联采集的人（根据选择的不同导联方式会有变换）
load_name = []  # 加载数据的文件名
type = {}  # 标记及其数量

for file in files:
    if file[0:5] in name_list:  # 选取文件的前五个字符（可以根据数据文件的命名特征进行修改）
        continue
    else:
        name_list.append(file[0:5])

print(name_list)

for name in name_list:  # 遍历每一个人
    print(name)
    if name[1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:  # 判断——跳过无用的文件
        continue
    record = wfdb.rdrecord(rootdir + '/' + name)  # 读取一条记录（100），不用加扩展名
    display(record.__dict__)
    print(record.p_signal.shape)
    display(record.sig_name)
    #     if Lead_way[flg] in record.sig_name:  # 这里我们记录Lead_Way导联的数据（也可以记录其他的，根据数据库的不同选择数据量多的一类导联方式即可）
    #     Using_lead_way.append(name)  # 记录下这个人
    load_data(Lead_way, Using_lead_way, record.sig_name, load_name, name)
    print(Using_lead_way)

    annotation = wfdb.rdann(rootdir + '/' + name, 'atr')  # 读取一条记录的atr文件，扩展名atr
    display(annotation.__dict__)
    print('asd:',len(annotation.symbol), annotation.sample.shape)
    for symbol in annotation.symbol:  # 记录下这个人所有的标记类型
        if symbol in type.keys():
            type[symbol] += 1
        else:
            type[symbol] = 1
    # # 画图
    # whole_signal = wfdb.rdrecord(rootdir + '/' + name).p_signal.transpose()
    # print(whole_signal.shape)
    # x = np.zeros(len(whole_signal[0]))
    # print(x.shape)
    # print(whole_signal[0])
    # print((annotation.sample))
    # # plt.plot(range(len(whole_signal[0])), whole_signal[0],  'b', label='before')
    # plt.plot(range(360), whole_signal[0][:360],  'b', label='before')
    # for i in range(360):
    #     plt.text(annotation.sample[i], whole_signal[0][annotation.sample[i]], annotation.symbol[i])
    # plt.grid()
    # plt.legend()
    # plt.show()
    s = sorted(type.items(), key=lambda d: d[1], reverse=True)  # s == list
    type = dict(s)
    print('sympbol_name', type)
print(len(Using_lead_way))
# ========================================================
'''
将MLII导联方式通道的数据（65000）分为每段（2500）再进行重采样（3600），
并且每段节拍注释进行综合标记
如果数据本身无标签，标记为Q；
如果数据标签均为N，标记为N；
如果数据标签中含有非N的标签，将其进行记录并统计个数，选取非N标签中出现最多的那个标签对这条数据进行标记。
'''
f = 250  # 数据库的原始采样频率
segmented_len = 10  # 将数据片段裁剪为10s
label_count = 0
count = 0
abnormal = 0

segmented_data = []  # 最后数据集中的X
segmented_label = []  # 最后数据集中的Y
print('begin!')

idx = 0  # 用于对应的load_data使用的导联方式下标
# MLII = ['100', '101', '103', ... '234']
for person in load_name:  # 读取导联方式为MLII的数据
    k = 0
    # transpose [650000, 2] => [2, 650000]
    whole_signal = wfdb.rdrecord(rootdir + '/' + person).p_signal.transpose()  # 这个人的一整条数据，
    # len(whole_signal[0]) = 650000
    while (k + 1) * f * segmented_len <= len(whole_signal[0]):  # 只要不到最后一组数据点
        count += 1
        # 0--2500
        record = wfdb.rdrecord(rootdir + '/' + person, sampfrom=k * f * segmented_len,
                               sampto=(k + 1) * f * segmented_len)  # 读取一条记录（100），不用加扩展名
        annotation = wfdb.rdann(rootdir + '/' + person, 'atr', sampfrom=k * f * segmented_len,
                                sampto=(k + 1) * f * segmented_len)  # 读取一条记录的atr文件，扩展名atr

        # 对采用MLII导联方式的心电信号进行重采样
        lead_index = record.sig_name.index(Using_lead_way[idx])  # 找到MLII导联对应的索引
        # signal.shape =(2, 2500)
        signal = record.p_signal.transpose()  # 两个导联，转置之后方便画图

        label = []  # 这一段数据对应的label，最后从这里面选择最终的label
        # segmented_data.append(signal[lead_index])   # 只记录MLII导联的数据段
        symbols = annotation.symbol
        # len(signal[lead_index]= 2500 => len(re_signal)=3600
        re_signal = scipy.signal.resample(signal[lead_index], 3600)  # 采样
        re_signal_3 = np.round(re_signal, 3)
        print('resignal', re_signal_3)
        segmented_data.append(re_signal_3)

        # segmented_data.append(re_signal)
        print('symbols', symbols, len(symbols))

        # if '+' in symbols:  # 删去+
        #     symbols.remove('+')
        if len(symbols) == 0:
            segmented_label.append('Q')
        elif symbols.count('N') / len(symbols) == 1 or symbols.count('N') + symbols.count('/') == len(
                symbols):  # 如果全是'N'或'/'和'N'的组合，就标记为N
            segmented_label.append('N')
        else:
            for i in symbols:
                if i != 'N':
                    label.append(i)
            segmented_label.append(label[0])

        # print(label)
        k += 1
    print('next')
    idx += 1
# ===========================================================
print('begin to save dataset!')

# print(pd.DataFrame(segmented_data))
# print(pd.DataFrame(segmented_label))
segmented_data = pd.DataFrame(segmented_data)
segmented_label = pd.DataFrame(segmented_label)
segmented_data.to_csv('european-st-t_X__MLII.csv', index=False)
segmented_label.to_csv('european-st-t_Y__MLII.csv', index=False)

print('Finished!')
