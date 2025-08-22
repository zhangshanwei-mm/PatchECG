# 

# 计算shap的分数，批量计算
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from scipy import signal
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    roc_auc_score, f1_score,auc,roc_curve,multilabel_confusion_matrix
from sklearn.metrics import precision_recall_curve
from helper_code import *
from utils import *
from new_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# from tensorboardX import SummaryWriter
# from torchsummary import summary
from my_transformer import *
# import seaborn as sns
import shap
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
np.set_printoptions(threshold=np.inf)
np.random.seed(40) # 40,41,42,43,44
substring_labels = '# Labels:'
import time
import h5py

class MyDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label
        # self.patch_length = patch_length
        # self.stride = stride
        
    def __getitem__(self, index):
        # slide sequence to patch
        raw_data = np.array(self.data[index]) # (12,1000)
        # (12,1000) -> (12,num_patch,patch_length)  切割数据集
        total_length = raw_data.shape[1] # 1000
        patch_length = 64
        stride = 16
        
        # # ==================================================================================================================================
        # # 判断是否需要填充数据
        # # remainder = (total_length- patch_length) % stride
        # # if remainder !=0:
        # #     padding_needed = stride-remainder # 需要填充的数量
        # #     print(padding_needed)
        # #     padded_data = np.pad(raw_data, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0)
        # #     print(padded_data.shape) # 
        
        
        # # 处理最后一个确实的部分
        # channel_num_patch = (total_length-patch_length)//stride+1 # 59 update->60
        
        # # print(channel_num_patch)
        # # 创建一个空数组来存储划分后的数据
        # sliced_data = np.empty((12, channel_num_patch, patch_length)) # shape (12,59,64)
        
        # for i in range(channel_num_patch): # 59
        #     start = i * stride
        #     end = start + patch_length
        #     # sliced_data[:, i, :] = raw_data[:, start:end] # 最后一个直接神略
        #     sliced_data[:, i, :] = raw_data[:, start:end] # 添加最后一个，用0进行填充 （12,60,64）
        
        
        # # print(sliced_data.shape) # 12,59,64 -> 12,60,64
        # # ==================================================================================================================================
        
        # 第二种patch的方案
        # sliced_data = silde_windows_without_overlap(raw_data,patch_length) # （12,16,64）
        
        # 不要最后一个不完整的，不重叠
        sliced_data = silde_windows_without_overlap_v1(raw_data,patch_length)# （12,15,64）
        
        # print(sliced_data.shape) 
        
        return (torch.tensor(sliced_data, dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float))

    def __len__(self):
        return len(self.data)
      
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    params:
        kernel_size: kernel size
        stride: the stride of the window. Default value is kernel_size
    
    input: (n_sample, n_channel, n_length)
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class Swish(nn.Module):
    def forward(self, x): # a activation function
        return x * F.sigmoid(x)

class BasicBlock(nn.Module):
    """
    Basic Block: 
        conv1 -> convk -> conv1

    params:
        in_channels: number of input channels
        out_channels: number of output channels
        ratio: ratio of channels to out_channels
        kernel_size: kernel window length
        stride: kernel step size
        groups: number of groups in convk
        downsample: whether downsample length
        use_bn: whether use batch_norm
        use_do: whether use dropout

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, downsample, is_first_block=False, use_bn=True, use_do=True):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.middle_channels = int(self.out_channels * self.ratio)

        # the first conv, conv1
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=self.in_channels, 
            out_channels=self.middle_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # the second conv, convk
        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=self.middle_channels, 
            out_channels=self.middle_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the third conv, conv1
        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(
            in_channels=self.middle_channels, 
            out_channels=self.out_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # Squeeze-and-Excitation
        r = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels//r)
        self.se_fc2 = nn.Linear(self.out_channels//r, self.out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        out = x
        # the first conv, conv1
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv, convk
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # the third conv, conv1
        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out) # (n_sample, n_channel, n_length)

        # Squeeze-and-Excitation
        se = out.mean(-1) # (n_sample, n_channel) ,求均值
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = F.sigmoid(se) # (n_sample, n_channel)
        out = torch.einsum('abc,ab->abc', out, se)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out

class BasicStage(nn.Module):
    """
    Basic Stage:
        block_1 -> block_2 -> ... -> block_M
    """
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, i_stage, m_blocks, use_bn=True, use_do=True, verbose=False):
        super(BasicStage, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.i_stage = i_stage
        self.m_blocks = m_blocks
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        self.block_list = nn.ModuleList()
        for i_block in range(self.m_blocks):
            
            # first block
            if self.i_stage == 0 and i_block == 0:
                self.is_first_block = True
            else:
                self.is_first_block = False
            # downsample, stride, input
            if i_block == 0:
                self.downsample = True
                self.stride = stride
                self.tmp_in_channels = self.in_channels
            else:
                self.downsample = False
                self.stride = 1
                self.tmp_in_channels = self.out_channels
            
            # build block
            tmp_block = BasicBlock(
                in_channels=self.tmp_in_channels, 
                out_channels=self.out_channels, 
                ratio=self.ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=self.groups, 
                downsample=self.downsample, 
                is_first_block=self.is_first_block,
                use_bn=self.use_bn, 
                use_do=self.use_do)
            self.block_list.append(tmp_block)

    def forward(self, x):

        out = x

        for i_block in range(self.m_blocks):
            net = self.block_list[i_block]
            out = net(out)
            if self.verbose:
                print('stage: {}, block: {}, in_channels: {}, out_channels: {}, outshape: {}'.format(self.i_stage, i_block, net.in_channels, net.out_channels, list(out.shape)))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv1.in_channels, net.conv1.out_channels, net.conv1.kernel_size, net.conv1.stride, net.conv1.groups))
                print('stage: {}, block: {}, convk: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv2.in_channels, net.conv2.out_channels, net.conv2.kernel_size, net.conv2.stride, net.conv2.groups))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv3.in_channels, net.conv3.out_channels, net.conv3.kernel_size, net.conv3.stride, net.conv3.groups))

        return out

class Net1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    params:
        in_channels
        base_filters
        filter_list: list, filters for each stage
        m_blocks_list: list, number of blocks of each stage
        kernel_size
        stride
        groups_width
        n_stages
        n_classes
        use_bn
        use_do

    """

    def __init__(self, in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes, use_bn=True, use_do=True, verbose=False):
        super(Net1D, self).__init__()
        
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.ratio = ratio
        self.filter_list = filter_list
        self.m_blocks_list = m_blocks_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.n_stages = len(filter_list)
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        # first conv
        self.first_conv = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=self.base_filters, 
            kernel_size=self.kernel_size, 
            stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        # stages
        self.stage_list = nn.ModuleList()
        in_channels = self.base_filters
        for i_stage in range(self.n_stages):

            out_channels = self.filter_list[i_stage]
            m_blocks = self.m_blocks_list[i_stage]
            tmp_stage = BasicStage(
                in_channels=in_channels, 
                out_channels=out_channels, 
                ratio=self.ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=out_channels//self.groups_width, 
                i_stage=i_stage,
                m_blocks=m_blocks, 
                use_bn=self.use_bn, 
                use_do=self.use_do, 
                verbose=self.verbose)
            self.stage_list.append(tmp_stage)
            in_channels = out_channels

        # final prediction
        self.dense = nn.Linear(in_channels, n_classes)
        
    def forward(self, x):
        # x [1,1,64]
        out = x
        # first conv
        out = self.first_conv(out)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)
        
        # stages
        for i_stage in range(self.n_stages):
            net = self.stage_list[i_stage]
            out = net(out)

        # final prediction
        out = out.mean(-1)
        out = self.dense(out)
        
        return out

def my_mask(lead_tp = 1, len = 1000):
    """
    parameters:
        lead_tp : type of ECG , 1 表示12，1的排布，2表示3，4的排布，3表示6，2的排布
        len : 数据的长度
    retrun (12,length)
    """
    step = 1000
    if lead_tp == 2: 
        step = int(len/4)
    if lead_tp == 3:
        step = int(len/2)
        
    #print("每条导联的长度："+str(step))
    msk = np.zeros((12,len))
    
    if lead_tp == 2:
        msk[0,step:len] = 1
        msk[2,step:len] = 1
        
        msk[3,0:step] = 1
        msk[3,2*step:len] = 1
        msk[4,0:step] = 1
        msk[4,2*step:len] = 1
        msk[5,0:step] = 1
        msk[5,2*step:len] = 1
        
        msk[6,0:2*step] = 1
        msk[6,3*step:len] = 1
        msk[7,0:2*step] = 1
        msk[7,3*step:len] = 1
        msk[8,0:2*step] = 1
        msk[8,3*step:len] = 1
        
        msk[9,0:3*step] = 1
        msk[10,0:3*step] = 1
        msk[11,0:3*step] = 1
        
    if lead_tp == 3:
        msk[0,step:len] = 1
        msk[2,step:len] = 1
        msk[3,step:len] = 1
        msk[4,step:len] = 1
        msk[5,step:len] = 1

        msk[6,0:step] = 1
        msk[7,0:step] = 1
        msk[8,0:step] = 1
        msk[9,0:step] = 1
        msk[10,0:step] = 1
        msk[11,0:step] = 1
         
    return msk

def encode_labels(data, label_to_index):
# 初始化零矩阵，形状为[数据长度, 标签数量]
    target = torch.zeros((len(data), len(label_to_index)), dtype=torch.float32)
    
    # 填充矩阵
    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target

def decode_labels(encoded_labels, labels_dic):
    # 获取每个样本的标签
    decoded_labels = []
    for row in encoded_labels:
        # 获取值为1的标签索引
        labels_for_sample = [labels_dic[i] for i in range(len(row)) if row[i] == 1]
        decoded_labels.append(labels_for_sample)
    return decoded_labels

def silde_windows_without_overlap_v1(data,patch_length):
    """
        input:
            data : (channel,length)
            stride :
            patch_length : 
        output : (channel,num_patch,patch_length) ndarray
        comment : 不要最后一个缺失的块
    """
    total_length = data.shape[1] # 1000
    channel = data.shape[0] # 12
    stride = patch_length
    # 不要最后一个
    num_patch = (total_length-patch_length)//stride+1 
    sliced_data = np.empty((channel, num_patch, patch_length))
    for i in range(num_patch): # 16
        start = i * patch_length
        end = start + patch_length
        # sliced_data[:, i, :] = raw_data[:, start:end] # 最后一个直接神略
        sliced_data[:, i, :] = data[:, start:end] # 添加最后一个，用0进行填充 
    
    return sliced_data

def evaluate_predictions(scores, th, labels):
    """
    输入:
    scores - 分数数组，形状[400,1]
    th - 阈值，大于th的预测为1，否则为0
    labels - 真实标签数组，形状[400,1]，前200为1，后200为0
    
    输出:
    predictions - 预测标签数组
    confusion_matrix - 混淆矩阵
    metrics - 评估指标
    """
    predictions = (scores > th).astype(int)
    labels = np.array(labels).reshape(-1, 1)
    
    TP = np.sum((predictions == 1) & (labels == 1))
    FP = np.sum((predictions == 1) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))
    TN = np.sum((predictions == 0) & (labels == 0))
    
    confusion_mat = np.array([[TP, FP], [FN, TN]])
    
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  # 特异度
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,  # 特异度
        'f1_score': f1_score,
        'confusion_matrix': confusion_mat
    }
    print(f"TP is {TP}")
    print(f"FP is {FP}")
    print(f"FN is {FN}")
    print(f"TN is {TN}")

    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"{k}: {v:.4f}")


with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_6x2.h5', 'r') as f:
    data1 = f['data'][:]
    label1 = f['labels'][:]
    numbers1 = f['numbers'][:]
    
test_label = torch.from_numpy(label1)

# read 12x1 AF 100 张

with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_12x1.h5', 'r') as f:
    data2 = f['data'][:]
    label2 = f['labels'][:]
    numbers2 = f['numbers'][:]

# read NAF 200 张
with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_NAF.h5', 'r') as f:
    data3 = f['data'][:]
    label3 = f['labels'][:]
    numbers3 = f['numbers'][:]

data1 = np.moveaxis(data1, 1, 2)
data2 = np.moveaxis(data2, 1, 2)
data3 = np.moveaxis(data3, 1, 2)

af_test_data = np.concatenate((data1, data2, data3), axis=0) # [400,1000,12]
af_test_label = np.concatenate([np.ones((200, 1)), np.zeros((200, 1))])
print(f"AF chaoyang data shape : {af_test_data.shape}")
print(f"AF chaoyang label shape :{af_test_label.shape}")


n_epoch = 30
batch_size = 1
input_dim = 768
header = 16
num_layers = 4
s = 16 # stride
patch_len = 64

# dataset_val = MyDataset(val_set,vaild_label)
dataset_test = MyDataset(af_test_data,af_test_label)

# dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=True,shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=True,shuffle=False)
    
device_str = "cuda:0"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")

# make model
model1 = Net1D(
    in_channels=2, 
    base_filters=16, 
    ratio=1.0, 
    filter_list=[16,32,32,40,40,64,64],
    m_blocks_list=[2,2,2,2,2,2,2], 
    kernel_size=16, 
    stride=2, 
    groups_width=16,
    verbose=False,
    n_classes=768)
# 加载模型
parameters = torch.load('/data/0shared/zhangshanwei/cinc/ours/src/case_study/attention2_2/checkpoint_29.pth', map_location='cuda:0') # , map_location='cuda:3'
thresholds = parameters['Best_thresholds']

model1.load_state_dict(parameters['model1'])
model1.to(device)

# model2 = TransformerEncoderForMultiLabelClassification(input_dim, header, num_layers, dim_feedforward=1024, num_labels=11, dropout=0.1)
model3 = VisionTransformer_2dembedding_v2(
                            embed_dim=768,
                            depth=3,
                            num_heads=8,
                            representation_size=None,
                            num_classes=1,
                            num_patch = 15,
                            channel = 12,
                            drop_ratio=0.,
                            attn_drop_ratio=0., 
                            drop_path_ratio=0.
                            )

# 加载模型
model3.load_state_dict(parameters['model2'])
model3.to(device)

model_params = list(model1.parameters()) + list(model3.parameters())

optimizer = optim.Adam(model_params, lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
loss_func = MultiLabelFocalLoss()
total_train_loss = [] # loss
temp = []

# test
model1.eval()
model3.eval()
prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
all_test_y = [] # true
all_test_pred = [] # pred score

input_feature = []
with torch.no_grad():
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)

        all_test_y.append(input_y.cpu().data.numpy()) # 真实标签

        mask = torch.isnan(input_x).all(dim=-1) # B,C,N [256,12,15] 
        input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
        temp_x = input_x.view(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # 256 x 12 x15 
            
        # add 2 channel mask
        channel_2_mask = torch.zeros_like(input_x)
        channel_2_mask[torch.isnan(input_x)] = 0.0
        channel_2_mask[~torch.isnan(input_x)] = 1.0
        channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
        
        PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [256 x 12 x15,1,768]
        # [256,12,15,768]
        output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)
        output_x[mask] = 0

        # print(output_x.shape) # [b,12,15,768]
        input_feature.append(output_x.cpu().data.numpy())
        pred = torch.sigmoid(model3(output_x)) # add sigmoid 最后一层需要
        all_test_pred.append(pred.cpu().data.numpy()) # add score

af_label = np.concatenate(all_test_y)
af_score = np.concatenate(all_test_pred)

Auroc_AF = roc_auc_score(af_label, af_score)
print(f"AUROC 400 = {Auroc_AF:.4f}")



def find_best_thresholds(labels, scores):

    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)  # Youden指数
    optimal_thresh = thresholds[optimal_idx]
    print(f"推荐阈值 = {optimal_thresh:.2f}")

    return optimal_thresh
best_400_th = find_best_thresholds(af_label, af_score)
evaluate_predictions(af_score, th=best_400_th, labels=af_label)


# ===========================================================================
# 利用shap计算每一个patch的分数
patch_input = np.concatenate(input_feature) # [400,12,15,768] 输入数据 
print(patch_input.shape)


# 使用Gradient
start_time = time.time()
tensor_data = torch.tensor(patch_input,dtype=torch.float32,device="cuda:0") # 放入gpu当中，放在同一个gpu当中需要
explainer = shap.GradientExplainer(model3,tensor_data)
shap_values = explainer.shap_values(tensor_data)
# print(shap_values.shape) # (50, 12, 15, 768, 1)

end_time = time.time()
elapsed_time = end_time - start_time
torch.save(shap_values,"/data/0shared/zhangshanwei/cinc/ours/src/case_study/attention2_2/score_save/shap_value_graident.pt")
print(f"代码运行时间: {elapsed_time:.4f} 秒")



# =============================
# 2、定义shap解释函数
# =============================
# model3 = model3.eval().cpu()

# 选择背景和待解释样本
# indices = np.random.choice(patch_input.shape[0], 20, replace=False)
# background = patch_input[indices]  # shape: [20, 12, 15, 768]


# data_to_explain = patch_input[0:25]  # 待解释样本: [1, 12,15, 768]
# background_tensor = torch.tensor(background, dtype=torch.float32)
# data_to_explain_tensor = torch.tensor(data_to_explain, dtype=torch.float32)

# print(background_tensor.shape) # (10, 12, 15, 768)
# print(data_to_explain_tensor.shape) # (1, 12, 15, 768)

# target_class = 1
# =============================
# Step 3: 计算 SHAP 值
# =============================

# start_time = time.time()
# explainer = shap.DeepExplainer(model3, background_tensor)
# # 添加, check_additivity=False，临时解决
# shap_values = explainer.shap_values(data_to_explain_tensor, check_additivity=False)

# torch.save(shap_values,"/data/0shared/zhangshanwei/cinc/ours/src/case_study/attention2/shape_value_0_25.pt")
# print(shap_values.shape)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"代码运行时间: {elapsed_time:.4f} 秒")







