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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
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
from tensorboardX import SummaryWriter
from torchsummary import summary
import pickle
import math
from my_transformer import *
import seaborn as sns
import ast,h5py
import json
np.set_printoptions(threshold=np.inf)
np.random.seed(40) # 45
substring_labels = '# Labels:'
test_fold = 6
vaild_fold = 5
device_str = "cuda:9"

# # nohup python -u code_v5_AF.py > ./AF/logs/2025_5_16_1fold 2>&1 &


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# Extract features.
def extract_features(record): # extract feature
    # 空数据全部置为0，即可
    lead_12,fields = load_signals(record)
    a = np.array(lead_12)
    b = np.transpose(a,[1,0]) # change dim
    df = pd.DataFrame(b) 
    arr = df.fillna(0).values
    signal = list(arr)
    return signal

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


# transformer model
class TransformerEncoderForMultiLabelClassification(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_labels, dropout=0.1):
        super(TransformerEncoderForMultiLabelClassification, self).__init__()
        """
            d_model:输入vector的维度
            nhead 
            num_layers:
            dim_feedforward : 前馈神经网络的隐藏测层维度
            dropout : 
            输出：
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_labels = num_labels
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))
        
        # self.time_positional_encoding = positionalencoding1d(708,d_model).unsqueeze(0) # (1,length,d_model) 708,128
        self.time_positional_encoding = positionalencoding1d(180,d_model).unsqueeze(0)
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 全连接层用于多标签分类
        # self.fc1 = nn.Linear(d_model*708, 128) # 90624 的输入
        self.fc1 = nn.Linear(d_model*180, 128)
        self.fc2 = nn.Linear(128, num_labels)
        
    def forward(self, src):
        # input (batch,length,d_model) 16,708,128
        # src: (batch, length, d_model)
        batch_size, length, d_model = src.size()
        
        # 添加位置编码
        # pe = self.positional_encoding[:length,d_model] # shape (d_model,length) 708,128 ->1,708,128
        # pe.to("cuda:2")
        # time_position = pe.unsqueeze(0) # 708,128 ->1,708,128
        # src = src + self.positional_encoding[:, :length, :] 
        self.time_positional_encoding.to("cuda:3")
        # print(self.time_positional_encoding.shape) # (1,length,d_model) 1,708,128
        
        src = src + self.time_positional_encoding
        
        # 调整维度顺序以适应Transformer的输入要求 (length, batch, d_model)
        # src = src.permute(1, 0, 2)
        # print("print input transformer dim:..")
        # print(src.shape) # 16,708,128
        
        # 通过Transformer编码器
        output = self.transformer_encoder(src)
        # print(output.shape) # 16,708,128
        output = torch.flatten(output,1) # (16,708,128) - > (16,90624)
        
        # print(output.shape) # （16,90624）
                
        output = self.fc1(output)
        output = self.fc2(output)
        

        
        
        # # 调整回原来的维度顺序 (batch, length, d_model)
        # output = output.permute(1, 0, 2)
        # 取最后一个时间步的输出作为分类器的输入
        # cls_output = output[:, -1, :]  # (batch, d_model)
        # 通过全连接层进行多标签分类
        # logits = self.fc(cls_output)  # (batch, num_labels)
        return output


def positionalencoding1d(length, d_model):
    """
        input : param d_model: dimension of the model
              : param length: length of positions
        output : return: length*d_model position matrix
        
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model).to("cuda:3")
    position = torch.arange(0, length).unsqueeze(1).to("cuda:3")
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model))).to("cuda:3")
    pe[:, 0::2] = torch.sin(position.float() * div_term).to("cuda:3")
    pe[:, 1::2] = torch.cos(position.float() * div_term).to("cuda:3")

    
    return pe 
        
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def create_2d_relative_bias_trainble_embeddings(n_head,height,width,dim):
    position_embedding = nn.Embedding((2*width-1)*(2*height-1),n_head)
    nn.init.constant_(position_embedding,0.)
    def get_relative_position_index(height,width):
        coords = torch.stack(torch.meshgrid(torch.arange(height),torch.arange(width))) # [2,height,width]
        coords_flatten = torch.flatten(coords,1) # [2,height * width]
        relative_coords_bias = coords_flatten[:,:,None]-coords_flatten[:,None,:] # [2,height*width,height*width]
        
        relative_coords_bias[0,:,:] += height - 1
        relative_coords_bias[1,:,:] += width - 1
        
        relative_coords_bias[0,:,:] *= relative_coords_bias[1,:,:].max()+1
        return relative_coords_bias.sum(0) # [height * width,height * width]
    
    relative_position_bias = get_relative_position_index(height,width)
    bias_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(height*width,height*width,n_head) # [height*width,height*width,n_head]
    bias_embedding = bias_embedding.permute(2,0,1).unsqueeze(0) # [1,n_head,height*width,height*width]
    
    return bias_embedding

def create_2d_absolate_sincos_embeddings(height,width,dim):
    """
        2d 绝对路径的embedding
    """
    assert dim % 4==0,"wrong dimension"
    
    position_embedding = torch.zeros(height*width,dim)
    coords = torch.stack()
    return position_embedding

def silde_windows(data,stride,patch_length):
    """
        input :
            data : (channel,length)
            stride :
            patch_length : 
        output : (channel,num_patch,patch_length)
        
    """
    
    total_length = data.shape[1]
    channel = data.shap[0]
    
    # 判断是否需要填充数据
    
    
    return 1

def silde_windows_without_overlap(data,patch_length):
    """
        input :
            data : (channel,length)
            stride :
            patch_length : 
        output : (channel,num_patch,patch_length) ndarray
    """
    
    total_length = data.shape[1] # 1000
    channel = data.shape[0] # 12
    
    # 判断需要填充的数据
    remainder = total_length % patch_length # 40
    padding_needed = 0 if remainder == 0 else patch_length - remainder  # 24
    padded_data = np.pad(data, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0) # (12,1024)
    num_patch = int(padded_data.shape[1]/patch_length) # 16
    sliced_data = np.empty((channel, num_patch, patch_length)) # 12,16,64
    for i in range(num_patch): # 16
        start = i * patch_length
        end = start + patch_length
        # sliced_data[:, i, :] = raw_data[:, start:end] # 最后一个直接神略
        sliced_data[:, i, :] = padded_data[:, start:end] # 添加最后一个，用0进行填充 
    
    return sliced_data

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

# 计算评估指标

def evaluate_binary_classification(y_true, y_prob, threshold=0.5):
    """
    评估二分类任务，输入真实标签和预测概率

    参数：
        y_true: ndarray, shape (n,) or (n,1)，真实标签，0/1
        y_prob: ndarray, shape (n,) or (n,1)，预测概率（sigmoid 输出）
        threshold: float，用于将概率转换为0/1预测标签

    返回：
        指标字典
    """
    # 展平
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)

    # 概率 → 二值预测
    y_pred = (y_prob >= threshold).astype(int)

    # 基本指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Specificity': specificity,
        'AUC-ROC': auc,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    }


path = "/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id') # 这个数据是database_里面包含了AFIB的数据
X = load_raw_data(Y, sampling_rate, path)

X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != vaild_fold))]
y_train = Y[((Y.strat_fold != test_fold)& (Y.strat_fold != vaild_fold))].scp_codes # 用于保存训练集的label

X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].scp_codes

X_vaild = X[np.where(Y.strat_fold == vaild_fold)]
y_vaild = Y[Y.strat_fold == vaild_fold].scp_codes

# 编码为label
y_label_list = []
y_label_test_list = []
y_label_val_list = []

# train
for i in range(len(y_train)): # 0 norm ,1 AFIB
    # print(y_train.iloc[i]) # {'CLBBB': 100.0, 'AFIB': 0.0}
    if 'AFIB' in y_train.iloc[i]:
        y_label_list.append(1)
    else:
        y_label_list.append(0)
train_label = np.array(y_label_list).reshape(-1,1)

# test
for i in range(len(y_test)): # 0 norm ,1 AFIB
    if 'AFIB' in y_test.iloc[i]:
        y_label_test_list.append(1)
    else:
        y_label_test_list.append(0)
test_label = np.array(y_label_test_list).reshape(-1,1)

# val
for i in range(len(y_vaild)): # 0 norm ,1 AFIB
    if 'AFIB' in y_vaild.iloc[i]:
        y_label_val_list.append(1)
    else:
        y_label_val_list.append(0)
vaild_label = np.array(y_label_val_list).reshape(-1,1)


# print(test_label[0:50])
# print(vaild_label[0:50])
print(f"Train label shape : {train_label.shape}") 
print(f"Test label shape : {test_label.shape}") 
print(f"Val label shape : {vaild_label.shape}")


# 处理data的部分
# 随机生成2个部分的数据集
val_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(X_vaild))
test_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(X_test))

temp_val_data = list()
for i in range(len(X_vaild)):
    temp_layout_data = gen_true_ecg_layout(X_vaild[i],length = 1000,layout = val_list[i])
    temp_val_data.append(temp_layout_data)

temp_test_data = list()
for i in range(len(X_test)):
    temp_layout_data = gen_true_ecg_layout(X_test[i],length=1000,layout = test_list[i])
    temp_test_data.append(temp_layout_data)

xtest = np.array(temp_test_data)
xtest = np.moveaxis(xtest, 1, 2)

xval = np.array(temp_val_data)
xval = np.moveaxis(xval, 1, 2)

print(f"X Test shape :{xtest.shape}")
print(f"X Val shape :{xval.shape}")

# add AF Test ,直接加在这个里面进行测试=======================

with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_6x2.h5', 'r') as f:
    data1 = f['data'][:]
    numbers1 = f['numbers'][:]
    
# read 12x1 AF 100 张
with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_12x1.h5', 'r') as f:
    data2 = f['data'][:]
    numbers2 = f['numbers'][:]

# read NAF 200 张
with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_NAF.h5', 'r') as f:
    data3 = f['data'][:]
    numbers3 = f['numbers'][:]

data1 = np.moveaxis(data1, 1, 2)
data2 = np.moveaxis(data2, 1, 2)
data3 = np.moveaxis(data3, 1, 2)

af_test_data = np.concatenate((data1, data2, data3), axis=0) # [400,1000,12]
af_test_label = np.concatenate([np.ones((200, 1)), np.zeros((200, 1))])

print(f"AF chaoyang data shape : {af_test_data.shape}")
print(f"AF chaoyang label shape :{af_test_label.shape}")

# add AF Test ,直接加在这个里面进行测试=======================

# build model
n_epoch = 30
batch_size = 128
input_dim = 768
header = 16
num_layers = 4
s = 16 # stride
patch_len = 64

dataset_val = MyDataset(xval,vaild_label)
dataset_test = MyDataset(xtest,test_label)
dataset_test_AF = MyDataset(af_test_data,af_test_label)

dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=True)
dataloader_test_AF = DataLoader(dataset_test_AF, batch_size=batch_size, drop_last=True)

device = torch.device(device_str if torch.cuda.is_available() else "cpu")

model1 = Net1D(
    in_channels=2, 
    base_filters=16, 
    ratio=1.0, 
    filter_list=[16,32,32,40,40,64,64], # [16,32,32,40,40,64,64]
    m_blocks_list=[2,2,2,2,2,2,2], # [2,2,2,2,2,2,2]
    kernel_size=16, 
    stride=2, 
    groups_width=16,
    verbose=False,
    n_classes=768)
model1.to(device)

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
model3.to(device)
model_params = list(model1.parameters()) + list(model3.parameters())

optimizer = optim.Adam(model_params, lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
loss_func = torch.nn.BCEWithLogitsLoss()
# MultiLabelFocalLoss()
total_train_loss = [] # loss
temp = []
step = 0
count = 0

for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
    # train
    model1.train()
    model3.train()
    
    # 每一次都重新mask train数据
    
    xtrain = np.moveaxis(X_train,1, 2)
    xtrain_1 = list()
    for i in range(xtrain.shape[0]):
    #     # temp_data = mask_12channal_train(xtrain[i])
        temp_data = xtrain[i]
        # noisy_data = add_noisy(temp_data,noise_std = 0.1) # 添加噪声
        mask_data = add_mask(temp_data) # 添加缺失
        xtrain_1.append(mask_data)
    train_set = np.array(xtrain_1) # 采用了mask 策略之后的训练集

    dataset = MyDataset(train_set, train_label)
    dataloader = DataLoader(dataset, batch_size=batch_size,drop_last=True)
    # 每一次添加，不同的随机mask，每一轮mask的位置不一样

    prog_iter = tqdm(dataloader, desc="Training", leave=False , ncols=80)
    for batch_idx, batch in enumerate(prog_iter):
        input_x, input_y = tuple(t.to(device) for t in batch) # [256, 12, 15, 64]            
        mask = torch.isnan(input_x).any(dim=-1) # b,12,15
        
        # 根据这个nan生成一个相同维度的mask
        channel_2_mask = torch.zeros_like(input_x)
        channel_2_mask[torch.isnan(input_x)] = 0.0
        channel_2_mask[~torch.isnan(input_x)] = 1.0
        channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
        
        # 2、将所有数据输入的时候全部置为nan变为0
        input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
        temp_x = input_x.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # B x 12 x15 
        # print(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1).shape)
        
        
        PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [B x 12 x15,2,768]
        output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)
        output_x[mask] = 0
        # input transformer model 
        pred = model3(output_x)
        # loss = loss_func(pred, input_y.float())
        loss = loss_func(pred.float(), input_y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        temp.append(loss.data.cpu())
        
    total_train_loss.append(np.mean(temp))
    print("第{}轮loss:".format(_+1),np.mean(temp))

    temp.clear()
    scheduler.step(_)

    # Val
    model1.eval()
    model3.eval()

    prog_iter_val = tqdm(dataloader_val, desc="Val", leave=False)
    all_pred_val = [] # 验证集 pred
    all_y_val = [] # 验证集 true
    all_val_score = [] # 验证集 score
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_val):
            input_x, input_y = tuple(t.to(device) for t in batch)
            all_y_val.append(input_y.cpu().data.numpy()) # 真实标签
            
            mask = torch.isnan(input_x).all(dim=-1) # B,C,N [256,12,15] 
            input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
            temp_x = input_x.view(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # 256 x 12 x15 
            # 根据这个nan生成一个相同维度的mask
            channel_2_mask = torch.zeros_like(input_x)
            channel_2_mask[torch.isnan(input_x)] = 0.0
            channel_2_mask[~torch.isnan(input_x)] = 1.0
            channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
        
            PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [256 x 12 x15,1,768]
            # [256,12,15,768]
            output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)
            output_x[mask] = 0
            
            # transformer model
            pred = torch.sigmoid(model3(output_x))
            # add score
            all_val_score.append(pred.cpu().data.numpy()) 
            
            # add pred 
            predicted_labels = (pred >= 0.5).float()
            all_pred_val.append(predicted_labels.cpu().data.numpy())
            
            
    pred_val = np.concatenate(all_pred_val)
    true_val = np.concatenate(all_y_val)
    score_val = np.concatenate(all_val_score)

    results_val = evaluate_binary_classification(true_val, score_val)
    for k, v in results_val.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


    thresholds = find_optimal_thresholds(true_val,score_val)

    print(f"thresholds :{thresholds}")

    # Test
    model1.eval()
    model3.eval()
    prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
    all_test_y = [] # true
    all_test_pred = [] # pred score
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

            pred = torch.sigmoid(model3(output_x)) # add sigmoid 最后一层需要
            all_test_pred.append(pred.cpu().data.numpy()) # add score
            
    test_true = np.concatenate(all_test_y)
    test_score = np.concatenate(all_test_pred)
    test_labels = np.zeros_like(test_score)
        
    # 使用验证集恢复后的结果
    results_test = evaluate_binary_classification(test_true, test_score,threshold=float(thresholds[0]))
    for k, v in results_test.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Test AF
    model1.eval()
    model3.eval()
    prog_iter_test_AF = tqdm(dataloader_test_AF, desc="Testing", leave=False)

    all_test_y_AF = [] # true
    all_test_pred_AF = [] # pred score
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test_AF):
            input_x, input_y = tuple(t.to(device) for t in batch)
            all_test_y_AF.append(input_y.cpu().data.numpy()) # 真实标签

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

            pred = torch.sigmoid(model3(output_x)) # add sigmoid 最后一层需要
            all_test_pred_AF.append(pred.cpu().data.numpy()) # add score

    test_true_AF = np.concatenate(all_test_y_AF)
    test_score_AF = np.concatenate(all_test_pred_AF)
    test_labels_AF = np.zeros_like(test_score_AF)

    print("================================")
    print("AF chaoyang result :")
    results_test_AF_adjust = evaluate_binary_classification(test_true_AF, test_score_AF,threshold=float(thresholds[0]))
    for k, v in results_test_AF_adjust.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("================================")

    results_test_AF_5 = evaluate_binary_classification(test_true_AF, test_score_AF,threshold=0.5)
    for k, v in results_test_AF_5.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("================================")
    save_check_AF(results_test,model1,model3,count,thresholds)
    count = count + 1

# nohup python -u code_v5_AF.py > ./AF/logs/2025_5_16_1fold 2>&1 &