#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

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
import h5py
import math
from my_transformer import *
import seaborn as sns
import ast
import json
np.set_printoptions(threshold=np.inf)
np.random.seed(40) # 40,41,42,43,44
substring_labels = '# Labels:'

# from tensorboardX import SummaryWriter
# writer = SummaryWriter(logdir='/data/zhangshanwei/code_path_ecg/logs')


def eva_classify_model(pic_path,table_path,layout,verbose):
    
    # read data
    database_path = "/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
    # pic_path = "../eva/ex1/picture/AUC_2fold_layout_4"
    # table_path = "../digitalization/ours_gen_ours_dig_1.xlsx"
    
    # df = pd.read_csv(database_path)
    sampling_rate=100
    # load and convert annotation data
    Y = pd.read_csv(database_path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, database_path)
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(database_path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    labels_dic = ['NORM','LAFB/LPFB','IRBBB','ILBBB','CLBBB','CRBBB','_AVB','IVCD','WPW','LVH','RVH',
                  'LAO/LAE','RAO/RAE','SEHYP','AMI','IMI','LMI','PMI','ISCA','ISCI','ISC_','STTC',
                  'NST_']
    
    label_to_index = {label: idx for idx, label in enumerate(labels_dic)}
    print("Label to index mapping:")
    print(label_to_index)
    
    test_fold = 10
    vaild_fold = 9
        
    
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    test_delete_data = list()
    test_delete_label = list()
    num_test = 0
    for i,j in zip(X_test,y_test):
        temp_data = i
        temp_label = j
        if not temp_label:
            num_test = num_test+1
            continue
        test_delete_data.append(temp_data)
        test_delete_label.append(temp_label)
    print("删除val中缺失label的数量:"+str(num_test))
    test_label = encode_labels(test_delete_label, label_to_index)
    
    xtest = np.array(test_delete_data)
    xtest = np.moveaxis(xtest, 1, 2)
    
    
    # # # vaild 
    X_vaild = X[np.where(Y.strat_fold == vaild_fold)]
    y_vaild = Y[Y.strat_fold == vaild_fold].diagnostic_superclass
    val_delete_data = list()
    val_delete_label = list()
    num_val = 0
    for i ,j in zip(X_vaild,y_vaild):
        temp_data = i
        temp_label = j
        if not temp_label:
            num_val=num_val+1
            continue
        val_delete_data.append(temp_data)
        val_delete_label.append(temp_label)
    print("删除val中缺失label的数量:"+str(num_val))
    vaild_label = encode_labels(val_delete_label, label_to_index)
        
    # 0 : 3 x 4
    # 1 : 3 x 4 + Ⅱ
    # 2 : 3 x 4 + Ⅱ + V1
    # 3 : 2 x 6 
    # 4 : 2 x 6 + Ⅱ
    # 5 : 12 
    # 6 : 各种排布随机出现，layout
    temp_test_data = list()
    if layout != 6:
        for i in range(len(test_delete_data)):
            temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = layout)
            temp_test_data.append(temp_layout_data)
    
    if layout == 6:
        test_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(test_delete_data))
        for i in range(len(test_delete_data)):
            temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = test_list[i])
            temp_test_data.append(temp_layout_data)
    
    xtest = np.array(temp_test_data)
    xtest = np.moveaxis(xtest, 1, 2)
    
    
    # xval = np.array(val_delete_data)
    # xval = np.moveaxis(xval, 1, 2)
    
    
    # 标准化xtest
    for i in range(len(xtest)):
        tmp_data = xtest[i]
        tmp_std = np.nanstd(tmp_data)
        tmp_mean = np.nanmean(tmp_data)
        xtest[i] = (tmp_data - tmp_mean) / tmp_std
    print(xtest.shape)
    
    
    
    # ===============================eva signal from ECG picture 
    # with h5py.File('/data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5', 'r') as f:
    #     xtest = f['signals'][:]
    #     test_label = f['labels'][:]
    
    # print(xtest.shape) # 
    # ===============================eva signal from ECG picture 
    

    # ======================eva siganl from ours dig and ours gen ecg data
    # 10fold 
    # /data/0shared/zhangshanwei/data/2024cinc/10fold_from_picture
    
    # with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/10fold_c4_f1.h5', 'r') as f:
    #     aaa = f['signals'][:]
    #     test_label = f['labels'][:]
    # xtest = np.array(aaa)
    
    # for i in range(len(xtest)):
    #     tmp_data = xtest[i]
    #     tmp_std = np.nanstd(tmp_data)
    #     tmp_mean = np.nanmean(tmp_data)
    #     xtest[i] = (tmp_data - tmp_mean) / tmp_std
    # ======================eva siganl from ours dig and ours gen ecg data
    
    
    # ============eva from chaoyan hospital ECG from paper ECG digitalization
    
    # with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan.h5', 'r') as f:
    #     xtest = f['data'][:]
    #     test_label_aa = f['labels'][:]
        

    # test_label = torch.from_numpy(test_label_aa)
    # xtest = np.moveaxis(xtest, 1, 2)
    # print(xtest.shape) # [80,1000,12]
    # print(test_label.shape) # [80,23]
    
    # ============eva from chaoyan hospital ECG from paper ECG digitalization
    
    
    
    lead_labels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # parameters
    n_epoch = 30
    batch_size = 16
    input_dim = 768
    header = 16
    num_layers = 4
    s = 16 # stride
    patch_len = 64
    
    # dataset_val = MyDataset(val_set,vaild_label)
    dataset_test = MyDataset(xtest,test_label)

    # dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=True,shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=True,shuffle=False)
     
    if verbose:
        print("mask is Done!")

    device_str = "cuda:9"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # make model
    model1 = Projection1(input_dim=patch_len,output_dim=input_dim,bias=0.1)


    # 加载模型
    parameters = torch.load('/data/0shared/zhangshanwei/cinc/ours/src/PatchEncoder/model/5fold/checkpoint_30.pth', map_location='cuda:9')
    thresholds = parameters['Best_thresholds']
    
    model1.load_state_dict(parameters['model1'])
    model1.to(device)
    
    # model2 = TransformerEncoderForMultiLabelClassification(input_dim, header, num_layers, dim_feedforward=1024, num_labels=11, dropout=0.1)
    model3 = VisionTransformer_2dembedding_v2(
                              embed_dim=768,
                              depth=3,
                              num_heads=8,
                              representation_size=None,
                              num_classes=23,
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
    
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            # print(input_x.shape)
            # print(input_y.shape)
            
            all_test_y.append(input_y.cpu().data.numpy()) # 真实标签

            mask = torch.isnan(input_x).all(dim=-1) # B,C,N [256,12,15] 
            # input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
            # temp_x = input_x.view(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # 256 x 12 x15 
                
            # # add 2 channel mask
            # channel_2_mask = torch.zeros_like(input_x)
            # channel_2_mask[torch.isnan(input_x)] = 0.0
            # channel_2_mask[~torch.isnan(input_x)] = 1.0
            # channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
            
            # PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [256 x 12 x15,1,768]
            # # [256,12,15,768]
            # output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)

            input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
            output_x = model1(input_x)# b,12,15,768
            output_x[mask] = 0
    
            pred = torch.sigmoid(model3(output_x)) # add sigmoid 最后一层需要
            all_test_pred.append(pred.cpu().data.numpy()) # add score
    
    
    # ==== eva AF
    # af_label = np.concatenate(all_test_y)
    # af_score = np.concatenate(all_test_pred)
    # print(af_label.shape)
    # print(af_score.shape)
    # af_pred = np.zeros_like(af_label[:,1])
    # af_pred = (af_score[:, 1] >= thresholds[1]) # 大于阈值即为这个
    
    # np.savetxt("/data/0shared/zhangshanwei/cinc/ours/digitalization/table/chaoyan_af_80.csv",af_pred, delimiter=",", fmt="%.6f")
    # exit()
    
    # precision = precision_score(af_label, af_pred)
    # print(precision)
    
    # 这部分代码写的不对，等一会想一下。
    # af_AUC = 0
    # af_AUC = roc_auc_score(test_true[:, 1], test_labels[:, 1])
    # print("AF AUC = {}".format(af_AUC))
    # ==========只计算AF的AUC
    # fpr_af, tpr_af, th = roc_curve(af_label[:, 1], af_score[:, 1])
    # roc_auc_af = auc(fpr_af, tpr_af)
    # print("AF AUC = {}".format(roc_auc_af))
    
    # exit()
    
    # ===== eva AF
            

    test_true = np.concatenate(all_test_y)
    test_score = np.concatenate(all_test_pred)
    test_labels = np.zeros_like(test_score)
    
    # thresholds = 0.5
    for i in range(23):
        test_labels[:, i] = (test_score[:, i] >= thresholds[i])
    print(parameters['Best_thresholds'])
    
    # 保存
    # np.savetxt("./case/subclasses/true_lead8.csv",test_true, delimiter=",", fmt="%.6f")
    # np.savetxt("./case/subclasses/pred_lead8.csv", test_labels, delimiter=",", fmt="%.6f")
    
    # # decode label 更容易看一些
    
    
    
    

    
    print("***********************************")
    f1_macro_test = f1_score(test_true,test_labels , average='macro')
    f1_micio_test = f1_score(test_true,test_labels , average='micro')
    print("F1_macro:{}".format(f1_macro_test))
    print("F1_micro:{}".format(f1_micio_test))
        
    auc_scores_test = []
    for i in range(23):
        # 计算当前标签的AUC
        try:
            auc_a = roc_auc_score(test_true[:, i], test_labels[:, i])
        except ValueError as e:
            print(f"Error: {e}")
            auc_a = 0 

        auc_scores_test.append(auc_a)

    print("NORM :{},\nLAFB/LPFB :{},\nIRBBB :{},\nILBBB :{},\nCLBBB :{},"\
            "\nCRBBB :{},\n_AVB :{},\nIVCD :{},\nWPW :{},\nLVH :{},\nRHV :{},"\
            "\nLAO/LAE:{},\nRAO/RAE :{},\nSEHYP :{},\nAMI :{},\nIMI :{},\nLMI :{},\nPMI :{},"\
            "\nISCA:{},\nISCI:{},\nISC_:{},\nSTTC:{},\nNST_:{}"
            .format(auc_scores_test[0],
                     auc_scores_test[1],
                     auc_scores_test[2],
                     auc_scores_test[3],
                     auc_scores_test[4],
                     auc_scores_test[5],
                     auc_scores_test[6],
                     auc_scores_test[7],
                     auc_scores_test[8],
                     auc_scores_test[9],
                     auc_scores_test[10],
                     auc_scores_test[11],
                     auc_scores_test[12],
                     auc_scores_test[13],
                     auc_scores_test[14],
                     auc_scores_test[15],
                     auc_scores_test[16],
                     auc_scores_test[17],
                     auc_scores_test[18],
                     auc_scores_test[19],
                     auc_scores_test[20],
                     auc_scores_test[21],
                     auc_scores_test[22]))
    print("***********************************")

    
    # plt csv result 
    # result_list = [[f1_macro_test, f1_micio_test, auc_scores_test[0],auc_scores_test[1],auc_scores_test[2],
    #                     auc_scores_test[3],auc_scores_test[4]]]
    
    # columns = ['F1_macro','F1_micro',
    #                'AUC_NORM','MI','AUC_STTC','AUC_CD',
    #                'AUC_HYP']
    
    # dt = pd.DataFrame(result_list, columns=columns)
    # dt.to_csv('./eva/table/v4.csv', mode='a')
    
    # plt roc curve
    # 计算ROC曲线和AUC（每个标签）
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict() 
    list_auc = list()
    th_comfusion_matrix = [] # 最优的阈值通过roc计算出来的
    for i in range(test_true.shape[1]):
        fpr[i], tpr[i], th = roc_curve(test_true[:, i], test_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        list_auc.append(roc_auc[i])
    plt.figure()
    for i in range(test_true.shape[1]):
        if i == 6:
            plt.plot(fpr[i], tpr[i], label=f'AVB (AUC = {roc_auc[i]:.2f})')
            continue
        plt.plot(fpr[i], tpr[i], label=f'{labels_dic[i]}(AUC = {roc_auc[i]:.2f})')
    
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--')
    # 设置图形属性S
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-Label Multi-Class')
    plt.legend(loc="lower right",fontsize='xx-small')
    plt.subplots_adjust(right=0.8) # 调整子图布局
    plt.savefig(pic_path)
    

    
    
    # save auc
    try:
        existing_df = pd.read_excel(table_path)
    except FileNotFoundError:
        print("文件 "+table_path+" 不存在，将创建新文件。")
        df1 = pd.DataFrame([list_auc])
        df1.to_excel(table_path, index=False, header=False)
        exit()
    df2 = pd.DataFrame([list_auc])
    existing_df = pd.read_excel(table_path, header=None)
    combined_df = pd.concat([existing_df, df2], ignore_index=True)
    combined_df.to_excel(table_path, index=False, header=False)


    # 关闭 SummaryWriter
    # writer.close()  
    if verbose:
        print('Done.')
        print()


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

class Projection1(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 activation: str = None,
                 bias: bool = True,
                 layer_norm: bool = False,
                 dropout: float = 0.0):
        """
        投影模型 (Model1)
        
        参数:
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        activation: 激活函数类型 [None, 'relu', 'gelu', 'sigmoid', 'tanh']
        bias: 是否使用偏置项
        layer_norm: 是否使用Layer Normalization
        dropout: Dropout概率 (0.0表示不使用)
        """
        super(Projection1, self).__init__()
        
        # 核心投影层
        self.projection = nn.Linear(input_dim, output_dim, bias=bias)
        
        # 激活函数
        self.activation = None
        if activation:
            activation = activation.lower()
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'gelu':
                self.activation = nn.GELU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 应用投影
        x = self.projection(x)
        
        # 应用激活函数 (如果有)
        if self.activation:
            x = self.activation(x)
        
        # 应用Layer Normalization (如果有)
        if self.layer_norm:
            x = self.layer_norm(x)
        
        # 应用Dropout (如果有)
        if self.dropout:
            x = self.dropout(x)
            
        return x
    
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