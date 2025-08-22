import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np

import pandas as pd
import numpy as np
import torch,wfdb,os,ast,h5py
from sklearn.impute import KNNImputer
import time
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

path = '/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/'
sampling_rate=100
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]
np.random.seed(40)

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic): # 需要修改
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))

def encode_labels(data, label_to_index):
# 初始化零矩阵，形状为[数据长度, 标签数量]
    target = torch.zeros((len(data), len(label_to_index))) # , dtype=torch.float32
    
    # 填充矩阵
    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target

def gen_true_ecg_layout(row_data,length = 1000,layout = 5):
    """
        row_data : 输入维度为[length,12]
        len : 输入的长度
        layout : 表示排布的方式
            0 : 3 x 4
            1 : 3 x 4 + Ⅱ
            2 : 3 x 4 + Ⅱ + V1
            3 : 2 x 6 
            4 : 2 x 6+Ⅱ
            5 : 12 
        return :
            [length,12] ndarray
    """
    
    
    data = np.array(row_data)
    
    if layout == 0:
        # 处理 3 x 4 的情况
        temp = length//4
        data[temp:length,0] = np.NaN
        data[temp:length,1] = np.NaN
        data[temp:length,2] = np.NaN
        
        data[0:temp,3] = np.NaN
        data[2*temp:length,3] = np.NaN
        data[0:temp,4] = np.NaN
        data[2*temp:length,4] = np.NaN
        data[0:temp,5] = np.NaN
        data[2*temp:length,5] = np.NaN

        data[0:2*temp,6] = np.NaN
        data[3*temp:length,6] = np.NaN
        data[0:2*temp,7] = np.NaN
        data[3*temp:length,7] = np.NaN
        data[0:2*temp,8] = np.NaN
        data[3*temp:length,8] = np.NaN        

        data[0:3*temp,9] = np.NaN
        data[0:3*temp,10] = np.NaN
        data[0:3*temp,11] = np.NaN
        return data
    
    elif layout == 1:
        # 处理 3 x 4 + Ⅱ 的情况
        temp = length//4
        data[temp:length,0] = np.NaN
        # data[temp:length,1] = np.NaN
        data[temp:length,2] = np.NaN
        
        data[0:temp,3] = np.NaN
        data[2*temp:length,3] = np.NaN
        data[0:temp,4] = np.NaN
        data[2*temp:length,4] = np.NaN
        data[0:temp,5] = np.NaN
        data[2*temp:length,5] = np.NaN

        data[0:2*temp,6] = np.NaN
        data[3*temp:length,6] = np.NaN
        data[0:2*temp,7] = np.NaN
        data[3*temp:length,7] = np.NaN
        data[0:2*temp,8] = np.NaN
        data[3*temp:length,8] = np.NaN        

        data[0:3*temp,9] = np.NaN
        data[0:3*temp,10] = np.NaN
        data[0:3*temp,11] = np.NaN
        return data
    elif layout == 2:
        # 处理 3 x 4 + Ⅱ + V1 的情况
        temp = length//4
        data[temp:length,0] = np.NaN
        # data[temp:length,1] = np.NaN
        data[temp:length,2] = np.NaN
        
        data[0:temp,3] = np.NaN
        data[2*temp:length,3] = np.NaN
        data[0:temp,4] = np.NaN
        data[2*temp:length,4] = np.NaN
        data[0:temp,5] = np.NaN
        data[2*temp:length,5] = np.NaN

        # data[0:2*temp,6] = np.NaN
        # data[3*temp:length,6] = np.NaN
        data[0:2*temp,7] = np.NaN
        data[3*temp:length,7] = np.NaN
        data[0:2*temp,8] = np.NaN
        data[3*temp:length,8] = np.NaN        

        data[0:3*temp,9] = np.NaN
        data[0:3*temp,10] = np.NaN
        data[0:3*temp,11] = np.NaN
        return data
    
    elif layout == 3:
        # 处理 2 x 6 的情况
        temp = length//2 # 
        data[temp:length,0] = np.NaN
        data[temp:length,1] = np.NaN
        data[temp:length,2] = np.NaN
        data[temp:length,3] = np.NaN
        data[temp:length,4] = np.NaN
        data[temp:length,5] = np.NaN
        
        data[0:temp,6] = np.NaN
        data[0:temp,7] = np.NaN
        data[0:temp,8] = np.NaN
        data[0:temp,9] = np.NaN
        data[0:temp,10] = np.NaN
        data[0:temp,11] = np.NaN
        return data
    
    elif layout == 4:
        # 处理 2 x 6 + Ⅱ 的情况
        temp = length//2 # 
        data[temp:length,0] = np.NaN
        # data[temp:length,1] = np.NaN
        data[temp:length,2] = np.NaN
        data[temp:length,3] = np.NaN
        data[temp:length,4] = np.NaN
        data[temp:length,5] = np.NaN
        
        data[0:temp,6] = np.NaN
        data[0:temp,7] = np.NaN
        data[0:temp,8] = np.NaN
        data[0:temp,9] = np.NaN
        data[0:temp,10] = np.NaN
        data[0:temp,11] = np.NaN
        return data
    elif layout == 5:
        return data
    else:
        raise ValueError("Invalid layout value")


def only_nan_interpolate(signal):
    # 筛选出 NaN 的位置
    is_nan = np.isnan(signal) # 1000
    valid_idx = np.where(~is_nan)[0].astype(np.int64)  # 非 NaN 的索引
    valid_vals = signal[valid_idx]                      # 非 NaN 的值
    
    # 只有当信号中的值为 NaN 时进行插补
    if len(valid_idx) > 0:
        # 创建一个与原始信号长度相同的输出数组
        
        output = np.copy(signal) # 1000 
        # 在 NaN 的位置进行二次样条插补
        spline = UnivariateSpline(
            x=valid_idx,
            y=valid_vals,
            k=2,
            s=1  # 设置平滑度，s=1 会使曲线更平滑
        )
        
        spline_data = spline(np.arange(len(signal)))
        
        
        print(spline(np.arange(len(signal))).shape) # 1000
        print(output.shape) # 
        print(is_nan.shape)
        output[is_nan] = spline_data[is_nan]
        
    else:
        # 如果没有 NaN 数据，则返回原数据
        output = signal.copy()
    
    return output



def plt_ecg_12lead(data,path):
    # 生成示例数据（假设形状为 (12, 1000) 的 ECG 信号）
    ecg_data = data  # 替换为实际数据

    # 导联名称（标准12导联名称）
    lead_names = [
        'I', 'II', 'III',
        'aVR', 'aVL', 'aVF',
        'V1', 'V2', 'V3',
        'V4', 'V5', 'V6'
    ]

    # 创建画布和子图布局
    plt.figure(figsize=(20, 12))  # 调整画布大小

    # 绘制每个导联的子图
    for i in range(12):
        plt.subplot(12, 1, i+1)  # 12行1列，第i+1个子图
        plt.plot(ecg_data[i], color='b', linewidth=0.8)  # 绘制第i个导联的信号
        
        # 隐藏坐标轴标签（除最后一个子图外）
        if i != 11:
            plt.xticks([])
        else:
            plt.xlabel('Time (samples)', fontsize=10)
        
        # 设置导联标题和坐标范围
        plt.ylabel('Amplitude', fontsize=8)
        plt.title(f'Lead {lead_names[i]}', loc='left', y=0.8, x=0.02, fontsize=10)
        plt.ylim(ecg_data[i].min()-1, ecg_data[i].max()+1)  # 调整纵轴范围

    # 全局标题和紧凑布局
    plt.suptitle('12-Lead ECG Signal', fontsize=14, y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # 调整子图垂直间距
    plt.savefig(path)
    
class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset_x,label, config, training_mode, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        X_train = torch.from_numpy(dataset_x) # numpy
        y_train = label # torch
        
        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        #X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples
        X_train = X_train[:, :1, :int(config.TSlength_aligned)]

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size *10
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size] 
            y_train = y_train[:subset_size]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def my_data(configs, training_mode, subset = True,layout = 0):
    """
    # 0 : 3 x 4
    # 1 : 3 x 4 + Ⅱ
    # 2 : 3 x 4 + Ⅱ + V1
    # 3 : 6 x 2 
    # 4 : 6 x 2+Ⅱ
    # 5 : 12 
    # 6 : 各种排布随机出现 -l --layout 
    """
    
    # add ours dataset 
    database_path = "/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
    sampling_rate = 100
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
    # print("Label to index mapping:")
    # print(label_to_index)
    
    test_fold = 2
    vaild_fold = 1
    
    # test 先试试完整的效果
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
    # print("删除test中缺失label的数量:"+str(num_test))
    test_label = encode_labels(test_delete_label, label_to_index)
    
    
    print("layout is :"+str(layout))
    
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
    # np.savetxt("./"+str(layout)+".csv", xtest[0], delimiter=',', fmt='%s')
    xtest = np.moveaxis(xtest, 1, 2)
    xtest = np.nan_to_num(xtest, nan=0.0)
    ytest = test_label
    
    # print(xtest.shape)
    # print(ytest.shape)
    
    if ytest.shape[0]>10*configs.target_batch_size:
        test_dataset = Load_Dataset(xtest,ytest, configs, training_mode, target_dataset_size=configs.target_batch_size*10, subset=subset)
    else:
        test_dataset = Load_Dataset(xtest,ytest, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)
    
    return test_loader

def data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset = True):

    # train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    # finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))
    # test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))
    """ Dataset notes:
    Epilepsy: train_dataset['samples'].shape = torch.Size([7360, 1, 178]); binary labels [7360] 
    valid: [1840, 1, 178]
    test: [2300, 1, 178]. In test set, 1835 are positive sampels, the positive rate is 0.7978"""
    """sleepEDF: finetune_dataset['samples']: [7786, 1, 3000]"""
    
    
    # add ours dataset 
    database_path = "/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
    sampling_rate = 100
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
    vaild_fold = 9 # =======================================
    
    X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != vaild_fold))]
    y_train = Y[((Y.strat_fold != test_fold)& (Y.strat_fold != vaild_fold))].diagnostic_superclass
    
    train_delete_data = list()
    train_delete_label = list()
    
    # 删除train没有label的样本
    num_train = 0 
    for i ,j in zip(y_train,X_train):
        temp_data = j
        temp_label = i
        if not temp_label:
            num_train=num_train+1
            continue
        train_delete_data.append(temp_data)
        train_delete_label.append(temp_label)
    print("删除train中缺失label的数量:"+str(num_train)) 
    
    # my data augmentation
    xtrain = np.array(train_delete_data) # 使用删除后的数据
    xtrain = np.moveaxis(xtrain, 1, 2)
    # 标准化
    for i in range(len(xtrain)):
        tmp_data = xtrain[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        xtrain[i] = (tmp_data - tmp_mean) / tmp_std
    ytrain = encode_labels(train_delete_label, label_to_index)

    # 标准化
    

    print(xtrain.shape)
    print(ytrain.shape)
    
    
    # val  
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
    
    # ==== 添加random 排布
    # val_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(val_delete_data))
    # temp_val_data = list()
    # for i in range(len(val_delete_data)):
    #     temp_layout_data = gen_true_ecg_layout(val_delete_data[i],length = 1000,layout = val_list[i])
    #     temp_val_data.append(temp_layout_data)
    # =====
    
    xval = np.array(val_delete_data)
    xval = np.moveaxis(xval, 1, 2)
    for i in range(len(xval)):
        tmp_data = xval[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        xval[i] = (tmp_data - tmp_mean) / tmp_std
        
    yval = vaild_label
    print(xval.shape)
    print(yval.shape)
    
    # test 先试试完整的效果
    # X_test = X[np.where(Y.strat_fold == test_fold)]
    # y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    # test_delete_data = list()
    # test_delete_label = list()
    
    # num_test = 0
    # for i,j in zip(X_test,y_test):
    #     temp_data = i
    #     temp_label = j

    #     if not temp_label:
    #         num_test = num_test+1
    #         continue
    #     test_delete_data.append(temp_data)
    #     test_delete_label.append(temp_label)
    # print("删除test中缺失label的数量:"+str(num_test))
    # test_label = encode_labels(test_delete_label, label_to_index)
    # ytest = test_label
    # knn_xtest = np.array(test_delete_data)
    
    # # 标准化
    # for i in range(len(knn_xtest)):
    #     tmp_data = knn_xtest[i]
    #     tmp_std = np.nanstd(tmp_data)
    #     tmp_mean = np.nanmean(tmp_data)
    #     knn_xtest[i] = (tmp_data - tmp_mean) / tmp_std
        
        
    
    
    # temp_test_data = list()
    
    # layout = 0 # === 1、不同排布
    
    # if layout != 6:
    #     for i in range(len(test_delete_data)):
    #         temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = layout)
    #         temp_test_data.append(temp_layout_data)
            
    # if layout == 6:
    #     test_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(test_delete_data))
    #     for i in range(len(test_delete_data)):
    #         temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = test_list[i])
    #         temp_test_data.append(temp_layout_data)
    
    # xtest = np.array(temp_test_data)
    # xtest = np.moveaxis(xtest, 1, 2)
    
    
    
    # ===============================
    # 2、saits 插补，read 插补过后的数据
    
    # data_asits_path = "/data/0shared/zhangshanwei/cinc/baseline/SAITS-main/NIPS_results/ptb_SAITS_base/imputed_data/fold5_layout6.h5"
    # with h5py.File(data_asits_path, "r") as f:  # read data from h5 file
    #     print(f.keys())
    #     xtest = f['data_imputed'][:]
    #     ytest_ndarray = f['labels'][:]
    # ytest =torch.from_numpy(ytest_ndarray)
    # 2、saits 插补，read 插补过后的数据
    # ================================
    
    
    # ================================
    # 3、加载signal from picture ======saits 插补
    # 3.1、在运行的时候进行knn插补，时间太长
    # 原始数据 /data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5
    
    # with h5py.File('/data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5', 'r') as f:
    #     print(f.keys())
    #     xtest = f['signals'][:] # 
    #     ytest_ndarray = f['labels'][:] # 
    # print(xtest.shape) 
    # np.savetxt('./011111.csv', xtest[0], delimiter=',', fmt='%.2f')
    # xtest[xtest == 0] = np.nan
    
    # saits 插补过后的数据
    # with h5py.File('/data/0shared/zhangshanwei/cinc/baseline/SAITS-main/NIPS_results/ptb_SAITS_base/ptb_xl_from_gen_pic/imputations.h5', 'r') as f:
    #     print(f.keys())
    #     xtest = f['imputed_test_set'][:] # 
    #     ytest_ndarray = f['labels'][:] #         
    
        
    # xtest = np.moveaxis(xtest, 1, 2)
    # ytest =torch.from_numpy(ytest_ndarray)
    # print("xtest shape:",xtest.shape) # (21388, 12, 1000)
    # print("ytest shape:",ytest.shape)
    
    # knn 插补
    # start_time = time.perf_counter()
    # data_2d_train = xtrain.reshape(xtrain.shape[0], -1) # n,12x1000
    # print("data_2d_train shape:",data_2d_train.shape)
    # data_2d_test = xtest.reshape(xtest.shape[0], -1) # n,12x1000
    # print("data_2d_test shape:",data_2d_test.shape)
    
    # imputer = KNNImputer(n_neighbors=5)
    # imputer.fit(data_2d_train) # 在train 拟合
    # data_2d_test_imputed = imputer.transform(data_2d_test) # data_test 插补
    
    # print("data_2d_test_imputed  shape:",data_2d_test_imputed .shape) # n,3000 ?
    
    # knn_xtest = data_2d_test_imputed.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2])
    
    # print("原始数据缺失值数量:", np.isnan(data_2d_test).sum())
    # print("插补后缺失值数量:", np.isnan(knn_xtest).sum())
    # # 计算运行时间
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"代码运行时间: {elapsed_time:.6f} 秒")
    # knn_xtest = np.nan_to_num(knn_xtest, nan=0.0)
    
    # for i in range(len(knn_xtest)): 
    #     tmp_data = knn_xtest[i]
    #     tmp_std = np.nanstd(tmp_data)
    #     tmp_mean = np.nanmean(tmp_data)
    #     knn_xtest[i] = (tmp_data - tmp_mean) / tmp_std
        
    
    
    
    # 3.2 knn插补，直接加载knn插补好的数据
    
    # with h5py.File('/data/0shared/zhangshanwei/cinc/baseline/SimMTM-main/SimMTM_Classification/data/5fold_knn_from_pic.h5', 'r') as f:
    #     print(f.keys())
    #     knn_xtest = f['signal'][:] # 
    #     ytest_ndarray = f['labels'][:] #   
         
    # knn_xtest[knn_xtest == 0] = np.nan # 确保无nan的数据
    # ytest = torch.from_numpy(ytest_ndarray) 
    
    # 3、加载signal from picture ======
    # ================================
    
    # ================================
    # 4、使用完整12-lead
    
    
    
    
    # 4、使用完整12-lead
    # ================================
    
    # ================================
    # 5.1、ours digitaization method and old generate ECG image 1.8w data
    # 5.1 /data/0shared/zhangshanwei/cinc/ours/data/from_ours_pic_and_dig.h5
    # or /data/0shared/zhangshanwei/cinc/ours/data/10fold_random.h5
    # with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/10fold_random.h5', 'r') as f:
    #     temp_test_data = f['signals'][:]
    #     test_label = f['labels'][:]
    
    # knn_xtest = np.array(temp_test_data)
    
    # # 标准化
    # for i in range(len(knn_xtest)):
    #     tmp_data = knn_xtest[i]
    #     tmp_std = np.nanstd(tmp_data)
    #     tmp_mean = np.nanmean(tmp_data)
    #     knn_xtest[i] = (tmp_data - tmp_mean) / tmp_std
        
    # knn_xtest = np.nan_to_num(knn_xtest, nan=0.0)
    # knn_xtest = np.moveaxis(knn_xtest, 1, 2)
    # ytest = torch.from_numpy(test_label) 
    # print(knn_xtest.shape) # 2173, 1000, 12
    # print(ytest.shape) # 2173, 23
    # 5.1、ours digitaization method and old generate ECG image 1.8w data
    # ================================
    

    # ================================
    # 5.2 、add 朝阳医院的数据
    with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_6x2.h5', 'r') as f:
        data1 = f['data'][:]
        label1 = f['labels'][:]
        numbers1 = f['numbers'][:]
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

    test_label1 = torch.from_numpy(label1)
    test_label2 = torch.from_numpy(label2)
    test_label3 = torch.from_numpy(label3)

    print(test_label1.shape) # [100,23]
    print(test_label2.shape) # [100,23]
    print(test_label3.shape) # [200,23]

    data1 = np.moveaxis(data1, 1, 2)
    data2 = np.moveaxis(data2, 1, 2)
    data3 = np.moveaxis(data3, 1, 2)

    xtest = np.concatenate((data1, data2, data3), axis=0) # [400,1000,12]
    ytest = torch.cat([test_label1, test_label2, test_label3], axis=0) # [400,23]


    # print(all_data.shape) # [400,1000,12]
    # print(all_label.shape) # [400,23]
    # for i in range(len(knn_xtest)): # 标准化
    #     tmp_data = knn_xtest[i]
    #     tmp_std = np.nanstd(tmp_data)
    #     tmp_mean = np.nanmean(tmp_data)
    #     knn_xtest[i] = (tmp_data - tmp_mean) / tmp_std
        
    # knn_xtest = np.nan_to_num(knn_xtest, nan=0.0)
    # print("***********")
    # print(knn_xtest.shape) # b,12,1000
    # print(ytest.shape)
    # print("***********")

    # 5.2 、add 朝阳医院的数据
    # ================================

    # ================================
    # 5.3 、add 朝阳医院的数据,SAits 插补
    # 读取插补后的数据
    # with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_6x2.h5', 'r') as f:
    #     label1 = f['labels'][:]
    #     numbers1 = f['numbers'][:]
    # # read 12x1 AF 100 张
    # with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_12x1.h5', 'r') as f:
    #     label2 = f['labels'][:]
    #     numbers2 = f['numbers'][:]
    # # read NAF 200 张
    # with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_NAF.h5', 'r') as f:
    #     label3 = f['labels'][:]
    #     numbers3 = f['numbers'][:]

    # test_label1 = torch.from_numpy(label1)
    # test_label2 = torch.from_numpy(label2)
    # test_label3 = torch.from_numpy(label3)

    # print(test_label1.shape) # [100,23]
    # print(test_label2.shape) # [100,23]
    # print(test_label3.shape) # [200,23]

    # ytest = torch.cat([test_label1, test_label2, test_label3], axis=0) # [400,23]

    # with h5py.File('/data/0shared/zhangshanwei/cinc/baseline/SAITS-main/NIPS_results/ptb_SAITS_base/ptb_xl_from_chaoyang_pic/imputations.h5', 'r') as f:
    #     print(f.keys())
    #     knn_xtest = f['imputed_test_set'][:] # 
    #     # ytest_ndarray = f['labels'][:] # 
    # knn_xtest = np.moveaxis(knn_xtest, 1, 2)
    # print(knn_xtest.shape) # b,1000,12

    # for i in range(len(knn_xtest)): # 标准化
    #     tmp_data = knn_xtest[i]
    #     tmp_std = np.nanstd(tmp_data)
    #     tmp_mean = np.nanmean(tmp_data)
    #     knn_xtest[i] = (tmp_data - tmp_mean) / tmp_std


    # print("***********")
    # print(knn_xtest.shape) # b,12,1000
    # print(ytest.shape)
    # print("***********")


    # 5.3 、add 朝阳医院的数据,SAits 插补
    # ================================



    # xtest = np.array(temp_test_data)
    # # 标准化
    # for i in range(len(xtest)):
    #     tmp_data = xtest[i]
    #     tmp_std = np.nanstd(tmp_data)
    #     tmp_mean = np.nanmean(tmp_data)
    #     xtest[i] = (tmp_data - tmp_mean) / tmp_std
    
    
    
    # knn 插补=============================
    xtest[xtest == 0] = np.nan
    start_time = time.perf_counter()
    data_2d_train = xtrain.reshape(xtrain.shape[0], -1) # n,12x1000
    print("data_2d_train shape:",data_2d_train.shape)
    data_2d_test = xtest.reshape(xtest.shape[0], -1) # n,12x1000
    print("data_2d_test shape:",data_2d_test.shape)
    
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(data_2d_train) # 在train 拟合
    data_2d_test_imputed = imputer.transform(data_2d_test) # data_test 插补
    
    print("data_2d_test_imputed  shape:",data_2d_test_imputed .shape) # n,3000 ?
    
    knn_xtest = data_2d_test_imputed.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2])
    
    print("原始数据缺失值数量:", np.isnan(data_2d_test).sum())
    print("插补后缺失值数量:", np.isnan(knn_xtest).sum())
    # 计算运行时间
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time:.6f} 秒")
    # knn_xtest = np.nan_to_num(knn_xtest, nan=0.0)
    # knn ================================
    
    for i in range(len(knn_xtest)): # 标准化
        tmp_data = knn_xtest[i]
        tmp_std = np.nanstd(tmp_data)
        tmp_mean = np.nanmean(tmp_data)
        knn_xtest[i] = (tmp_data - tmp_mean) / tmp_std
    
    # spline插补 ================================
    # start_time = time.perf_counter()
    # b = xtest[0]
    # a = xtest[0,0,:]
    # a = only_nan_interpolate(a) # 只插补第一个
    
    # b[0,250:1000] = a [250:1000]
    
    # b = np.nan_to_num(b, nan=0.0)
    # plt_ecg_12lead(b,"./ecg_12lead_spline_1.png")

    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"代码运行时间: {elapsed_time:.6f} 秒")
    # spline插补 ================================
    
    
    
    
    # ytest = test_label
    # b = xtest[0]
    # np.savetxt("./"+str(layout)+".csv", b, delimiter=',', fmt='%s')   
    # print(xtest.shape)
    # print(ytest.shape)
    

    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(xtrain, ytrain,configs, training_mode, target_dataset_size=configs.batch_size, subset=subset) # for self-supervised, the data are augmented here
    
    finetune_dataset = Load_Dataset(xval,yval, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset)
    if ytest.shape[0]>10*configs.target_batch_size: # xtest
        test_dataset = Load_Dataset(knn_xtest,ytest, configs, training_mode, target_dataset_size=configs.target_batch_size*10, subset=subset)
    else:
        test_dataset = Load_Dataset(knn_xtest,ytest, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset)



    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    """the valid and test loader would be finetuning set and test set."""
    valid_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader