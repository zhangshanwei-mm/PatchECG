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

def aggregate_diagnostic(y_dic): # 
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))

def encode_labels(data, label_to_index):

    target = torch.zeros((len(data), len(label_to_index))) # , dtype=torch.float32
    

    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target

def gen_data_from_pic_knn(save_path,fold):

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
    
    test_fold = (2 * fold)
    vaild_fold =  (2 * fold - 1) # =======================================
    
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
    xtrain = np.array(train_delete_data) # 
    xtrain = np.moveaxis(xtrain, 1, 2)

    for i in range(len(xtrain)):
        tmp_data = xtrain[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        xtrain[i] = (tmp_data - tmp_mean) / tmp_std
    ytrain = encode_labels(train_delete_label, label_to_index)

    print(xtrain.shape)
    print(ytrain.shape)
    
    #  /data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5
    with h5py.File('/data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5', 'r') as f:
        print(f.keys())
        xtest = f['signals'][:] # 
        ytest_ndarray = f['labels'][:] # 
    xtest[xtest == 0] = np.nan
    

    for i in range(len(xtest)):
        tmp_data = xtest[i]
        tmp_std = np.nanstd(tmp_data)
        tmp_mean = np.nanmean(tmp_data)
        xtest[i] = (tmp_data - tmp_mean) / tmp_std
    
    
    # xtest = np.moveaxis(xtest, 1, 2)
    ytest =torch.from_numpy(ytest_ndarray)
    print("xtest shape:",xtest.shape) # (21388, 12, 1000)
    print("ytest shape:",ytest.shape)
    
    # knn 插补
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
    knn_xtest = np.nan_to_num(knn_xtest, nan=0.0)
    
    
    # 保存数据
    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("signal", data=knn_xtest)
        hf.create_dataset("labels", data=ytest_ndarray)
        
    print("数据保存成功！")
    
if __name__ == '__main__':
    path = "./data/4fold_knn_from_pic.h5"
    
    gen_data_from_pic_knn(path,fold=4)
    # nohup python -u gen_from_pic_knn.py > ./data/logs/imputation_knn_1fold 2>&1 &
        
    