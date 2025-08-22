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
import sys
from scipy import signal
from tqdm import tqdm
from helper_code import *
from utils import *
from new_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import pickle
import math
from my_transformer import *
import seaborn as sns
import ast
import json
np.set_printoptions(threshold=np.inf)
np.random.seed(40) # 45
substring_labels = '# Labels:'



def gen_sub_sex_eva():
    # read data
    database_path = "/data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
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
    
    # Train
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
    
    train_label = encode_labels(train_delete_label, label_to_index)
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    # 区分年龄
    sex_test = Y[Y.strat_fold == test_fold].sex
    
    test_delete_data = list()
    test_delete_label = list()
    test_delete_sex = list()
    
    num_test = 0
    for i,j,k in zip(X_test,y_test,sex_test):
        temp_data = i
        temp_label = j
        temp_sex = k
        if not temp_label:
            num_test = num_test+1
            continue
        test_delete_data.append(temp_data)
        test_delete_label.append(temp_label)
        test_delete_sex.append(temp_sex)
    print("删除val中缺失label的数量:"+str(num_test))
    test_label = encode_labels(test_delete_label, label_to_index)
    
    
    
    # make test 
    # test_delete_data 
    # test_label 
    # test_delete_sex 
    # 0 女性，1男性
    female = 0
    male = 0
    
    test_female_data = list()
    test_female_label = list()
    
    test_male_data = list()
    test_male_label = list()
    for i in range(test_label.shape[0]):
        if test_delete_sex[i]==0:
            female = female+1
            test_female_data.append(test_delete_data[i])
            test_female_label.append(test_label[i])
            continue
        male = male+1
        test_male_data.append(test_delete_data[i])
        test_male_label.append(test_label[i])
    
    # 修改维度
    x_female = np.array(test_female_data)
    x_female = np.moveaxis(x_female, 1, 2)
    
    x_male = np.array(test_male_data)
    x_male = np.moveaxis(x_male, 1, 2)
    # mask interlal lead
    x_1 = list()
    for i in range(x_female.shape[0]):
        temp_data = mask_12channal_test(x_female[i])
        x_1.append(temp_data)
    female_set = np.array(x_1)
    
    x_2 = list()
    for i in range(x_male.shape[0]):
        temp_data = mask_12channal_test(x_male[i])
        x_2.append(temp_data)
    male_set = np.array(x_2)
    # mask lead
    
    permutation_female = np.random.permutation(female_set.shape[1])
    female_set = female_set[:,permutation_female,:]
    
    permutation_male = np.random.permutation(male_set.shape[1])
    male_set = male_set[:,permutation_male,:]

    
    dic_val = {}
    dic_val["data"] = female_set
    dic_val["label"] = test_female_label
    torch.save(dic_val, './experment/data/eva_sub_female.pt')
    
    dic_test = {}
    dic_test["data"] = male_set
    dic_test["label"] = test_male_label
    torch.save(dic_test, './experment/data/eva_sub_male.pt')
    
    print(female)
    print(male)
    
def gen_super_sex_eva():
    # read data
    database_path = "/data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
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

    labels_dic = ['NORM','MI','STTC','CD','HYP']
    label_to_index = {label: idx for idx, label in enumerate(labels_dic)}
    print("Label to index mapping:")
    print(label_to_index)
    
    # Test
    test_fold = 10
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    test_label = encode_labels(y_test, label_to_index)
    sex_test = Y[Y.strat_fold == test_fold].sex
    
    sex = sex_test.tolist()
    female = 0
    male = 0

    test_female_data = list()
    test_female_label = list()
    
    test_male_data = list()
    test_male_label = list()
    for i in range(len(X_test)):
        if sex[i]==0:
            female = female+1
            test_female_data.append(X_test[i])
            test_female_label.append(test_label[i])
            continue
        male = male+1
        test_male_data.append(X_test[i])
        test_male_label.append(test_label[i])
    
    print(female)
    print(male)
    # 修改维度
    x_female = np.array(test_female_data)
    x_female = np.moveaxis(x_female, 1, 2)
    
    x_male = np.array(test_male_data)
    x_male = np.moveaxis(x_male, 1, 2)
    # mask interlal lead
    x_1 = list()
    for i in range(x_female.shape[0]):
        temp_data = mask_12channal_test(x_female[i])
        x_1.append(temp_data)
    female_set = np.array(x_1)
    
    x_2 = list()
    for i in range(x_male.shape[0]):
        temp_data = mask_12channal_test(x_male[i])
        x_2.append(temp_data)
    male_set = np.array(x_2)
    
    # mask lead
    
    permutation_female = np.random.permutation(female_set.shape[1])
    female_set = female_set[:,permutation_female,:]
    
    permutation_male = np.random.permutation(male_set.shape[1])
    male_set = male_set[:,permutation_male,:]

    

    dic_val = {}
    dic_val["data"] = female_set
    dic_val["label"] = test_female_label
    torch.save(dic_val, './experment/data/eva_super_female.pt')
    
    dic_test = {}
    dic_test["data"] = male_set 
    dic_test["label"] = test_male_label
    torch.save(dic_test, './experment/data/eva_super_male.pt')
    print("done!")

def encode_labels(data, label_to_index):
# 初始化零矩阵，形状为[数据长度, 标签数量]
    target = torch.zeros((len(data), len(label_to_index)), dtype=torch.float32)
    
    # 填充矩阵
    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target

def gen_super_noisy(): # ex1
    """
        每个导联随机0,10%....100%随机添加高斯噪声
        每个导联随机0,10%....100%随机缺失
        super标签实验1的验证集和测试集
        第9折验证集,第10折测试集
    """
    # read data
    database_path = "/data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
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

    labels_dic = ['NORM','MI','STTC','CD','HYP']
    label_to_index = {label: idx for idx, label in enumerate(labels_dic)}
    print("Label to index mapping:")
    print(label_to_index)
    
    test_fold = 8
    vaild_fold = 7
    

    # Train
    X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != vaild_fold))]
    y_train = Y[((Y.strat_fold != test_fold)& (Y.strat_fold != vaild_fold))].diagnostic_superclass
    train_label = encode_labels(y_train, label_to_index)
    
    print(X_train.shape)
    print(train_label.shape)
    
    # =====================================================
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    test_label = encode_labels(y_test, label_to_index)

    # vaild 
    X_vaild = X[np.where(Y.strat_fold == vaild_fold)]
    y_vaild = Y[Y.strat_fold == vaild_fold].diagnostic_superclass
    vaild_label = encode_labels(y_vaild, label_to_index)
    
    X_test = np.moveaxis(X_test,1,2)
    X_vaild = np.moveaxis(X_vaild,1,2)
    print(X_test.shape)
    print(X_vaild.shape)
    # 添加高斯噪声
    
    # test_data = X_test[0]
    
    # noisy_data = add_noisy(test_data,noise_std = 0.1)
    # mask_data = add_mask(noisy_data)
    
    # fig, axes = plt.subplots(12, 1, figsize=(12, 24), sharex=True)

    # # 绘制每个导联的数据
    # for i in range(12):
    #     axes[i].plot(mask_data[i])
    #     axes[i].set_title(f'Lead {i+1}')
    #     axes[i].set_ylabel('Amplitude (mV)')

    # # 设置共同的x轴标签
    # axes[-1].set_xlabel('Time (ms)')

    # # 调整子图之间的间距
    # plt.tight_layout()
    # plt.savefig("./test_mask&noisy.png")
    
    enhance_test = list()
    enhance_val = list()
    for i in range(X_test.shape[0]):
        temp = X_test[i]
        noisy_data = add_noisy(temp,noise_std = 0.1)
        mask_data = add_mask(noisy_data)
        enhance_test.append(mask_data)
        
    for i in range(X_vaild.shape[0]):
        temp = X_vaild[i]
        noisy_data = add_noisy(temp,noise_std = 0.1)
        mask_data = add_mask(noisy_data)
        enhance_val.append(mask_data)
    
    # 保存数据
    dic_val = {}
    dic_val["data"] = enhance_val
    dic_val["label"] = vaild_label
    torch.save(dic_val, './new_experiment/test_data/super_ex1_val_4fold.pt')
    
    dic_test = {}
    dic_test["data"] = enhance_test 
    dic_test["label"] = test_label
    torch.save(dic_test, './new_experiment/test_data/super_ex1_test_4fold.pt')
    
def gen_sub_noisy(): # ex1
    """
        每个导联随机0,10%....100%随机添加高斯噪声
        每个导联随机0,10%....100%随机缺失
        super标签实验1的验证集和测试集
        第9折验证集,第10折测试集
    """
    # read data
    database_path = "/data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
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
    
    test_fold = 8
    vaild_fold = 7
    
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
    
    # vaild 
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
    
    
    test_set = np.array(test_delete_data)
    val_set = np.array(val_delete_data)
    
    test_set = np.moveaxis(test_set,1,2)
    val_set = np.moveaxis(val_set,1,2)
    
    
    enhance_test = list()
    enhance_val = list()
    for i in range(test_set.shape[0]):
        temp = test_set[i]
        noisy_data = add_noisy(temp,noise_std = 0.1)
        mask_data = add_mask(noisy_data)
        enhance_test.append(mask_data)
        
    for i in range(val_set.shape[0]):
        temp = val_set[i]
        noisy_data = add_noisy(temp,noise_std = 0.1)
        mask_data = add_mask(noisy_data)
        enhance_val.append(mask_data)
    
    
    
    # 保存数据
    dic_val = {}
    dic_val["data"] = enhance_val
    dic_val["label"] = vaild_label
    torch.save(dic_val, './new_experiment/test_data/sub_ex1_val_4fold.pt')
    
    dic_test = {}
    dic_test["data"] = enhance_test 
    dic_test["label"] = test_label
    torch.save(dic_test, './new_experiment/test_data/sub_ex1_test_5fold.pt')
    
def gen_sub_ex2():
    """
        完整的通道随机添加高斯噪声0-11个lead
    """
    database_path = "/data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
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
    
    path1 = './new_experiment/ex2_data/sub_ex2_val_5fold.pt'
    path2 = './new_experiment/ex2_data/sub_ex2_test_5fold.pt'
    
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
    
    # vaild 
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
    
    
    test_set = np.array(test_delete_data)
    val_set = np.array(val_delete_data)
    
    test_set = np.moveaxis(test_set,1,2)
    val_set = np.moveaxis(val_set,1,2)
    
    
    enhance_test = list()
    enhance_val = list()
    for i in range(test_set.shape[0]):
        temp = test_set[i]
        noisy_data = add_chnnel_noisy(temp,noise_std = 0.1)
        enhance_test.append(noisy_data)
        
    for i in range(val_set.shape[0]):
        temp = val_set[i]
        noisy_data = add_chnnel_noisy(temp,noise_std = 0.1)
        enhance_val.append(noisy_data)
    
    # test_data = test_set[0]
    
    # noisy_data = add_chnnel_noisy(test_data,noise_std = 0.1)
    
    # fig, axes = plt.subplots(12, 1, figsize=(12, 24), sharex=True)

    # # 绘制每个导联的数据
    # for i in range(12):
    #     axes[i].plot(noisy_data[i])
    #     axes[i].set_title(f'Lead {i+1}')
    #     axes[i].set_ylabel('Amplitude (mV)')

    # # 设置共同的x轴标签
    # axes[-1].set_xlabel('Time (ms)')

    # # 调整子图之间的间距
    # plt.tight_layout()
    # plt.savefig("./test_mask&noisy.png")
    
    # # 保存数据
    dic_val = {}
    dic_val["data"] = enhance_val
    dic_val["label"] = vaild_label
    torch.save(dic_val, path1)
    
    dic_test = {}
    dic_test["data"] = enhance_test 
    dic_test["label"] = test_label
    torch.save(dic_test, path2)

def gen_super_ex2():
    """
        随机完整导联添加噪声
    """
        # read data
    database_path = "/data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
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

    labels_dic = ['NORM','MI','STTC','CD','HYP']
    label_to_index = {label: idx for idx, label in enumerate(labels_dic)}
    print("Label to index mapping:")
    print(label_to_index)
    
    test_fold = 10
    vaild_fold = 9
    

    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    test_label = encode_labels(y_test, label_to_index)

    # vaild 
    X_vaild = X[np.where(Y.strat_fold == vaild_fold)]
    y_vaild = Y[Y.strat_fold == vaild_fold].diagnostic_superclass
    vaild_label = encode_labels(y_vaild, label_to_index)
    
    X_test = np.moveaxis(X_test,1,2)
    X_vaild = np.moveaxis(X_vaild,1,2)
    print(X_test.shape)
    print(X_vaild.shape)

    enhance_test = list()
    enhance_val = list()
    for i in range(X_test.shape[0]):
        temp = X_test[i]
        noisy_data = add_chnnel_noisy(temp,noise_std = 0.1)
        enhance_test.append(noisy_data)
        
    for i in range(X_vaild.shape[0]):
        temp = X_vaild[i]
        noisy_data = add_chnnel_noisy(temp,noise_std = 0.1)
        enhance_val.append(noisy_data)
    
    # 保存数据
    dic_val = {}
    dic_val["data"] = enhance_val
    dic_val["label"] = vaild_label
    torch.save(dic_val, './new_experiment/ex2_data/super_ex2_val_5fold.pt')
    
    dic_test = {}
    dic_test["data"] = enhance_test 
    dic_test["label"] = test_label
    torch.save(dic_test, './new_experiment/ex2_data/super_ex2_test_5fold.pt')
    
def gen_sub_ex4():
    """
        每个导联随机0,10%....100%随机添加高斯噪声
        每个导联随机0,10%....80%随机缺失
        super标签实验1的验证集和测试集
        第9折验证集,第10折测试集
    """
    # read data
    database_path = "/data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
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
    
    # vaild 
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
    
    
    test_set = np.array(test_delete_data)
    val_set = np.array(val_delete_data)
    
    test_set = np.moveaxis(test_set,1,2)
    val_set = np.moveaxis(val_set,1,2)
    
    
    enhance_test = list()
    enhance_val = list()
    for i in range(test_set.shape[0]):
        temp = test_set[i]
        noisy_data = add_noisy(temp,noise_std = 0.1)
        mask_data = add_mask(noisy_data)
        enhance_test.append(mask_data)
        
    for i in range(val_set.shape[0]):
        temp = val_set[i]
        noisy_data = add_noisy(temp,noise_std = 0.1)
        mask_data = add_mask(noisy_data)
        enhance_val.append(mask_data)
    
    
    
    # 保存数据
    dic_val = {}
    dic_val["data"] = enhance_val
    dic_val["label"] = vaild_label
    torch.save(dic_val, './new_experiment/ex4_data/sub_ex4_val_5fold.pt')
    
    dic_test = {}
    dic_test["data"] = enhance_test 
    dic_test["label"] = test_label
    torch.save(dic_test, './new_experiment/ex4_data/sub_ex4_test_5fold.pt')
 
def gen_super_ex4():
    """
        每个导联随机0,10%....100%随机添加高斯噪声
        每个导联随机0,10%....80%随机缺失
        super标签实验1的验证集和测试集
        第9折验证集,第10折测试集
    """
    # read data
    database_path = "/data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
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

    labels_dic = ['NORM','MI','STTC','CD','HYP']
    label_to_index = {label: idx for idx, label in enumerate(labels_dic)}
    print("Label to index mapping:")
    print(label_to_index)
    
    test_fold = 8
    vaild_fold = 7
    

    # Train
    X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != vaild_fold))]
    y_train = Y[((Y.strat_fold != test_fold)& (Y.strat_fold != vaild_fold))].diagnostic_superclass
    train_label = encode_labels(y_train, label_to_index)
    
    print(X_train.shape)
    print(train_label.shape)
    
    # =====================================================
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    test_label = encode_labels(y_test, label_to_index)

    # vaild 
    X_vaild = X[np.where(Y.strat_fold == vaild_fold)]
    y_vaild = Y[Y.strat_fold == vaild_fold].diagnostic_superclass
    vaild_label = encode_labels(y_vaild, label_to_index)
    
    X_test = np.moveaxis(X_test,1,2)
    X_vaild = np.moveaxis(X_vaild,1,2)
    print(X_test.shape)
    print(X_vaild.shape)
    # 添加高斯噪声
    
    # test_data = X_test[0]
    
    # noisy_data = add_noisy(test_data,noise_std = 0.1)
    # mask_data = add_mask(noisy_data)
    
    # fig, axes = plt.subplots(12, 1, figsize=(12, 24), sharex=True)

    # # 绘制每个导联的数据
    # for i in range(12):
    #     axes[i].plot(mask_data[i])
    #     axes[i].set_title(f'Lead {i+1}')
    #     axes[i].set_ylabel('Amplitude (mV)')

    # # 设置共同的x轴标签
    # axes[-1].set_xlabel('Time (ms)')

    # # 调整子图之间的间距
    # plt.tight_layout()
    # plt.savefig("./test_mask&noisy.png")
    
    enhance_test = list()
    enhance_val = list()
    for i in range(X_test.shape[0]):
        temp = X_test[i]
        noisy_data = add_noisy(temp,noise_std = 0.1)
        mask_data = add_mask(noisy_data)
        enhance_test.append(mask_data)
        
    for i in range(X_vaild.shape[0]):
        temp = X_vaild[i]
        noisy_data = add_noisy(temp,noise_std = 0.1)
        mask_data = add_mask(noisy_data)
        enhance_val.append(mask_data)
    
    # 保存数据
    dic_val = {}
    dic_val["data"] = enhance_val
    dic_val["label"] = vaild_label
    torch.save(dic_val, './new_experiment/ex4_data/super_ex4_val_4fold.pt')
    
    dic_test = {}
    dic_test["data"] = enhance_test 
    dic_test["label"] = test_label
    torch.save(dic_test, './new_experiment/ex4_data/super_ex4_test_4fold.pt')
    
    

# 生成我们所需要的数据
def gen_sub_():
    
    print(1)
if __name__ == "__main__":
    
    
    gen_sub_()
    