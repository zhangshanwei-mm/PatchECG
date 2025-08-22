"""
The script for generating ptb-xl dataset.
"""

import argparse
import os
import sys,wfdb,ast,torch,h5py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    window_truncate,
    random_mask,
    random_mask_ours,
    add_artificial_mask,
    add_artificial_mask_ours,
    saving_into_h5,
    saving_into_h5_ours,
)

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
def gen_true_ecg_layout(row_data,length = 1000,layout = 5):
    """
        row_data : [length,12]
        len : 
        layout : 
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
        #  3 x 4 
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
        #  3 x 4 + Ⅱ 
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
        #  3 x 4 + Ⅱ + V1 
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
        #  2 x 6 
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
        #  2 x 6 + Ⅱ 
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pyb-xl dataset")

    parser.add_argument( # add missing_rate 
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument( # save_path
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    parser.add_argument( #  
        "--fold",type=int,help="5 fold ",default=1
    )
    
    parser.add_argument(
        "--layout",type=int,help="layout of the dataset",default=0
    )
    
    args = parser.parse_args() # 
    # save
    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)
    
    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating"+str(args.fold)+"_"+str(args.layout)+".log"),
        "Generate ptb-xl dataset",
        mode="w",
    )
    logger.info(args)    
    
    # load ptb-xl dataset

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
    
    test_fold = 2*args.fold
    vaild_fold = 2*args.fold-1 # =======================================
    
    X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != vaild_fold))]
    y_train = Y[((Y.strat_fold != test_fold)& (Y.strat_fold != vaild_fold))].diagnostic_superclass
    
    X_vaild = X[np.where(Y.strat_fold == vaild_fold)]
    y_vaild = Y[Y.strat_fold == vaild_fold].diagnostic_superclass
    
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    
    # delete test no label
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
    print("删除test中缺失label的数量:"+str(num_test))
    # test_dlete_data
    # test_label = encode_labels(test_delete_label, label_to_index)
    
    # add layout
    layout = args.layout
    # temp_test_data = list()
    # if layout != 6:
    #     for i in range(len(test_delete_data)):
    #         temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = layout)
    #         temp_test_data.append(temp_layout_data)
            
    # if layout == 6:
    #     test_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(test_delete_data))
    #     for i in range(len(test_delete_data)):
    #         temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = test_list[i])
    #         temp_test_data.append(temp_layout_data)
    # ours_X = np.array(test_delete_data) # 原始数据
    # layout_X = np.array(temp_test_data) # different layout test set
    
    # =================
    # 2、signal from pic
    # with h5py.File('/data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5', 'r') as f:
    #     xtest = f['signals'][:]
    #     test_label = f['labels'][:]
    
    # print("=====text label shape:",test_label.shape)
    
    # 2、signal from pic
    # =================   


    # ================
    # 3、signal from chaoyang
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
    test_label = torch.cat([test_label1, test_label2, test_label3], axis=0) # [400,23]
    # 3、signal from chaoyang
    # ================

    
    xtest = np.moveaxis(xtest, 1, 2)
    ours_X = np.array(xtest) # 

    layout_X = np.array(xtest) # different layout test set
    scaler = StandardScaler()
    
    train_ndarray = np.array(X_train)
    val_set_X = np.array(X_vaild)
    # np.savetxt("./a_x.csv", ours_X[0], delimiter=',', fmt='%s')
    # np.savetxt("./a_layout_x.csv", layout_X[0], delimiter=',', fmt='%s')
    # add ours layout
    if args.artificial_missing_rate > 0:
        train_set_X_shape = train_ndarray.shape
        train_set_X = train_ndarray.reshape(-1)
        indices = random_mask(train_set_X, args.artificial_missing_rate)
        train_set_X[indices] = np.nan
        train_set_X = train_set_X.reshape(train_set_X_shape)
        logger.info(
            f"Already masked out {args.artificial_missing_rate * 100}% values in train set"
        )
    
    train_set_dict = add_artificial_mask(train_set_X, args.artificial_missing_rate, "train")
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask_ours(ours_X,layout_X, test_label,args.artificial_missing_rate, "test")
    
    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, fold is {args.fold}'
    )
    logger.info(
        f'In test set, layout is {layout}'
    )
    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict, # 
    }
    

    train_sample_num = len(train_set_dict["X"])
    val_sample_num = len(val_set_dict["X"])
    test_sample_num = len(test_set_dict["X"])
    total_sample_num = train_sample_num + val_sample_num + test_sample_num
    feature_num = train_ndarray.shape[-1]
    logger.info(
        f"Feature num: {feature_num},\n"
        f"{train_sample_num} ({(train_sample_num / total_sample_num):.3f}) samples in train set\n"
        f"{val_sample_num} ({(val_sample_num / total_sample_num):.3f}) samples in val set\n"
        f"{test_sample_num} ({(test_sample_num / total_sample_num):.3f}) samples in test set\n"
    )

    
    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    pickle_dump(scaler, os.path.join(dataset_saving_dir, "scaler"))
    logger.info(f"All done. Saved to {dataset_saving_dir}.")
    
    