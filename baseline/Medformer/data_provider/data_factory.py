from data_provider.data_loader import (
    APAVALoader,
    ADFTDLoader,
    DependentLoader,
    TDBRAINLoader,
    PTBLoader,
    PTBXLLoader,
    FLAAPLoader,
    UCIHARLoader,
    my_dataLoader,
)
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import pandas as pd
from new_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import h5py
data_dict = {
    # Subject-Dependent setup
    "ADFTD-Dependent": DependentLoader,  # dataset ADFTD with subject-dependent setup

    # Subject-Independent setup
    "APAVA": APAVALoader,  # dataset APAVA
    "TDBRAIN": TDBRAINLoader,  # dataset TDBRAIN
    "ADFTD": ADFTDLoader,  # dataset ADFTD
    "PTB": PTBLoader,  # dataset PTB
    "PTB-XL": PTBXLLoader,  # dataset PTB-XL
    "FLAAP": FLAAPLoader,  # dataset FLAAP
    "UCI-HAR": UCIHARLoader,  # dataset HAR
    "my_data": my_dataLoader,  # dataset my_data
}

def encode_labels(data, label_to_index):

    target = torch.zeros((len(data), len(label_to_index)), dtype=torch.float32)
    
 
    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target


def data_provider(args, flag): # args.data = ,flag = TRAIN,VALID,TEST
    Data = data_dict[args.data] # Data = my_dataLoader
    timeenc = 0 if args.embed != "timeF" else 1

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    input_data = None
    input_label = None

    
    if args.task_name == "anomaly_detection":
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
    elif args.task_name == "classification":
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            args=args,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(
                x, max_len=args.seq_len
            ),  # only called when yeilding batches
        )
        return data_set, data_loader  
    elif args.task_name == "my_classification":
        drop_last = False
        print(flag)
        database_path = "/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
        # df = pd.read_csv(database_path)
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
        vaild_fold = 9
        
        if flag == "TRAIN":

            # Train
            X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != vaild_fold))]
            y_train = Y[((Y.strat_fold != test_fold)& (Y.strat_fold != vaild_fold))].diagnostic_superclass
            
            train_delete_data = list()
            train_delete_label = list()
            

            num_train = 0 
            for i ,j in zip(y_train,X_train):
                temp_data = j
                temp_label = i
                if not temp_label:
                    num_train=num_train+1
                    continue
                train_delete_data.append(temp_data)
                train_delete_label.append(temp_label)

            
            # my data augmentation
            xtrain = np.array(train_delete_data) # 
            xtrain = np.moveaxis(xtrain, 1, 2)
            
            
            input_label = encode_labels(train_delete_label, label_to_index)
            
            input_data =  np.array(train_delete_data)
            
            
        elif flag == "VAL":
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

            vaild_label = encode_labels(val_delete_label, label_to_index)
            
            # ==== 添加random 排布
            val_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(val_delete_data))
            temp_val_data = list()
            for i in range(len(val_delete_data)):
                temp_layout_data = gen_true_ecg_layout(val_delete_data[i],length = 1000,layout = val_list[i])
                temp_val_data.append(temp_layout_data)
            # =====
            
            
            input_data = np.array(temp_val_data) # val_delete_data
            input_label = vaild_label
    
        elif flag == "TEST": 

            # 
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

            # test_label = encode_labels(test_delete_label, label_to_index)
            
            # ==== 1、add random layout
            # test_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(test_delete_data))
            # temp_test_data = list()
            # for i in range(len(test_delete_data)):
            #     temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = test_list[i])
            #     temp_test_data.append(temp_layout_data)
                
            # ====
            
            
            # =============================
            # ==== 2、 add from picture
            # with h5py.File('/data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5', 'r') as f:
            #     temp_test_data = f['signals'][:]
            #     test_label = f['labels'][:]
            # read from picture.h5
            # ==== 2、 add from picture
            # =============================
            
            # =============================
            # ==== 3、 add from picture ,ours generate ecg old version
            # /data/0shared/zhangshanwei/cinc/ours/data/10fold_c4_f1.h5
            # with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/10fold_random.h5', 'r') as f:
            #     temp_test_data = f['signals'][:]
            #     test_label = f['labels'][:]
            
            # ==== 3、 add from picture,ours generate ecg old version
            # =============================
            
            # =============================
            # ====4、add chaoyang hospital AF data ====
            with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_6x2.h5', 'r') as f:
                data1 = f['data'][:]
                label1 = f['labels'][:]
                numbers1 = f['numbers'][:]
            # read 12x1 AF 100 
            with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/chaoyan_AF_12x1.h5', 'r') as f:
                data2 = f['data'][:]
                label2 = f['labels'][:]
                numbers2 = f['numbers'][:]
            # read NAF 200 
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

            knn_xtest = np.concatenate((data1, data2, data3), axis=0) # [400,1000,12]
            test_label = torch.cat([test_label1, test_label2, test_label3], axis=0) # [400,23]

            # print(all_data.shape) # [400,1000,12]
            # print(all_label.shape) # [400,23]
            for i in range(len(knn_xtest)): # 标准化
                tmp_data = knn_xtest[i]
                tmp_std = np.nanstd(tmp_data)
                tmp_mean = np.nanmean(tmp_data)
                knn_xtest[i] = (tmp_data - tmp_mean) / tmp_std
                
            knn_xtest = np.nan_to_num(knn_xtest, nan=0.0)
            knn_xtest = np.moveaxis(knn_xtest, 1, 2)

            # ====4、add 朝阳医院数据 ====      
            # =============================
            



            # xtest = np.array(temp_test_data)
            # xtest = np.moveaxis(xtest, 1, 2)
            # input_data = np.moveaxis(xtest, 1, 2) # np.array(temp_test_data) # test_delete_data


            input_data = knn_xtest
            input_label = test_label 
            
                
        data_set = Data(
            data = input_data,
            label = input_label,
            root_path=args.root_path,
            args=args,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(
                x, max_len=args.seq_len
            ),  
        )
        return data_set, data_loader 
    else:
        if args.data == "m4":
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
