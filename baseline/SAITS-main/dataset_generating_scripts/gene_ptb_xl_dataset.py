"""
The script for generating ETT-m1 dataset.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""


# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


import argparse
import os
import sys,wfdb,ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    window_truncate,
    random_mask,
    add_artificial_mask,
    saving_into_h5,
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

def aggregate_diagnostic(y_dic): # 需要修改
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pyb-xl dataset")
    # parser.add_argument("--file_path", help="path of dataset file", type=str) # 自定义path读取ptb-xl
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
    
    
    args = parser.parse_args()
    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)
    
    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
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
    
    test_fold = 2
    vaild_fold = 1 # =======================================
    
    X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != vaild_fold))]
    y_train = Y[((Y.strat_fold != test_fold)& (Y.strat_fold != vaild_fold))].diagnostic_superclass
    
    X_vaild = X[np.where(Y.strat_fold == vaild_fold)]
    y_vaild = Y[Y.strat_fold == vaild_fold].diagnostic_superclass
    
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    
    
    # add missing values in train set manually
    scaler = StandardScaler()
    train_ndarray = np.array(X_train)
    val_set_X = np.array(X_vaild)
    
    # add ours layout 
    
    
    test_set_X = np.array(X_test) # n,1000,12 [sample,len,feature]
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
    test_set_dict = add_artificial_mask(test_set_X, args.artificial_missing_rate, "test")
    
    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
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
    
    