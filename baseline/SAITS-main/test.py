import h5py
import matplotlib.pyplot as plt
import numpy as np
fold = 5
layout = 6
with h5py.File("/data/0shared/zhangshanwei/cinc/baseline/SAITS-main/generated_datasets/ptb_xl_fold"+str(fold)+"_layout"+str(layout)+"/datasets.h5", "r") as hf:  # read data from h5 file
    # print(hf.keys()) <KeysViewHDF5 ['empirical_mean_for_GRUD', 'test', 'train', 'val']>
    # print(hf["test"].keys()) <KeysViewHDF5 ['X', 'X_hat', 'indicating_mask', 'labels', 'missing_mask']>
    X = hf['test']["X"][:]
    X_hat = hf['test']["X_hat"][:]
    missing_mask = hf['test']["missing_mask"][:]
    indicating_mask = hf['test']["indicating_mask"][:]
    labels = hf['test']["labels"][:]

with h5py.File("/data/0shared/zhangshanwei/cinc/baseline/SAITS-main/NIPS_results/ptb_SAITS_base/fold"+str(fold)+"_layout"+str(layout)+"/imputations.h5", "r") as f:  # read data from h5 file
    # print(f.keys()) # <KeysViewHDF5 ['imputed_test_set', 'imputed_train_set', 'imputed_val_set', 'labels']>
    X_layout = f['imputed_test_set'][:]

# print(X.shape) # 2138
# print(X_layout.shape) 2138
# print(labels[0]) label

row_test_date = X
row_test_data_layout = X_hat
imputated_data_layout = X_layout
test_label = labels



print(row_test_date.shape)
print(row_test_data_layout.shape)
print(imputated_data_layout.shape)
print(test_label.shape)

# print(row_test_date[0])
# print(row_test_data_layout[0])
# print(imputated_data_layout[0])
# print(test_label[0])
print("fold"+str(fold)+"_layout"+str(layout)+".h5")
# save as a new h5 file for us
save_path = "imputed_data/fold"+str(fold)+"_layout"+str(layout)+".h5"
with h5py.File('/data/0shared/zhangshanwei/cinc/baseline/SAITS-main/NIPS_results/ptb_SAITS_base/'+save_path, 'w') as p:
    p.create_dataset('data_raw', data=row_test_date, compression='gzip')
    p.create_dataset('data_layout', data=row_test_data_layout, compression='gzip')
    p.create_dataset('data_imputed', data=imputated_data_layout, compression='gzip')
    p.create_dataset('labels', data=test_label, compression='gzip')








