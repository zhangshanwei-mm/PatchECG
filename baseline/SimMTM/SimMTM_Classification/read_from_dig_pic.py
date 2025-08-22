import pandas as pd
import numpy as np
import torch,wfdb,os,ast
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


path = '/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/'
sampling_rate=100
agg_df = pd.read_csv('/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/'+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]
np.random.seed(40)

def load_raw_data(df, sampling_rate, path): # ours data path
    if sampling_rate == 100:
    
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        # print(wfdb.rdsamp("/data/zhangshanwei/output_path/"+f)
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

def read_ecg_data():
    # add ours dataset 
    database_path = "/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/"
    # /data/zhangshanwei/output_path
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
    
    X_train = X
    y_train = Y.diagnostic_superclass
    
    # print(X_train.shape) # 21799,1000,21 data 
    # print(y_train.shape) # 21799,
    

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


    exit()

    xtrain = np.array(train_delete_data) # 使用删除后的数据
    xtrain = np.moveaxis(xtrain, 1, 2)
    ytrain = encode_labels(train_delete_label, label_to_index)

    # print(xtrain.shape)
    # print(ytrain.shape) # 21388
    # print(ytrain[0])
    
    # 保存数据

    with h5py.File('/data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5', 'w') as f:
        f.create_dataset('signals', data=xtrain, compression='gzip')
        f.create_dataset('labels', data=ytrain, compression='gzip')
    # 会强制转为ndarray的数据类型
    
    # print("数据保存成功！")


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
    plt.savefig("./ecg_12lead_spline.png")

def quadratic_spline_impute(signal):
    """
    对一维信号（含 NaN）进行二次样条插值填补。
    :param signal: 一维数组，形状为 (1000,)
    :return: 填补后的信号
    """
    # 提取有效数据点（非 NaN 的索引和值）
    valid_indices = np.where(~np.isnan(signal))[0]
    valid_values = signal[valid_indices]
    
    # 若有效数据点不足 3 个，无法拟合二次样条，返回原信号（或自定义处理）
    if len(valid_indices) < 3:
        return signal  # 或抛出警告、使用其他方法填补
    
    # 创建二次样条插值器（k=2 表示二次样条）
    spline = UnivariateSpline(valid_indices, valid_values, k=2, s=0)  # s=0 表示无平滑
    
    # 生成插值结果（覆盖所有 1000 个时间点）
    interpolated = spline(np.arange(1000))
    
    # 将插值结果替换原信号中的 NaN
    signal_filled = np.copy(signal)
    nan_indices = np.where(np.isnan(signal))[0]
    signal_filled[nan_indices] = interpolated[nan_indices]
    
    return signal_filled

def impute_ecg_data(data):
    """
    处理三维 ECG 数据，维度为 [sample, 12, 1000]
    :param data: 输入数据，含 NaN
    :return: 填补后的数据
    """
    filled_data = np.zeros_like(data)
    
    # 遍历每个样本
    for sample_idx in range(data.shape[0]):
        # 遍历每个导联（12 个）
        for lead_idx in range(data.shape[1]):
            # 提取当前导联的时间序列
            signal = data[sample_idx, lead_idx, :]
            # 使用二次样条插值填补
            filled_signal = quadratic_spline_impute(signal)
            filled_data[sample_idx, lead_idx, :] = filled_signal
    
    return filled_data


def spline_interpolate(signal):
    valid_indices = np.where(~np.isnan(signal))[0]
    valid_values = signal[valid_indices]
    spline = UnivariateSpline(valid_indices, valid_values, k=2, s=0)  # 二次样条
    return spline(np.arange(len(signal)))


if __name__ == '__main__':
    # read_ecg_data()
    
    # 读取数据
    # with h5py.File('/data/0shared/zhangshanwei/data/2024cinc/from_picture_to_signal.h5', 'r') as f:
    #     signals = f['signals'][:]
    #     labels = f['labels'][:]
    
    # print(signals.shape)
    # print(labels.shape)
    
    # print(labels[0])
    # print(type(signals))
    # print(type(labels))
    # plt_ecg_12lead(signals[0])
    
    
    
    
    # np.savetxt("./tesst2.csv", signals[2], delimiter=',', fmt='%s') 
    # problem 0 为nan，不是填充了nan的类型
    


    signal = np.array([1, np.nan, 3, np.nan, 5])
    filled_signal = spline_interpolate(signal)
    print(filled_signal)  # 输出：[1.0, 2.0, 3.0, 4.0, 5.0]
    
    