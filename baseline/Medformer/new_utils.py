import pandas as pd
import numpy as np
import torch,wfdb,os,ast
import openpyxl
path = '/data/0shared/zhangshanwei/data/2024cinc/ptb-xl/physionet.org/files/ptb-xl/1.0.3/'
sampling_rate=100
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic): 
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))


def mask_temporal(data,mask_temporal_length):
    """
        args:
            data : [n,12,1000] ndarray
            mask_temporal_length : mask time length
            
        
        return 
            masked_data : [n,12,1000]
    """
    masked_data = data
    
    

    return masked_data

# mask lead
def mask_lead(data,mask_lead_num):
    """
        args:
            mask_lead_num : the number of mask lead
            data : [n,12,1000] ndarray
        
        return 
            masked_data : [n,12,1000]
    """
    
    masked_data = data 
    
    matrix = np.ones((data.shape[0], 12)) # [n,12]
    for i in range(data.shape[0]):
        
        indices_to_zero = np.random.choice(12, mask_lead_num, replace=False)
        matrix[i, indices_to_zero] = 0
            
    for i in range(data.shape[0]):
        for j in range(12):
            if matrix[i][j] == 0:
                masked_data[i,j,:] = np.nan
    
    return masked_data

def mask_lead_50(data, mask_lead_num):
    """
    Randomly masks `mask_lead_num` leads (columns) of each sample in the ECG data and shuffles 50% of the remaining leads.

    Args:
        mask_lead_num : the number of leads to mask in each sample (1-12).
        data : [n, 12, 1000] ndarray (ECG data with n samples, 12 leads, and 1000 time points).

    Returns:
        masked_data : [n, 12, 1000] ndarray (with masked leads set to NaN and 50% of remaining leads shuffled).
    """
    
    # Initialize the masked data as a copy of the original data
    masked_data = np.copy(data)
    
    # Create a matrix for mask selection, shape [n, 12]
    mask_matrix = np.ones((data.shape[0], 12), dtype=bool)
    
    # For each sample, randomly choose `mask_lead_num` leads to mask (set to NaN)
    for i in range(data.shape[0]):
        # Randomly select indices to mask
        indices_to_mask = np.random.choice(12, mask_lead_num, replace=False)
        mask_matrix[i, indices_to_mask] = False
    
    # Apply the mask: set the corresponding leads to NaN
    masked_data[~mask_matrix] = np.nan

    # Shuffle 50% of the remaining leads for each sample
    for i in range(data.shape[0]):
        remaining_indices = np.where(mask_matrix[i])[0]  # Get the remaining unmasked lead indices
        num_to_shuffle = len(remaining_indices) // 2    # Calculate 50% of remaining leads
        if num_to_shuffle > 0:
            shuffle_indices = np.random.choice(remaining_indices, num_to_shuffle, replace=False)
            shuffled_data = masked_data[i, shuffle_indices, :]
            np.random.shuffle(shuffled_data)  # Shuffle the data along the lead dimension
            masked_data[i, shuffle_indices, :] = shuffled_data
    
    return masked_data

def mask_lead_100(data, mask_lead_num):
    """
    Randomly masks `mask_lead_num` leads (columns) of each sample in the ECG data and shuffles 100% of the remaining leads (which can appear in any position).

    Args:
        mask_lead_num : the number of leads to mask in each sample (1-12).
        data : [n, 12, 1000] ndarray (ECG data with n samples, 12 leads, and 1000 time points).

    Returns:
        masked_data : [n, 12, 1000] ndarray (with masked leads set to NaN and 100% of remaining leads shuffled).
    """
    
    # Initialize the masked data as a copy of the original data
    masked_data = np.copy(data)
    
    # Create a matrix for mask selection, shape [n, 12]
    mask_matrix = np.ones((data.shape[0], 12), dtype=bool)
    
    # For each sample, randomly choose `mask_lead_num` leads to mask (set to NaN)
    for i in range(data.shape[0]):
        # Randomly select indices to mask
        indices_to_mask = np.random.choice(12, mask_lead_num, replace=False)
        mask_matrix[i, indices_to_mask] = False
    
    # Apply the mask: set the corresponding leads to NaN
    masked_data[~mask_matrix] = np.nan

    # Shuffle 100% of the remaining leads for each sample and reassign positions
    for i in range(data.shape[0]):
        remaining_indices = np.where(mask_matrix[i])[0]  # Get the remaining unmasked lead indices
        if len(remaining_indices) > 0:
            shuffled_indices = np.random.permutation(remaining_indices)  # Get a shuffled order of the remaining indices
            shuffled_data = masked_data[i, remaining_indices, :]  # Extract the data to shuffle
            masked_data[i, remaining_indices, :] = 0  # Temporarily clear original positions
            for original_idx, shuffled_idx in zip(remaining_indices, shuffled_indices):
                masked_data[i, shuffled_idx, :] = shuffled_data[np.where(remaining_indices == original_idx)[0][0]]
    
    return masked_data


def mask_12channal_train(signal):
    """
        input:12,1000 ndarray
        return : 12,1000
    """
    
    length = 1000 # signal 的长度
    
    mask_window_size = np.random.randint(0, 750, size=12)
    mask_start = []
    for i in range(12):
        mask_start.append(np.random.randint(0, length-mask_window_size[i]))
        
    for i in range(12):
        signal[i,mask_start[i]:mask_start[i]+mask_window_size[i]] = np.nan
        
    return signal

def mask_12channal_test(signal):
    """
        input:12,1000 ndarray
        return : 12,1000
    """
    
    length = 1000 # signal 
    
    mask_window_size = np.random.randint(0, 500, size=12)
    mask_start = []
    for i in range(12):
        mask_start.append(np.random.randint(0, length-mask_window_size[i]))
        
    for i in range(12):
        signal[i,mask_start[i]:mask_start[i]+mask_window_size[i]] = np.nan
        
    return signal


def save_check(avg_auc, model1,model2,epoch,th):
    print('Model Saving...')
    torch.save({
        'model1': model1.state_dict(),
        'model2': model2.state_dict(),
        'Avg_AUC_Test': avg_auc,
        'Best_thresholds':th
    }, os.path.join('./last/add_s3/model/', 'checkpoint_'+str(epoch)+'.pth')) # 
    

def add_noisy(data,noise_std = 0.1):
    """
        input : 
            data :12,1000
            noise_ste : 
        return : 12,1000
    """
    length = 1000
    random_integers = np.random.randint(0, 11, size=12)
    rato_raw = random_integers*length/10 # 
    rato = rato_raw.astype(int)
    # print(rato)
    # [0,200,300,....,1000]
    start = [] # 
    for i in range(12):
        if rato[i]==1000: # 
            start.append(0)
            continue
        start.append(np.random.randint(0, length-rato[i]))
 
    
    
    # add noisy
    for i in range(12):
        if rato[i]==0:# 
            continue
        
        
        noise = np.random.normal(0, noise_std, size=(rato[i])) 
        
        data[i,start[i]:start[i]+rato[i]] += noise
    
    return data


def add_chnnel_noisy(data,noise_std = 0.1):
    """
        input : 
            data :12,1000
            noise_ste : 
        return : 12,1000
    """
    length = 1000
    random_array = np.random.randint(2, size=12) 

    for i in range(12):
        if random_array[i]==1:# add noisy
            noise = np.random.normal(0, noise_std, size=(1000))
            data[i] +=noise

    return data
    
    

def add_mask(data):
    """
        input : 12,1000
        return : 12,1000
    """
    
    length = 1000
    random_integers = np.random.randint(0, 9, size=12) # ex1 :11,ex4 :9
    rato_raw = random_integers*length/10 # 12
    rato = rato_raw.astype(int)
    # print(rato)
    # [0,200,300,....,1000]
    start = [] # 
    for i in range(12):
        if rato[i]==1000:
            start.append(0)
            continue
        start.append(np.random.randint(0, length-rato[i]))
    # print(start) 
    
    
    # add
    for i in range(12):
        if rato[i]==0:# 
            continue
        
        data[i,start[i]:start[i]+rato[i]] = np.nan
    
    return data


def shuffle_lead(data,channel,nums):
    """
        data :
        channel : numbers of input channel
        nums ：
    """
    if nums==0:
        return data
    
    channels = np.random.choice(channel, nums, replace=False)
    shuffled_channels = np.random.permutation(channels)
    data[channels] = data[shuffled_channels]
    
    return data

# 
def gen_true_ecg_layout(row_data,length = 1000,layout = 5):
    """
        row_data : [length,12]
        len : input length
        layout : lyaouts
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
        # 3 x 4 
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
    
# save data
def write_to_excel(data, filename='output.xlsx'):
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Sheet1'  
    

    for i, row in enumerate(data):

        for j, value in enumerate(row):
            ws.cell(row=i+1, column=j+1, value=value)  
    
    wb.save(filename)