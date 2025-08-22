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
import ast

from models.Medformer import Model as MedformerModel


np.set_printoptions(threshold=np.inf)
np.random.seed(40) # 45

def encode_labels(data, label_to_index):
    target = torch.zeros((len(data), len(label_to_index)), dtype=torch.float32)
    
    
    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target

class MyDataset(Dataset): # 
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float))

    def __len__(self):
        return len(self.data)

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        #  p_t
        pt = torch.exp(-BCE_loss)
        
        #  Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class Configs:
    def __init__(self):
        self.task_name = "classification"
        self.pred_len = 0
        self.output_attention = False
        self.enc_in = 7
        self.single_channel = False
        self.patch_len_list = "2,2,4"
        self.seq_len = 1000 # 
        self.d_model = 128
        self.dropout = 0.1
        self.augmentations = "none" # jitter0.2,scale0.2,drop0.5
        self.num_class = 23
        self.e_layers = 6
        self.n_heads = 8
        self.d_layers = 1
        self.c_out = 7
        self.num_kernels = 6
        self.d_conv = 4
        self.no_inter_attn = False
        self.d_ff = 256
        self.activation = "gelu"

# Train classification model.
if __name__ == "__main__":
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
    
    test_fold = 10 # 8,6,4,2
    vaild_fold = 9 # 7,5,3,1
    
    # Train
    X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != vaild_fold))]
    y_train = Y[((Y.strat_fold != test_fold)& (Y.strat_fold != vaild_fold))].diagnostic_superclass
    
    train_delete_data = list()
    train_delete_label = list()
    
    # 
    num_train = 0 
    for i ,j in zip(y_train,X_train):
        temp_data = j
        temp_label = i
        if not temp_label:
            num_train=num_train+1
            continue
        train_delete_data.append(temp_data)
        train_delete_label.append(temp_label)
   
    
    train_label = encode_labels(train_delete_label, label_to_index)
    
    # 
    dic_val = torch.load('/data/zhangshanwei/code_path_ecg/new_experiment/ex4_data/sub_ex4_val_5fold.pt')
    dic_test = torch.load('/data/zhangshanwei/code_path_ecg/new_experiment/ex4_data/sub_ex4_test_5fold.pt')
    val_set = dic_val["data"]
    vaild_label = dic_val["label"]
    val_set = np.array(val_set)
    test_set = dic_test["data"]
    test_label = dic_test["label"]
    test_set = np.array(test_set)
    
    
    
    for i in range(val_set.shape[0]):
        permutation = np.random.permutation(val_set.shape[1])
        val_set[i] = val_set[i, permutation, :]
    
    for i in range(test_set.shape[0]):
        permutation = np.random.permutation(test_set.shape[1])
        test_set[i] = test_set[i, permutation, :]
        
    
    print("train data shape:",np.array(train_delete_data).shape) # (n,12,1000)
    print("train label shape:",train_label.shape) # (n,23)
    
    print("val data shape:",val_set.shape) # (n,12,1000)
    print("val label shape:",vaild_label.shape) # (n,23)
    
    # parameters
    n_epoch = 100
    batch_size = 64
    
    dataset_train = MyDataset(train_delete_data,train_label)
    dataset_val = MyDataset(val_set,vaild_label)
    dataset_test = MyDataset(test_set,test_label)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=True)
    
    print("Load data successfully!") 
    
    # Fit the model.
    device_str = "cuda:3"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    

    # make model
    para_config = Configs()
    model = MedformerModel(configs=para_config) 

    model.to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = MultiLabelFocalLoss()
    total_train_loss = [] # loss
    temp = []
    step = 0
    count = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
        # train
        model.train()
        prog_iter = tqdm(dataloader_train, desc="Training", leave=False , ncols=80)
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch) # [256, 12, 15, 64]     
            print(input_x.shape)
            print(input_y.shape)
            pred = model(input_x,None,None,None)
            
            print(pred.shape)
            
            exit()
            loss = loss_func(pred.float(), input_y.float())
                        
            exit()       


            PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [B x 12 x15,2,768]
            output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)
            output_x[mask] = 0
            # input transformer model 
            pred = model3(output_x)
            # loss = loss_func(pred, input_y.float())
            loss = loss_func(pred.float(), input_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            temp.append(loss.data.cpu())
            
        total_train_loss.append(np.mean(temp))
        print("第{}轮loss:".format(_+1),np.mean(temp))
        # writer.add_scalar('training loss', np.mean(temp), step) # add loss
        temp.clear()
        scheduler.step(_)


        # plt loss picture 
        plt.figure(figsize=(10, 5))
        plt.plot(total_train_loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig("./loss.png") 

        # val
        # model.eval()
        # prog_iter_val = tqdm(dataloader_val, desc="Val", leave=False)

        # all_pred_val = [] #  pred
        # all_y_val = [] #  true
        # all_val_score = [] #  score
        # with torch.no_grad():
        #     for batch_idx, batch in enumerate(prog_iter_val):
        #         input_x, input_y = tuple(t.to(device) for t in batch)
        #         all_y_val.append(input_y.cpu().data.numpy()) # \
                
        #         mask = torch.isnan(input_x).all(dim=-1) # B,C,N [256,12,15] 
        #         input_x[torch.isnan(input_x)] = 0 # \
        #         temp_x = input_x.view(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # 256 x 12 x15 
\
        #         channel_2_mask = torch.zeros_like(input_x)
        #         channel_2_mask[torch.isnan(input_x)] = 0.0
        #         channel_2_mask[~torch.isnan(input_x)] = 1.0
        #         channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
            
                
        #         PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [256 x 12 x15,1,768]
        #         # [256,12,15,768]
        #         output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)
        #         output_x[mask] = 0
                
        #         # transformer model
        #         pred = torch.sigmoid(model3(output_x))
        #         # add score
        #         all_val_score.append(pred.cpu().data.numpy()) 
                
        #         # add pred 
        #         predicted_labels = (pred >= 0.5).float()
        #         all_pred_val.append(predicted_labels.cpu().data.numpy())
                
                
        # pred_val = np.concatenate(all_pred_val)
        # true_val = np.concatenate(all_y_val)
        # score_val = np.concatenate(all_val_score)
        
        # # find optimal thresholds
        # thresholds = find_optimal_thresholds(true_val,score_val) # score and true 
        # print("=======================")
        # f1_macro_val = f1_score(true_val,pred_val , average='macro')
        # f1_micio_val = f1_score(true_val,pred_val , average='micro')
        # # f1_numpy = f1_score(pred_test, true_test, average='None')
        # print("F1_macro:{}".format(f1_macro_val))
        # print("F1_micro:{}".format(f1_micio_val))
        # auc_scores_val = []
        # # print(true_val.shape)
        # # print(pred_val.shape)
        # # print(true_val)
        
        
        # for i in range(23):
        #     auc_a = roc_auc_score(true_val[:, i], pred_val[:, i])
        #     auc_scores_val.append(auc_a)
        
                
        # print("NORM :{},\nLAFB/LPFB :{},\nIRBBB :{},\nILBBB :{},\nCLBBB :{},"\
        #     "\nCRBBB :{},\n_AVB :{},\nIVCD :{},\nWPW :{},\nLVH :{},\nRHV :{},"\
        #     "\nLAO/LAE:{},\nRAO/RAE :{},\nSEHYP :{},\nAMI :{},\nIMI :{},\nLMI :{},\n'PMI' :{},"\
        #     "\nISCA:{},\nISCI:{},\nISC_:{},\nSTTC:{},\nNST_:{}"
        #     .format(auc_scores_val[0],
        #              auc_scores_val[1],
        #              auc_scores_val[2],
        #              auc_scores_val[3],
        #              auc_scores_val[4],
        #              auc_scores_val[5],
        #              auc_scores_val[6],
        #              auc_scores_val[7],
        #              auc_scores_val[8],
        #              auc_scores_val[9],
        #              auc_scores_val[10],
        #              auc_scores_val[11],
        #              auc_scores_val[12],
        #              auc_scores_val[13],
        #              auc_scores_val[14],
        #              auc_scores_val[15],
        #              auc_scores_val[16],
        #              auc_scores_val[17],
        #              auc_scores_val[18],
        #              auc_scores_val[19],
        #              auc_scores_val[20],
        #              auc_scores_val[21],
        #              auc_scores_val[22]))
        # print("=======================")
        
        # # test
        # model1.eval()
        # model3.eval()
        # prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        # all_test_y = [] # true
        # all_test_pred = [] # pred score
        # with torch.no_grad():
        #     for batch_idx, batch in enumerate(prog_iter_test):
        #         input_x, input_y = tuple(t.to(device) for t in batch)
        #         all_test_y.append(input_y.cpu().data.numpy()) # 

        #         mask = torch.isnan(input_x).all(dim=-1) # B,C,N [256,12,15] 
        #         input_x[torch.isnan(input_x)] = 0 # 
        #         temp_x = input_x.view(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # 256 x 12 x15 
                
        #         # add 2 channel mask
        #         channel_2_mask = torch.zeros_like(input_x)
        #         channel_2_mask[torch.isnan(input_x)] = 0.0
        #         channel_2_mask[~torch.isnan(input_x)] = 1.0
        #         channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
            
                
        #         PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [256 x 12 x15,1,768]
        #         # [256,12,15,768]
        #         output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)
        #         output_x[mask] = 0
    
        #         pred = torch.sigmoid(model3(output_x)) # add sigmoid 
        #         all_test_pred.append(pred.cpu().data.numpy()) # add score
                
        # test_true = np.concatenate(all_test_y)
        # test_score = np.concatenate(all_test_pred)
        # test_labels = np.zeros_like(test_score)
        # # thresholds = 0.5
        # for i in range(23):
        #     test_labels[:, i] = (test_score[:, i] >= thresholds[i])

        # print("***********************************")
        # f1_macro_test = f1_score(test_true,test_labels , average='macro')
        # f1_micio_test = f1_score(test_true,test_labels , average='micro')
        # print("F1_macro:{}".format(f1_macro_test))
        # print("F1_micro:{}".format(f1_micio_test))
        
        # auc_scores_test = []
        # for i in range(23):
        #     # AUC
        #     auc_a = roc_auc_score(test_true[:, i], test_labels[:, i])
        #     auc_scores_test.append(auc_a)
                

        # print("***********************************")
        # print("NORM :{},\nLAFB/LPFB :{},\nIRBBB :{},\nILBBB :{},\nCLBBB :{},"\
        #     "\nCRBBB :{},\n_AVB :{},\nIVCD :{},\nWPW :{},\nLVH :{},\nRHV :{},"\
        #     "\nLAO/LAE:{},\nRAO/RAE :{},\nSEHYP :{},\nAMI :{},\nIMI :{},\nLMI :{},\nPMI :{},"\
        #     "\nISCA:{},\nISCI:{},\nISC_:{},\nSTTC:{},\nNST_:{}"
        #     .format(auc_scores_test[0],
        #              auc_scores_test[1],
        #              auc_scores_test[2],
        #              auc_scores_test[3],
        #              auc_scores_test[4],
        #              auc_scores_test[5],
        #              auc_scores_test[6],
        #              auc_scores_test[7],
        #              auc_scores_test[8],
        #              auc_scores_test[9],
        #              auc_scores_test[10],
        #              auc_scores_test[11],
        #              auc_scores_test[12],
        #              auc_scores_test[13],
        #              auc_scores_test[14],
        #              auc_scores_test[15],
        #              auc_scores_test[16],
        #              auc_scores_test[17],
        #              auc_scores_test[18],
        #              auc_scores_test[19],
        #              auc_scores_test[20],
        #              auc_scores_test[21],
        #              auc_scores_test[22]))
        # print("***********************************")




        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict() 
        # list_auc = list()
        # th_comfusion_matrix = [] # 
        # for i in range(test_true.shape[1]):
        #     fpr[i], tpr[i], th = roc_curve(test_true[:, i], test_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        #     list_auc.append(roc_auc[i])

        

        # all_auc = 0
        # plt.figure()
        # for i in range(test_true.shape[1]):
        #     if i == 6:
        #         plt.plot(fpr[i], tpr[i], label=f'AVB (AUC = {roc_auc[i]:.2f})')
        #         continue
        #     plt.plot(fpr[i], tpr[i], label=f'{labels_dic[i]} (AUC = {roc_auc[i]:.2f})')
        #     all_auc = all_auc+roc_auc[i]
        # avg_auc = all_auc/5    
            
        # # 
        # plt.plot([0, 1], [0, 1], 'k--')
        # # 
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic for Multi-Label Multi-Class')
        # plt.legend(loc="lower right",fontsize='xx-small')
        # plt.subplots_adjust(right=0.8) # 
        # plt.savefig("./last/add_se/sub_5fold/roc_"+str(count)+"epoch_v5.png")
        # count = count + 1

        

        # save_check(avg_auc,model1,model3,count,thresholds)
        
        # try:
        #     existing_df = pd.read_excel('./output_s3.xlsx')
        # except FileNotFoundError:

        #     df1 = pd.DataFrame([list_auc])
        #     df1.to_excel('./output_s3.xlsx', index=False, header=False)
        #     continue
        # df2 = pd.DataFrame([list_auc])
        # existing_df = pd.read_excel('./output_s3.xlsx', header=None)
        # combined_df = pd.concat([existing_df, df2], ignore_index=True)
        # combined_df.to_excel('./output_s3.xlsx', index=False, header=False) 
        
        
 
    
    #  SummaryWriter
    # writer.close()  
    # torch.save(model.state_dict(), './model/13channel_stand_v1.pth')
    



