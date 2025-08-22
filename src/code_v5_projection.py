import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    roc_auc_score, f1_score,auc,roc_curve,multilabel_confusion_matrix
from helper_code import *
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
from tensorboardX import SummaryWriter
from torchsummary import summary
import math
from my_transformer import *
import seaborn as sns
import ast,h5py
np.set_printoptions(threshold=np.inf)
np.random.seed(40) # 45
substring_labels = '# Labels:'

# from tensorboardX import SummaryWriter
# writer = SummaryWriter(logdir='/data/zhangshanwei/code_path_ecg/logs')
# Train classification model.
def train_dx_model(data_folder, model_folder, verbose):
    # read data
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
    device_str = "cuda:4"

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
    
    # # 区分年龄
    # sex_test = Y[Y.strat_fold == test_fold].sex
    
    test_delete_data = list()
    test_delete_label = list()
    # test_delete_sex = list()
    
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
    
    # 随机生成2个部分的数据集
    val_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(val_delete_data))
    test_list = np.random.choice([0, 1, 2, 3, 4, 5], size=len(test_delete_data))
    
    temp_val_data = list()
    for i in range(len(val_delete_data)):
        temp_layout_data = gen_true_ecg_layout(val_delete_data[i],length = 1000,layout = val_list[i])
        temp_val_data.append(temp_layout_data)
    
    temp_test_data = list()
    for i in range(len(test_delete_data)):
        temp_layout_data = gen_true_ecg_layout(test_delete_data[i],length=1000,layout = test_list[i])
        temp_test_data.append(temp_layout_data)
    
    
    xtest = np.array(temp_test_data)
    xtest = np.moveaxis(xtest, 1, 2)
    
    xval = np.array(temp_val_data)
    xval = np.moveaxis(xval, 1, 2)
    print(train_label.shape)
    print(xval.shape) # 2146
    print(xtest.shape) # 2158
    

    # 标准化
    for i in range(len(xtest)):
        tmp_data = xtest[i]
        tmp_std = np.nanstd(tmp_data)
        tmp_mean = np.nanmean(tmp_data)
        xtest[i] = (tmp_data - tmp_mean) / tmp_std

    for i in range(len(xval)):
        tmp_data = xval[i]
        tmp_std = np.nanstd(tmp_data)
        tmp_mean = np.nanmean(tmp_data)
        xval[i] = (tmp_data - tmp_mean) / tmp_std 

    # ======================================
    # 2、Test ours gen ours digitization 
    # with h5py.File('/data/0shared/zhangshanwei/cinc/ours/data/from_ours_pic_and_dig.h5', 'r') as f:
    #     raw_signal = f['signals'][:]
    #     bbby = f['labels'][:]
        
    
    # temp_our_test = np.array(raw_signal)
    # print(raw_signal.shape) # b,12,1000
    # # test_label = torch.from_numpy(bbby)
    
    # # filter data ,delete all 0
    # temp_list_ours_signal = list()
    # temp_list_ours_label = list()
    # for i in range(temp_our_test.shape[0]):
    #     tmp_data = temp_our_test[i]
    #     if not np.all(tmp_data==0):
    #         temp_list_ours_signal.append(temp_our_test[i])
    #         temp_list_ours_label.append(bbby[i])
    
    # xtest = np.array(temp_list_ours_signal)
    # label1 = np.array(temp_list_ours_label)
    # test_label = torch.from_numpy(label1)
    
    # print(xtest.shape)
    # print(label1.shape)
        
    # for i in range(len(xtest)):
    #     tmp_data = xtest[i]
    #     tmp_std = np.nanstd(tmp_data)
    #     tmp_mean = np.nanmean(tmp_data)
    #     xtest[i] = (tmp_data - tmp_mean) / tmp_std
    # 2、Test ours gen ours digitization 
    # ======================================
    
    
    # # # mask test 不同的几种方案
    # # # 1、直接使用原本的随机mask的测试
    # xtest = np.array(test_delete_data)
    # xtest = np.moveaxis(xtest, 1, 2)
    
    # xval = np.array(val_delete_data)
    # xval = np.moveaxis(xval, 1, 2)
    
    
    # ============================
    # # 生成测试和验证数据
    
    # dic_val = {}
    # dic_val["data"] = val_set
    # dic_val["label"] = vaild_label
    # torch.save(dic_val, './experment/data/val_eva_sub.pt')
    
    # dic_test = {}
    # dic_test["data"] = test_set
    # dic_test["label"] = test_label
    # torch.save(dic_test, './experment/data/test_eva_sub.pt')
    
    # 加载测试集和验证集
    # dic_val = torch.load('./new_experiment/ex1_data/sub_ex1_val_1fold.pt')
    # dic_test = torch.load('./new_experiment/ex1_data/sub_ex1_test_1fold.pt')
    # val_set = dic_val["data"]
    # vaild_label = dic_val["label"]
    # val_set = np.array(val_set)
    # test_set = dic_test["data"]
    # test_label = dic_test["label"]
    # test_set = np.array(test_set)
    
    
        
    # parameters
    n_epoch = 30
    batch_size = 64
    input_dim = 768
    header = 16
    num_layers = 4
    s = 16 # stride
    patch_len = 64
    
    dataset_val = MyDataset(xval,vaild_label)
    dataset_test = MyDataset(xtest,test_label)

    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=True)
    
    if verbose:
        print("mask is Done!")
    # Fit the model.
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    

    # make model
    model1 = Projection1(input_dim=patch_len,output_dim=input_dim,bias=0.1)
    model1.to(device)

    
    # model2 = TransformerEncoderForMultiLabelClassification(input_dim, header, num_layers, dim_feedforward=1024, num_labels=11, dropout=0.1)
    model3 = VisionTransformer_2dembedding_v2(
                              embed_dim=768,
                              depth=3,
                              num_heads=8,
                              representation_size=None,
                              num_classes=23,
                              num_patch = 15,
                              channel = 12,
                              drop_ratio=0.,
                              attn_drop_ratio=0., 
                              drop_path_ratio=0.
                              )
    model3.to(device)
    model_params = list(model1.parameters()) + list(model3.parameters())
    # Train the model.
    if verbose:
        print('Training the model on the data...')
    # optimizer = optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-5)
    # new 
    optimizer = optim.Adam(model_params, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    # loss_func = Hill()
    # torch.nn.BCEWithLogitsLoss()
    # AsymmetricLossOptimized
    # MultiLabelFocalLoss
    # balanced 的一个方法
    
    loss_func = MultiLabelFocalLoss()
    total_train_loss = [] # loss
    temp = []
    step = 0
    count = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
        # train
        model1.train()
        model3.train()
        
        # 每一次都重新mask train数据
        xtrain = np.array(train_delete_data) # 使用删除后的数据
        xtrain = np.moveaxis(xtrain, 1, 2)
        xtrain_1 = list()
        for i in range(xtrain.shape[0]):
        #     # temp_data = mask_12channal_train(xtrain[i])
            temp_data = xtrain[i]
            # noisy_data = add_noisy(temp_data,noise_std = 0.1) # 添加噪声
            mask_data = add_mask(temp_data) # 添加缺失
            xtrain_1.append(mask_data)
        train_set = np.array(xtrain_1) # 采用了mask 策略之后的训练集
        
        
        # ======================================
        # mask lead 
        # train_set = mask_lead(train_set,3)
        # shuffle
        # for i in range(train_set.shape[0]):
        #     permutation = np.random.permutation(train_set.shape[1]) # 生成和通道相同的随机数
        #     train_set[i] = train_set[i, permutation, :] # 随机按照通道打乱
        # ======================================
        
        
        # ======================================
        # 标准化
        for i in range(len(train_set)):
            tmp_data = train_set[i]
            tmp_std = np.nanstd(tmp_data)
            tmp_mean = np.nanmean(tmp_data)
            train_set[i] = (tmp_data - tmp_mean) / tmp_std
        
        # 标准化
        # ======================================
        

        dataset = MyDataset(train_set, train_label)
        dataloader = DataLoader(dataset, batch_size=batch_size,drop_last=True)

        prog_iter = tqdm(dataloader, desc="Training", leave=False , ncols=80)
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch) # [256, 12, 15, 64]            
            mask = torch.isnan(input_x).any(dim=-1) # b,12,15
            
            # 根据这个nan生成一个相同维度的mask
            # channel_2_mask = torch.zeros_like(input_x)
            # channel_2_mask[torch.isnan(input_x)] = 0.0
            # channel_2_mask[~torch.isnan(input_x)] = 1.0
            # channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
            

            # 2、将所有数据输入的时候全部置为nan变为0
            # input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
            # temp_x = input_x.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # B x 12 x15 
            # # print(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1).shape)
            
            
            # PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [B x 12 x15,2,768]
            # output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)

            input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
            output_x = model1(input_x)# b,12,15,768
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


        # val
        model1.eval()
        model3.eval()
        prog_iter_val = tqdm(dataloader_val, desc="Val", leave=False)

        all_pred_val = [] # 验证集 pred
        all_y_val = [] # 验证集 true
        all_val_score = [] # 验证集 score
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_val):
                input_x, input_y = tuple(t.to(device) for t in batch)
                all_y_val.append(input_y.cpu().data.numpy()) # 真实标签
                
                mask = torch.isnan(input_x).all(dim=-1) # B,C,N [256,12,15] 
                # input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
                # temp_x = input_x.view(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # 256 x 12 x15 
                # # 根据这个nan生成一个相同维度的mask
                # channel_2_mask = torch.zeros_like(input_x)
                # channel_2_mask[torch.isnan(input_x)] = 0.0
                # channel_2_mask[~torch.isnan(input_x)] = 1.0
                # channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
            
                
                # PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [256 x 12 x15,1,768]
                # # [256,12,15,768]
                # output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)
                input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
                output_x = model1(input_x)# b,12,15,768
                output_x[mask] = 0
                
                # transformer model
                pred = torch.sigmoid(model3(output_x))
                # add score
                all_val_score.append(pred.cpu().data.numpy()) 
                
                # add pred 
                predicted_labels = (pred >= 0.5).float()
                all_pred_val.append(predicted_labels.cpu().data.numpy())
                
                
        pred_val = np.concatenate(all_pred_val)
        true_val = np.concatenate(all_y_val)
        score_val = np.concatenate(all_val_score)
        
        # find optimal thresholds
        thresholds = find_optimal_thresholds(true_val,score_val) # score and true 
        print("=======================")
        f1_macro_val = f1_score(true_val,pred_val , average='macro')
        f1_micio_val = f1_score(true_val,pred_val , average='micro')
        # f1_numpy = f1_score(pred_test, true_test, average='None')
        print("F1_macro:{}".format(f1_macro_val))
        print("F1_micro:{}".format(f1_micio_val))
        auc_scores_val = []
        # print(true_val.shape)
        # print(pred_val.shape)
        # print(true_val)
        
        
        for i in range(23):
            auc_a = roc_auc_score(true_val[:, i], pred_val[:, i])
            auc_scores_val.append(auc_a)
        
                
        print("NORM :{},\nLAFB/LPFB :{},\nIRBBB :{},\nILBBB :{},\nCLBBB :{},"\
            "\nCRBBB :{},\n_AVB :{},\nIVCD :{},\nWPW :{},\nLVH :{},\nRHV :{},"\
            "\nLAO/LAE:{},\nRAO/RAE :{},\nSEHYP :{},\nAMI :{},\nIMI :{},\nLMI :{},\n'PMI' :{},"\
            "\nISCA:{},\nISCI:{},\nISC_:{},\nSTTC:{},\nNST_:{}"
            .format(auc_scores_val[0],
                     auc_scores_val[1],
                     auc_scores_val[2],
                     auc_scores_val[3],
                     auc_scores_val[4],
                     auc_scores_val[5],
                     auc_scores_val[6],
                     auc_scores_val[7],
                     auc_scores_val[8],
                     auc_scores_val[9],
                     auc_scores_val[10],
                     auc_scores_val[11],
                     auc_scores_val[12],
                     auc_scores_val[13],
                     auc_scores_val[14],
                     auc_scores_val[15],
                     auc_scores_val[16],
                     auc_scores_val[17],
                     auc_scores_val[18],
                     auc_scores_val[19],
                     auc_scores_val[20],
                     auc_scores_val[21],
                     auc_scores_val[22]))
        print("=======================")
        
        # test
        model1.eval()
        model3.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        all_test_y = [] # true
        all_test_pred = [] # pred score
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                all_test_y.append(input_y.cpu().data.numpy()) # 真实标签

                mask = torch.isnan(input_x).all(dim=-1) # B,C,N [256,12,15] 
                # input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
                # temp_x = input_x.view(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3]) # 256 x 12 x15 
                
                # # add 2 channel mask
                # channel_2_mask = torch.zeros_like(input_x)
                # channel_2_mask[torch.isnan(input_x)] = 0.0
                # channel_2_mask[~torch.isnan(input_x)] = 1.0
                # channel_2_temp = channel_2_mask.reshape(input_x.shape[0]*input_x.shape[1]*input_x.shape[2],input_x.shape[3])
            
                
                # PatchEncode_output = model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1)) # [256 x 12 x15,1,768]
                # # [256,12,15,768]
                # output_x = PatchEncode_output.squeeze(1).view(input_x.shape[0],input_x.shape[1],input_x.shape[2],768)
                input_x[torch.isnan(input_x)] = 0 # 将所有nan 填充为0
                output_x = model1(input_x)# b,12,15,768
                output_x[mask] = 0
    
                pred = torch.sigmoid(model3(output_x)) # add sigmoid 最后一层需要
                all_test_pred.append(pred.cpu().data.numpy()) # add score
                
        test_true = np.concatenate(all_test_y)
        test_score = np.concatenate(all_test_pred)
        test_labels = np.zeros_like(test_score)
        # thresholds = 0.5
        for i in range(23):
            test_labels[:, i] = (test_score[:, i] >= thresholds[i])

        print("***********************************")
        f1_macro_test = f1_score(test_true,test_labels , average='macro')
        f1_micio_test = f1_score(test_true,test_labels , average='micro')
        print("F1_macro:{}".format(f1_macro_test))
        print("F1_micro:{}".format(f1_micio_test))
        
        auc_scores_test = []
        for i in range(23):
            # 计算当前标签的AUC
            auc_a = roc_auc_score(test_true[:, i], test_labels[:, i])
            auc_scores_test.append(auc_a)
                

        print("***********************************")
        print("NORM :{},\nLAFB/LPFB :{},\nIRBBB :{},\nILBBB :{},\nCLBBB :{},"\
            "\nCRBBB :{},\n_AVB :{},\nIVCD :{},\nWPW :{},\nLVH :{},\nRHV :{},"\
            "\nLAO/LAE:{},\nRAO/RAE :{},\nSEHYP :{},\nAMI :{},\nIMI :{},\nLMI :{},\nPMI :{},"\
            "\nISCA:{},\nISCI:{},\nISC_:{},\nSTTC:{},\nNST_:{}"
            .format(auc_scores_test[0],
                     auc_scores_test[1],
                     auc_scores_test[2],
                     auc_scores_test[3],
                     auc_scores_test[4],
                     auc_scores_test[5],
                     auc_scores_test[6],
                     auc_scores_test[7],
                     auc_scores_test[8],
                     auc_scores_test[9],
                     auc_scores_test[10],
                     auc_scores_test[11],
                     auc_scores_test[12],
                     auc_scores_test[13],
                     auc_scores_test[14],
                     auc_scores_test[15],
                     auc_scores_test[16],
                     auc_scores_test[17],
                     auc_scores_test[18],
                     auc_scores_test[19],
                     auc_scores_test[20],
                     auc_scores_test[21],
                     auc_scores_test[22]))
        print("***********************************")
        # plt csv result 
        # result_list = [[_, f1_macro_test, f1_micio_test, auc_scores_test[0],auc_scores_test[1],auc_scores_test[2],
        #                 auc_scores_test[3],auc_scores_test[4]]]
        # if _ ==0:
        #     columns = ['epoch','F1_macro','F1_micro',
        #            'AUC_NORM','MI','AUC_STTC','AUC_CD',
        #            'AUC_HYP']
        # else:
        #     columns = ['', '', '', '', '', '', '', '']
    
        # dt = pd.DataFrame(result_list, columns=columns)
        # dt.to_csv('./result/v4.csv', mode='a')
    
        # plt roc curve
        # 计算ROC曲线和AUC（每个标签）
        fpr = dict()
        tpr = dict()
        roc_auc = dict() 
        list_auc = list()
        th_comfusion_matrix = [] # 最优的阈值通过roc计算出来的
        for i in range(test_true.shape[1]):
            fpr[i], tpr[i], th = roc_curve(test_true[:, i], test_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            list_auc.append(roc_auc[i])
        # ==================================================================
            # 图片上的应该是最优的阈值，不是我们选取阈值的方法，优化的那个方法
        #     optimal_idx = np.argmax(tpr[i] - fpr[i])
        #     optimal_threshold = th[optimal_idx]
        #     th_comfusion_matrix.append(optimal_threshold)
    
        # # th_matrix = np.array(th_comfusion_matrix) # 最优的阈值
        # confusion_labels = np.zeros_like(test_score) 
        # for i in range(11):
        #     confusion_labels[:, i] = (test_score[:, i] >= thresholds[i])
        
        # print(test_true.shape)
        # print(confusion_labels.shape)
        
        # conf_matrices = multilabel_confusion_matrix(test_true, confusion_labels)
        # fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        # for i, ax in enumerate(axes.flatten()):
        #     if i < 11:  # 只绘制前11个标签的混淆矩阵
        #         sns.heatmap(conf_matrices[i], annot=True, fmt='d', cmap='Blues', ax=ax)
        #         ax.set_title(f'{labels_dic[i]}')
        #         ax.set_xlabel('Predicted')
        #         ax.set_ylabel('True')
        #     else:
        #         ax.axis('off')  # 关闭多余的子图
        
        # plt.tight_layout()
        # plt.savefig("./picture/patch_confusion.png")
        # =========================================================================
        
        
        # 绘制ROC曲线
        all_auc = 0
        plt.figure()
        for i in range(test_true.shape[1]):
            if i == 6:
                plt.plot(fpr[i], tpr[i], label=f'AVB (AUC = {roc_auc[i]:.2f})')
                continue
            plt.plot(fpr[i], tpr[i], label=f'{labels_dic[i]} (AUC = {roc_auc[i]:.2f})')
            all_auc = all_auc+roc_auc[i]
        avg_auc = all_auc/5    
            
        # # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--')
        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Multi-Label Multi-Class')
        plt.legend(loc="lower right",fontsize='xx-small')
        plt.subplots_adjust(right=0.8) # 调整子图布局
        plt.savefig("/data/0shared/zhangshanwei/cinc/ours/src/PatchEncoder/picture/5fold/roc_"+str(count)+"epoch.png")
        count = count + 1

        
        # 保存每一轮的参数
        save_check(avg_auc,model1,model3,count,thresholds)
        
        # save table path
        save_table_path = '/data/0shared/zhangshanwei/cinc/ours/src/PatchEncoder/table/5fold.xlsx'

        try:
            existing_df = pd.read_excel(save_table_path)
        except FileNotFoundError:
            print("文件 'existing_file.xlsx' 不存在，将创建新文件。")
            df1 = pd.DataFrame([list_auc])
            df1.to_excel(save_table_path, index=False, header=False)
            continue
        df2 = pd.DataFrame([list_auc])
        existing_df = pd.read_excel(save_table_path, header=None)
        combined_df = pd.concat([existing_df, df2], ignore_index=True)
        combined_df.to_excel(save_table_path, index=False, header=False) 
        
        
    # plt loss picture
    # plt.figure(figsize=(10, 5))
    # plt.plot(total_train_loss, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Time')
    # plt.legend()
    # plt.savefig("./picture/lose/v5.png")  
    
    # 关闭 SummaryWriter
    # writer.close()  
    # torch.save(model.state_dict(), './model/13channel_stand_v1.pth')
    
    if verbose:
        print('Done.')
        print()

# Extract features.
def extract_features(record): # extract feature
    # 空数据全部置为0，即可
    lead_12,fields = load_signals(record)
    a = np.array(lead_12)
    b = np.transpose(a,[1,0]) # change dim
    df = pd.DataFrame(b) 
    arr = df.fillna(0).values
    signal = list(arr)
    return signal

class MyDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label
        # self.patch_length = patch_length
        # self.stride = stride
        
    def __getitem__(self, index):
        # slide sequence to patch
        raw_data = np.array(self.data[index]) # (12,1000)
        # (12,1000) -> (12,num_patch,patch_length)  切割数据集
        total_length = raw_data.shape[1] # 1000
        patch_length = 64
        stride = 16
        
        # # ==================================================================================================================================
        # # 判断是否需要填充数据
        # # remainder = (total_length- patch_length) % stride
        # # if remainder !=0:
        # #     padding_needed = stride-remainder # 需要填充的数量
        # #     print(padding_needed)
        # #     padded_data = np.pad(raw_data, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0)
        # #     print(padded_data.shape) # 
        
        
        # # 处理最后一个确实的部分
        # channel_num_patch = (total_length-patch_length)//stride+1 # 59 update->60
        
        # # print(channel_num_patch)
        # # 创建一个空数组来存储划分后的数据
        # sliced_data = np.empty((12, channel_num_patch, patch_length)) # shape (12,59,64)
        
        # for i in range(channel_num_patch): # 59
        #     start = i * stride
        #     end = start + patch_length
        #     # sliced_data[:, i, :] = raw_data[:, start:end] # 最后一个直接神略
        #     sliced_data[:, i, :] = raw_data[:, start:end] # 添加最后一个，用0进行填充 （12,60,64）
        
        
        # # print(sliced_data.shape) # 12,59,64 -> 12,60,64
        # # ==================================================================================================================================
        
        # 第二种patch的方案
        # sliced_data = silde_windows_without_overlap(raw_data,patch_length) # （12,16,64）
        
        # 不要最后一个不完整的，不重叠
        sliced_data = silde_windows_without_overlap_v1(raw_data,patch_length)# （12,15,64）
        
        # print(sliced_data.shape) 
        
        return (torch.tensor(sliced_data, dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float))

    def __len__(self):
        return len(self.data)
      

def my_mask(lead_tp = 1, len = 1000):
    """
    parameters:
        lead_tp : type of ECG , 1 表示12，1的排布，2表示3，4的排布，3表示6，2的排布
        len : 数据的长度
    retrun (12,length)
    """
    step = 1000
    if lead_tp == 2: 
        step = int(len/4)
    if lead_tp == 3:
        step = int(len/2)
        
    #print("每条导联的长度："+str(step))
    msk = np.zeros((12,len))
    
    if lead_tp == 2:
        msk[0,step:len] = 1
        msk[2,step:len] = 1
        
        msk[3,0:step] = 1
        msk[3,2*step:len] = 1
        msk[4,0:step] = 1
        msk[4,2*step:len] = 1
        msk[5,0:step] = 1
        msk[5,2*step:len] = 1
        
        msk[6,0:2*step] = 1
        msk[6,3*step:len] = 1
        msk[7,0:2*step] = 1
        msk[7,3*step:len] = 1
        msk[8,0:2*step] = 1
        msk[8,3*step:len] = 1
        
        msk[9,0:3*step] = 1
        msk[10,0:3*step] = 1
        msk[11,0:3*step] = 1
        
    if lead_tp == 3:
        msk[0,step:len] = 1
        msk[2,step:len] = 1
        msk[3,step:len] = 1
        msk[4,step:len] = 1
        msk[5,step:len] = 1

        msk[6,0:step] = 1
        msk[7,0:step] = 1
        msk[8,0:step] = 1
        msk[9,0:step] = 1
        msk[10,0:step] = 1
        msk[11,0:step] = 1
         
    return msk

def encode_labels(data, label_to_index):
# 初始化零矩阵，形状为[数据长度, 标签数量]
    target = torch.zeros((len(data), len(label_to_index)), dtype=torch.float32)
    
    # 填充矩阵
    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target


class Projection1(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 activation: str = None,
                 bias: bool = True,
                 layer_norm: bool = False,
                 dropout: float = 0.0):
        """
        投影模型 (Model1)
        
        参数:
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        activation: 激活函数类型 [None, 'relu', 'gelu', 'sigmoid', 'tanh']
        bias: 是否使用偏置项
        layer_norm: 是否使用Layer Normalization
        dropout: Dropout概率 (0.0表示不使用)
        """
        super(Projection1, self).__init__()
        
        # 核心投影层
        self.projection = nn.Linear(input_dim, output_dim, bias=bias)
        
        # 激活函数
        self.activation = None
        if activation:
            activation = activation.lower()
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'gelu':
                self.activation = nn.GELU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 应用投影
        x = self.projection(x)
        
        # 应用激活函数 (如果有)
        if self.activation:
            x = self.activation(x)
        
        # 应用Layer Normalization (如果有)
        if self.layer_norm:
            x = self.layer_norm(x)
        
        # 应用Dropout (如果有)
        if self.dropout:
            x = self.dropout(x)
            
        return x

    def __repr__(self):
        """模型描述"""
        params = f"input_dim={self.projection.in_features}, output_dim={self.projection.out_features}"
        if self.activation:
            params += f", activation={self.activation.__class__.__name__}"
        if self.layer_norm:
            params += f", layer_norm=True"
        if self.dropout:
            params += f", dropout={self.dropout.p}"
        return f"Model1({params})"


# transformer model
class TransformerEncoderForMultiLabelClassification(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_labels, dropout=0.1):
        super(TransformerEncoderForMultiLabelClassification, self).__init__()
        """
            d_model:输入vector的维度
            nhead 
            num_layers:
            dim_feedforward : 前馈神经网络的隐藏测层维度
            dropout : 
            输出：
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_labels = num_labels
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))
        
        # self.time_positional_encoding = positionalencoding1d(708,d_model).unsqueeze(0) # (1,length,d_model) 708,128
        self.time_positional_encoding = positionalencoding1d(180,d_model).unsqueeze(0)
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 全连接层用于多标签分类
        # self.fc1 = nn.Linear(d_model*708, 128) # 90624 的输入
        self.fc1 = nn.Linear(d_model*180, 128)
        self.fc2 = nn.Linear(128, num_labels)
        
    def forward(self, src):
        # input (batch,length,d_model) 16,708,128
        # src: (batch, length, d_model)
        batch_size, length, d_model = src.size()
        
        # 添加位置编码
        # pe = self.positional_encoding[:length,d_model] # shape (d_model,length) 708,128 ->1,708,128
        # pe.to("cuda:2")
        # time_position = pe.unsqueeze(0) # 708,128 ->1,708,128
        # src = src + self.positional_encoding[:, :length, :] 
        self.time_positional_encoding.to("cuda:3")
        # print(self.time_positional_encoding.shape) # (1,length,d_model) 1,708,128
        
        src = src + self.time_positional_encoding
        
        # 调整维度顺序以适应Transformer的输入要求 (length, batch, d_model)
        # src = src.permute(1, 0, 2)
        # print("print input transformer dim:..")
        # print(src.shape) # 16,708,128
        
        # 通过Transformer编码器
        output = self.transformer_encoder(src)
        # print(output.shape) # 16,708,128
        output = torch.flatten(output,1) # (16,708,128) - > (16,90624)
        
        # print(output.shape) # （16,90624）
                
        output = self.fc1(output)
        output = self.fc2(output)
        

        
        
        # # 调整回原来的维度顺序 (batch, length, d_model)
        # output = output.permute(1, 0, 2)
        # 取最后一个时间步的输出作为分类器的输入
        # cls_output = output[:, -1, :]  # (batch, d_model)
        # 通过全连接层进行多标签分类
        # logits = self.fc(cls_output)  # (batch, num_labels)
        return output


def positionalencoding1d(length, d_model):
    """
        input : param d_model: dimension of the model
              : param length: length of positions
        output : return: length*d_model position matrix
        
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model).to("cuda:3")
    position = torch.arange(0, length).unsqueeze(1).to("cuda:3")
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model))).to("cuda:3")
    pe[:, 0::2] = torch.sin(position.float() * div_term).to("cuda:3")
    pe[:, 1::2] = torch.cos(position.float() * div_term).to("cuda:3")

    
    return pe 
        
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def create_2d_relative_bias_trainble_embeddings(n_head,height,width,dim):
    position_embedding = nn.Embedding((2*width-1)*(2*height-1),n_head)
    nn.init.constant_(position_embedding,0.)
    def get_relative_position_index(height,width):
        coords = torch.stack(torch.meshgrid(torch.arange(height),torch.arange(width))) # [2,height,width]
        coords_flatten = torch.flatten(coords,1) # [2,height * width]
        relative_coords_bias = coords_flatten[:,:,None]-coords_flatten[:,None,:] # [2,height*width,height*width]
        
        relative_coords_bias[0,:,:] += height - 1
        relative_coords_bias[1,:,:] += width - 1
        
        relative_coords_bias[0,:,:] *= relative_coords_bias[1,:,:].max()+1
        return relative_coords_bias.sum(0) # [height * width,height * width]
    
    relative_position_bias = get_relative_position_index(height,width)
    bias_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(height*width,height*width,n_head) # [height*width,height*width,n_head]
    bias_embedding = bias_embedding.permute(2,0,1).unsqueeze(0) # [1,n_head,height*width,height*width]
    
    return bias_embedding

def create_2d_absolate_sincos_embeddings(height,width,dim):
    """
        2d 绝对路径的embedding
    """
    assert dim % 4==0,"wrong dimension"
    
    position_embedding = torch.zeros(height*width,dim)
    coords = torch.stack()
    return position_embedding

def silde_windows(data,stride,patch_length):
    """
        input :
            data : (channel,length)
            stride :
            patch_length : 
        output : (channel,num_patch,patch_length)
        
    """
    
    total_length = data.shape[1]
    channel = data.shap[0]
    
    # 判断是否需要填充数据
    
    
    return 1

def silde_windows_without_overlap(data,patch_length):
    """
        input :
            data : (channel,length)
            stride :
            patch_length : 
        output : (channel,num_patch,patch_length) ndarray
    """
    
    total_length = data.shape[1] # 1000
    channel = data.shape[0] # 12
    
    # 判断需要填充的数据
    remainder = total_length % patch_length # 40
    padding_needed = 0 if remainder == 0 else patch_length - remainder  # 24
    padded_data = np.pad(data, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0) # (12,1024)
    num_patch = int(padded_data.shape[1]/patch_length) # 16
    sliced_data = np.empty((channel, num_patch, patch_length)) # 12,16,64
    for i in range(num_patch): # 16
        start = i * patch_length
        end = start + patch_length
        # sliced_data[:, i, :] = raw_data[:, start:end] # 最后一个直接神略
        sliced_data[:, i, :] = padded_data[:, start:end] # 添加最后一个，用0进行填充 
    
    return sliced_data

def silde_windows_without_overlap_v1(data,patch_length):
    """
        input:
            data : (channel,length)
            stride :
            patch_length : 
        output : (channel,num_patch,patch_length) ndarray
        comment : 不要最后一个缺失的块
    """
    total_length = data.shape[1] # 1000
    channel = data.shape[0] # 12
    stride = patch_length
    # 不要最后一个
    num_patch = (total_length-patch_length)//stride+1 
    sliced_data = np.empty((channel, num_patch, patch_length))
    for i in range(num_patch): # 16
        start = i * patch_length
        end = start + patch_length
        # sliced_data[:, i, :] = raw_data[:, start:end] # 最后一个直接神略
        sliced_data[:, i, :] = data[:, start:end] # 添加最后一个，用0进行填充 
    
    return sliced_data