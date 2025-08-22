from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    roc_auc_score, f1_score,auc,roc_curve,multilabel_confusion_matrix
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import pandas as pd

def random_mask_channels(batch_x, k):
    n, length, channels = batch_x.shape
    for i in range(n):
        mask_indices = torch.randperm(channels)[:k]  # 
        batch_x[i, :, mask_indices] = 0  # 
    return batch_x

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
   

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class Exp_Classification_my(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        

        # test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = 1000  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = 12  # redefine enc_in
        self.args.num_class = 23
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        
        data_set, data_loader = data_provider(self.args, flag)
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = MultiLabelFocalLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        
        
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        
        
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        
        accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def my_vali(self, vali_data, vali_loader, criterion,mask_channels):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                if mask_channels!=0:
                    batch_x = random_mask_channels(batch_x, mask_channels) # 
                
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                    
                    a = torch.sigmoid(outputs)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
                    a = torch.sigmoid(outputs)

                pred = a.detach().cpu()
                loss = criterion(pred, label.float().cpu())
                total_loss.append(loss)

                preds.append(a.cpu().data.numpy())
                trues.append(label.cpu().data.numpy())

        total_loss = np.average(total_loss)

        
        my_pred = np.concatenate(preds)
        my_true = np.concatenate(trues)
        


        
        fpr = dict()
        tpr = dict()
        roc_auc = dict() 
        list_auc = list()
        
        for i in range(23):
            fpr[i], tpr[i], th = roc_curve(my_true[:, i], my_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            list_auc.append(roc_auc[i])
        
        print(list_auc)
        
        return list_auc # 
        
    def test_v2(self,vali_data, vali_loader, criterion,layout_type):
        """
            layout_type : 
            0 : 3 x 4
            1 : 3 x 4 + Ⅱ
            2 : 3 x 4 + Ⅱ + V1
            3 : 6 x 2 
            4 : 6 x 2+Ⅱ
            5 : 12 
            6 : random layouts -l --layout 
        """
        
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader): # layout_type
                
                print(batch_x.shape) # n,1000,12
                batch_x_numpy = batch_x.cpu().numpy() # n,1000,12
                temp_batch_x = list()
                for i in range(batch_x_numpy.shape[0]):
                    temp_xx = batch_x_numpy[i]
                    temp_xx = gen_true_ecg_layout(temp_xx,length = 1000,layout = layout_type)
                    temp_batch_x.append(temp_xx)
                
                # batch_x = gen_true_ecg_layout(batch_x,length = 1000,layout = layout_type)
                b_x = torch.tensor(temp_batch_x)
                b_x = torch.nan_to_num(b_x, nan=0.0) # nan to zero
                
                b_x = b_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(b_x, padding_mask, None, None)
                    
                    a = torch.sigmoid(outputs)
                else:
                    outputs = self.model(b_x, padding_mask, None, None)
                    a = torch.sigmoid(outputs)

                pred = a.detach().cpu()
                loss = criterion(pred, label.float().cpu())
                total_loss.append(loss)

                preds.append(a.cpu().data.numpy())
                trues.append(label.cpu().data.numpy())

        total_loss = np.average(total_loss)

        my_pred = np.concatenate(preds)
        my_true = np.concatenate(trues)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict() 
        list_auc = list()
        
        for i in range(23):
            fpr[i], tpr[i], th = roc_curve(my_true[:, i], my_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            list_auc.append(roc_auc[i])
        
        print(list_auc)
        
        return list_auc     
        
    def test_v2_random_layout(self,vali_data, vali_loader, criterion):
        """
            
            
        """
        
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader): # layout_type
                
                b_x = batch_x
                b_x = torch.nan_to_num(b_x, nan=0.0) # nan转为0
                
                b_x = b_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(b_x, padding_mask, None, None)
                    
                    a = torch.sigmoid(outputs)
                else:
                    outputs = self.model(b_x, padding_mask, None, None)
                    a = torch.sigmoid(outputs)

                pred = a.detach().cpu()
                loss = criterion(pred, label.float().cpu())
                total_loss.append(loss)

                preds.append(a.cpu().data.numpy())
                trues.append(label.cpu().data.numpy())

        total_loss = np.average(total_loss)

        my_pred = np.concatenate(preds)
        my_true = np.concatenate(trues)
        
        # save pred
        need_save = {
        'score' : my_pred
        }
        torch.save(need_save,'./table_chaoyang/score.pt')
        

        fpr = dict()
        tpr = dict()
        roc_auc = dict() 
        list_auc = list()
        
        for i in range(23):
            fpr[i], tpr[i], th = roc_curve(my_true[:, i], my_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            list_auc.append(roc_auc[i])
        
        print(list_auc)
        
        return list_auc # 

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(train_data.data.shape)
        print(train_data.label.shape)
        print(vali_data.data.shape)
        print(vali_data.label.shape)
        print(test_data.data.shape)
        print(test_data.label.shape)

        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device) # 64,1000
                label = label.to(self.device)
                # print(label)
                # print(padding_mask.shape) # 64，10000
                # print(self.model) # 

                outputs = self.model(batch_x, padding_mask, None, None)
                # print(outputs)
                
                loss = criterion(outputs, label.float())
                train_loss.append(loss.item())
                
                
                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            

            # self.my_vali(vali_data, vali_loader, criterion,2)
            # self.my_vali(test_data, test_loader, criterion,2)
            
            my_result = [] # 
            
            table_path = './table_chaoyang/chaoyang.xlsx'
            print("Test is reading ....")
            # ==================================================================

            # for type_layout in range(6):
            #     temp = self.test_v2(test_data, test_loader, criterion, layout_type=type_layout)
                
            #     # 
            #     try:
            #         existing_df = pd.read_excel(table_path)
            #     except FileNotFoundError:

            #         df1 = pd.DataFrame([temp])
            #         df1.to_excel(table_path, index=False, header=False)
            #         continue
            #     df2 = pd.DataFrame([temp])
            #     existing_df = pd.read_excel(table_path, header=None)
            #     combined_df = pd.concat([existing_df, df2], ignore_index=True)
            #     combined_df.to_excel(table_path, index=False, header=False) 
            #     my_result.append(temp)
            
            # ==================================================================================
            
            result = self.test_v2_random_layout(test_data, test_loader, criterion)
            try:
                existing_df = pd.read_excel(table_path)
            except FileNotFoundError:

                df1 = pd.DataFrame([result])
                df1.to_excel(table_path, index=False, header=False)
                continue
            df2 = pd.DataFrame([result])
            existing_df = pd.read_excel(table_path, header=None)
            combined_df = pd.concat([existing_df, df2], ignore_index=True)
            combined_df.to_excel(table_path, index=False, header=False) 
            
            
            # vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            # test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n")
            
            


        return self.model

