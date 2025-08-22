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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from new_utils import *
from data_provider.uea import collate_fn

warnings.filterwarnings("ignore")

class MyDataset(Dataset):

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
        # 计算交叉熵损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算 p_t
        pt = torch.exp(-BCE_loss)
        
        # 计算 Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss    
np.random.seed(40) # 45

def encode_labels(data, label_to_index):
# 初始化零矩阵，形状为[数据长度, 标签数量]
    target = torch.zeros((len(data), len(label_to_index)), dtype=torch.float32)
    
    # 填充矩阵
    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target

class Exp_my(Exp_Basic):
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
        criterion = nn.BCEWithLogitsLoss()
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
        # accuracy = cal_accuracy(predictions, trues)
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

    def train(self, setting):
        
        # load data
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
        
        # 加载测试集和验证集
        dic_val = torch.load('/data/zhangshanwei/code_path_ecg/new_experiment/ex4_data/sub_ex4_val_5fold.pt')
        dic_test = torch.load('/data/zhangshanwei/code_path_ecg/new_experiment/ex4_data/sub_ex4_test_5fold.pt')
        val_set = dic_val["data"]
        vaild_label = dic_val["label"]
        val_set = np.array(val_set)
        test_set = dic_test["data"]
        test_label = dic_test["label"]
        test_set = np.array(test_set)
        
        # 打乱这个
        
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
        
        dataset_train = MyDataset(train_delete_data,train_label)
        dataset_val = MyDataset(val_set,vaild_label)
        dataset_test = MyDataset(test_set,test_label)

        dataloader_train = DataLoader(dataset_train, batch_size=64, drop_last=True,collate_fn=lambda x: collate_fn(
                x, max_len=1000
            ))
        dataloader_val = DataLoader(dataset_val, batch_size=64, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=64, drop_last=True)
        
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

        train_steps = len(dataloader_train)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(30):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(dataloader_train):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long())
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
            print(train_loss)
            
        #     vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        #     test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        #     print(
        #         f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
        #         f"Validation results --- Loss: {vali_loss:.5f}, "
        #         f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
        #         f"Precision: {val_metrics_dict['Precision']:.5f}, "
        #         f"Recall: {val_metrics_dict['Recall']:.5f}, "
        #         f"F1: {val_metrics_dict['F1']:.5f}, "
        #         f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
        #         f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
        #         f"Test results --- Loss: {test_loss:.5f}, "
        #         f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
        #         f"Precision: {test_metrics_dict['Precision']:.5f}, "
        #         f"Recall: {test_metrics_dict['Recall']:.5f} "
        #         f"F1: {test_metrics_dict['F1']:.5f}, "
        #         f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
        #         f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        #     )
        #     early_stopping(
        #         -val_metrics_dict["F1"],
        #         self.swa_model if self.swa else self.model,
        #         path,
        #     )
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
        #     """if (epoch + 1) % 5 == 0:
        #         adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        # best_model_path = path + "checkpoint.pth"
        # if self.swa:
        #     self.swa_model.load_state_dict(torch.load(best_model_path))
        # else:
        #     self.model.load_state_dict(torch.load(best_model_path))

        return self.model



    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
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
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return
