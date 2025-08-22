import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score
from utils.augmentations import data_transform_masked4cl
from model import TFC, target_classifier
import warnings
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_curve, auc,precision_recall_curve
warnings.filterwarnings("ignore")


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        targets = targets.float()
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

def build_model(args, lr, configs, device, chkpoint=None):
    # Model Backbone
    model = TFC(configs, args).to(device)
    if chkpoint:
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Classification Head
    classifier = target_classifier(configs).to(device)

    # Optimizer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(configs.beta1, configs.beta2),
                                            weight_decay=0)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=args.finetune_epoch)

    return model, classifier, model_optimizer, classifier_optimizer, model_scheduler

def Tester(model,
            model_optimizer,
            model_scheduler,
            train_dl,
            valid_dl,
            test_dl,
            device,
            logger,
            args,
            configs,
            experiment_log_dir,
            seed):
    
    # 加载模型
    chkpoint = torch.load(os.path.join(experiment_log_dir, f"saved_models/", f'ckp_ep9.pt'))
    th = chkpoint['th']
    print(th) # 打印了阈值
    
    
    ft_model, ft_classifier, ft_model_optimizer, ft_classifier_optimizer, ft_scheduler = build_model(args,
                                                                                                        args.lr,
                                                                                                        configs,
                                                                                                        device,
                                                                                                        chkpoint)
    
    #  run test

    total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance = model_test(ft_model,
                                                                                            test_dl,
                                                                                            device,
                                                                                            classifier=ft_classifier,
                                                                                            th = th)
    
    
    return 0

def model_test(model, test_dl, device, classifier=None,th=None):
    # th 来自验证集的阈值
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    total_precision, total_recall, total_f1 = [], [], []
    
    

    criterion = MultiLabelFocalLoss()
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []

    
    pred_score = []
    true_label = []
    
    
    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels in test_dl:

            data, labels = data.float().to(device), labels.long().to(device) # label n,23

            # Add supervised classifier: 1) it's unique to fine-tuning. 2) this classifier will also be used in test
            h, z = model(data)

            fea_concat = h
            predictions_test = classifier(fea_concat) # 32,23
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1) # 32,1280
            emb_test_all.append(fea_concat_flat)

            loss = criterion(predictions_test, labels)
            
            # ours
            a = torch.sigmoid(predictions_test) # a 添加一个sigmoid
            
            pred_score.append(a.cpu().data.numpy())
            true_label.append(labels.cpu().data.numpy())
            total_loss.append(loss.item())
            
            outs = np.append(outs, predictions_test.detach().cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())



    # ours
    test_pred_score = np.concatenate(pred_score)
    test_true = np.concatenate(true_label)
    
    # print(test_pred_score.shape) # 2112,23
    # print(test_true.shape) # 2112,23
    
    test_pred_labels = np.zeros_like(test_pred_score)
    for i in range(23):
        test_pred_labels[:, i] = (test_pred_score[:, i] >= th[i]) # th来自验证集的调整
        
    F1 = f1_score(test_true, test_pred_labels, average='macro')  # F1输入的应该是标签不是score，auc得到的是score
    print("F1_macro_test:{}".format(F1))
    

    # auc 
    fpr = dict()
    tpr = dict()
    roc_auc = dict() 
    list_auc = list()
    
    for i in range(test_true.shape[1]):
        fpr[i], tpr[i], th = roc_curve(test_true[:, i], test_pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        list_auc.append(roc_auc[i])
    
    print(list_auc)
    
    # 保存一下这个数据
    try:
        existing_df = pd.read_excel('./code/table/sub_ex1_1fold_test.xlsx')
    except FileNotFoundError:
        print("文件 'existing_file.xlsx' 不存在，将创建新文件。")
        df1 = pd.DataFrame([list_auc])
        df1.to_excel('./code/table/sub_ex1_1fold_test.xlsx', index=False, header=False)
    
    
    df2 = pd.DataFrame([list_auc])
    existing_df = pd.read_excel('./code/table/sub_ex1_1fold_test.xlsx', header=None)
    combined_df = pd.concat([existing_df, df2], ignore_index=True)
    combined_df.to_excel('./code/table/sub_ex1_1fold_test.xlsx', index=False, header=False) 
        
        

    precision = 0
    recall = 0
    acc = 0
    total_loss = torch.tensor(total_loss).mean()
    total_acc = 0
    total_auc = 0
    total_prc = 0

    performance = [acc * 100, precision * 100, recall * 100, F1, total_auc * 100, total_prc * 100]
    # 最终选择的是最好的acc的性能部分

    emb_test_all = torch.cat(tuple(emb_test_all))
    return total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance