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


def find_optimal_thresholds(gt, pred):
    """
    Find optimal threshold for each task based on Balanced Accuracy.

    Args:
        gt: Ground truth labels (numpy array)
        pred: Prediction probabilities (numpy array)

    Returns:
        optimal_thresholds: Optimal threshold for each task(list)
    """
    n_task = gt.shape[1]
    optimal_thresholds = []

    for i in range(n_task):
        best_ba = -1  
        best_thresh = 0.5  
        for thresh in np.linspace(0.01, 0.99, 99):  
            pred_labels = (pred[:, i] > thresh).astype(int)
            ba = balanced_accuracy_score(gt[:, i], pred_labels)  
            if ba > best_ba:
                best_ba = ba
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)
    # print("best thresholds:\n")
    # print(optimal_thresholds)
    return optimal_thresholds


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


def Trainer(model,
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
            seed,
            test_0=None,
            test_1=None,
            test_2=None,
            test_3=None,
            test_4=None,
            test_random=None):
    
    # for data, labels in test_0:
    #     b = data[0].cpu().data.numpy()
    #     np.savetxt("./"+str(0)+".csv", b, delimiter=',', fmt='%s')
        
    # for data, labels in test_1:
    #     b = data[0].cpu().data.numpy()
    #     np.savetxt("./"+str(1)+".csv", b, delimiter=',', fmt='%s')       
    # exit()
    
    logger.debug("Pre-training started ....")
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)

    # Pre-training
    best_performance = None
    best_F1 = None
    best_auc = None
    for epoch in range(1, args.pretrain_epoch + 1):

        total_loss, total_cl_loss, total_rb_loss = model_pretrain(model, model_optimizer, model_scheduler, train_dl,
                                                                  configs, args, device)

        logger.debug(
            f'Pre-training Epoch: {epoch}\t Train Loss: {total_loss:.4f}\t CL Loss: {total_cl_loss:.4f}\t RB Loss: {total_rb_loss:.4f}\n')

        chkpoint = {'seed': seed, 'epoch': epoch, 'train_loss': total_loss, 'model_state_dict': model.state_dict()}
        # torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_ep{epoch}.pt')) # 保存的是预训练的模型

        # pretrain 
        if epoch % 2 == 0:

            # Fine-tuning
            logger.debug("Fine-tuning started ....") # 构建调优模型
            ft_model, ft_classifier, ft_model_optimizer, ft_classifier_optimizer, ft_scheduler = build_model(args,
                                                                                                             args.lr,
                                                                                                             configs,
                                                                                                             device,
                                                                                                             chkpoint)

            for ep in range(1, args.finetune_epoch + 1): # 选择acc和F1作为调优的参数
                valid_loss, valid_acc, valid_auc, valid_prc, emb_finetune, label_finetune, F1 ,valid_thresholds,val_model= model_finetune(ft_model,
                                                                                                               valid_dl,
                                                                                                               device,
                                                                                                               ft_model_optimizer,
                                                                                                               ft_scheduler,
                                                                                                               classifier=ft_classifier,
                                                                                                               classifier_optimizer=ft_classifier_optimizer)
                if best_F1 is None:
                    best_F1 = F1
                else:
                    if F1 > best_F1:
                        best_F1 = F1
                        logger.debug(
                                'EP%s - Better Testing:  F1 = %.4f' % (
                                ep, F1))

                        chkpoint = {'seed': seed, 'epoch': epoch, 'train_loss': total_loss,
                                        'model_state_dict': ft_model.state_dict(),
                                        'th':valid_thresholds}
                        # torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_best_val.pt'))

                # 调优之后返回F1,th
                if ep % args.log_epoch == 0: # finetune 5次 ，test一次
                    
                    # 打印调优之后的数据，debug中输出的这个F1分数
                    logger.debug(
                        f'\nEpoch : {ep}\t | \t  finetune Loss: {valid_loss:.4f}\t | \tAcc: {valid_acc:2.4f}\t | \tF1: {F1:0.4f}')
                    
                    performance= model_test(ft_model,test_dl,device,classifier=ft_classifier,th = valid_thresholds)
                    # performance_0 , auc_list_0 = model_test(ft_model,test_0,device,classifier=ft_classifier,th = valid_thresholds,layout =0 )
                    # performance_1 ,auc_list_1= model_test(ft_model,test_1,device,classifier=ft_classifier,th = valid_thresholds,layout =1)
                    # performance_2 ,auc_list_2= model_test(ft_model,test_2,device,classifier=ft_classifier,th = valid_thresholds,layout = 2)
                    # performance_3 ,auc_list_3= model_test(ft_model,test_3,device,classifier=ft_classifier,th = valid_thresholds,layout =3)
                    # performance_4 ,auc_list_4= model_test(ft_model,test_4,device,classifier=ft_classifier,th = valid_thresholds,layout =4)
                    # performance_random ,auc_list_random = model_test(ft_model,test_random,device,classifier=ft_classifier,th = valid_thresholds,layout =6)
                    
                    
                    # 0 : 3 x 4
                    # 1 : 3 x 4 + Ⅱ
                    # 2 : 3 x 4 + Ⅱ + V1
                    # 3 : 6 x 2 
                    # 4 : 6 x 2+Ⅱ
                    # 5 : 12 
                    # 6 : 各种排布随机出现 -l --layout 
                    # print('Testing : 3 x 4 ==%.4f'%(performance_0[3]))
                    # print('Testing : 3 x 4 + 2 ==%.4f'%(performance_1[3]))
                    # print('Testing : 3 x 4 + 2,6 ==%.4f'%(performance_2[3]))
                    # print('Testing : 6 x 2 ==%.4f'%(performance_3[3]))
                    # print('Testing : 6 x 2 + 2 ==%.4f'%(performance_4[3]))
                    # print('Testing : random ==%.4f'%(performance_random[3]))
                    
                    
                    # save our data
                    
                    if best_performance is None:
                        best_performance = performance
                    else:
                        if performance[3] > best_performance[3]:
                            best_performance = performance
                            logger.debug(
                                'EP%s - Better Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
                                ep, performance[0], performance[1], performance[2], performance[3]))

                            chkpoint = {'seed': seed, 'epoch': epoch, 'train_loss': total_loss,
                                        'model_state_dict': model.state_dict(),
                                        'th':valid_thresholds}
                            # torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_best.pt'))
                            
                            
            logger.debug("Fine-tuning ended ....")
            logger.debug("=" * 100)
            logger.debug('EP%s - Best Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
            epoch, best_performance[0], best_performance[1], best_performance[2], best_performance[3]))
            logger.debug(best_performance[3]) # F1 输出一个F1
            logger.debug(best_performance[6]) 
            logger.debug("=" * 100)
            
            # 保存一下这个数据
            path_auc = './code/chaoyang/chaoyang_knn.xlsx' # =====================只保存了这个
            try:
                existing_df = pd.read_excel(path_auc) 
            except FileNotFoundError:
                print("文件 'existing_file.xlsx' 不存在，将创建新文件。")
                df1 = pd.DataFrame([best_performance[6]])
                df1.to_excel(path_auc, index=False, header=False)
            
            
            df2 = pd.DataFrame([best_performance[6]])
            existing_df = pd.read_excel(path_auc, header=None)
            combined_df = pd.concat([existing_df, df2], ignore_index=True)
            combined_df.to_excel(path_auc, index=False, header=False) 

    return best_performance


def model_pretrain(model, model_optimizer, model_scheduler, train_loader, configs, args, device):
    total_loss = []
    total_cl_loss = []
    total_rb_loss = []

    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        model_optimizer.zero_grad()

        data_masked_m, mask = data_transform_masked4cl(data, args.masking_ratio, args.lm, args.positive_nums)
        data_masked_om = torch.cat([data, data_masked_m], 0)

        data, labels, data_masked_om = data.float().to(device), labels.float().to(device), data_masked_om.float().to(
            device)
        
        # print(data.shape) # n,1,178
        # print(labels.shape) # n,23
        # Produce embeddings of original and masked samples 
        loss, loss_cl, loss_rb = model(data_masked_om, pretrain=True)
        

        loss.backward()
        model_optimizer.step()

        total_loss.append(loss.item())
        total_cl_loss.append(loss_cl.item())
        total_rb_loss.append(loss_rb.item())

    total_loss = torch.tensor(total_loss).mean()
    total_cl_loss = torch.tensor(total_cl_loss).mean()
    total_rb_loss = torch.tensor(total_rb_loss).mean()

    model_scheduler.step()

    return total_loss, total_cl_loss, total_rb_loss


def model_finetune(model, val_dl, device, model_optimizer, model_scheduler, classifier=None, classifier_optimizer=None):
    model.train()
    classifier.train()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = MultiLabelFocalLoss()
    outs = np.array([])
    trgs = np.array([])
    
    pred_score = []
    true_label = []
    

    for data, labels in val_dl:
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        data, labels = data.float().to(device), labels.long().to(device)

        # Produce embeddings
        h, z = model(data)

        # Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test
        fea_concat = h

        # print(h.shape)# 1280
        
        predictions = classifier(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        
        # print(predictions)
        # print(predictions.shape) # n,23 没问题
        # print(labels.shape) # n,23 torch
        # print(type(labels))
        # print(type(predictions))        
        a = torch.sigmoid(predictions)
        loss = criterion(predictions, labels)
        

        
        # ours ==========================
        # print(a.shape)
        # print(labels.shape)
        
        pred_numpy = a # 概率值pred_numpy a.detach().cpu().numpy()
        total_loss.append(loss.item())
        pred_score.append(a.cpu().data.numpy())
        true_label.append(labels.cpu().data.numpy())
        
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        
        # pred = predictions.max(1, keepdim=True)[1] # pred 是预测的label
        
        outs = np.append(outs, pred_numpy.detach().cpu().numpy())
        trgs = np.append(trgs, labels.data.cpu().numpy())

    
    # find best 阈值，来选择最优的阈值，计算F1分数
    
    labels_numpy = labels.detach().cpu().numpy() # 真实label 
    # pred_numpy # 预测概率
    
    val_pred_score = np.concatenate(pred_score)
    val_true = np.concatenate(true_label)
    
    
    # print(val_pred_score.shape) # 2112,23
    # print(val_true .shape) # 2112,23
    
    
    # 选择最优的阈值
    bs_thresholds = find_optimal_thresholds(val_true,val_pred_score) # score and true 
    val_pred_labels = np.zeros_like(val_pred_score)
    for i in range(23):
        val_pred_labels[:, i] = (val_pred_score[:, i] >= bs_thresholds[i])
        
        
    F1 = f1_score(val_true, val_pred_labels, average='macro')  # F1输入的应该是标签不是score，auc得到的是score
    print("F1_macro_finetune:{}".format(F1))
    
    
    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = 0 # torch.tensor(total_acc).mean()  # average acc
    total_auc = 0 # torch.tensor(total_auc).mean()  # average auc
    total_prc = 0 # torch.tensor(total_prc).mean()

    model_scheduler.step(total_loss)

    return total_loss, total_acc, total_auc, total_prc, fea_concat_flat, trgs, F1 ,bs_thresholds,model


def model_test(model, test_dl, device, classifier=None,th=None,layout=None):
    # th 来自验证集的阈值
    # 0 : 3 x 4
    # 1 : 3 x 4 + Ⅱ
    # 2 : 3 x 4 + Ⅱ + V1
    # 3 : 6 x 2 
    # 4 : 6 x 2+Ⅱ
    # 5 : 12 
    # 6 : 各种排布随机出现 -l --layout 
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
            
            # b = data[0].cpu().data.numpy()
            # np.savetxt("./"+str(layout)+".csv", b, delimiter=',', fmt='%s')
            
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
    
    # save score and th for chaoyang AF
    need_save = {
        'score' : test_pred_score,
        'th':th
    }
    torch.save(need_save,'./code/chaoyang/score_knn.pt')
     # save score and th for chaoyang AF


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
    
    
    
    # 保存一下这个数据
    # try:
    #     existing_df = pd.read_excel('./code/table/sub_ex1_1fold_train.xlsx')
    # except FileNotFoundError:
    #     print("文件 'existing_file.xlsx' 不存在，将创建新文件。")
    #     df1 = pd.DataFrame([list_auc])
    #     df1.to_excel('./code/table/sub_ex1_1fold_train.xlsx', index=False, header=False)
    
    
    # df2 = pd.DataFrame([list_auc])
    # existing_df = pd.read_excel('./code/table/sub_ex1_1fold_train.xlsx', header=None)
    # combined_df = pd.concat([existing_df, df2], ignore_index=True)
    # combined_df.to_excel('./code/table/sub_ex1_1fold_train.xlsx', index=False, header=False) 

    precision = 0
    recall = 0
    acc = 0
    total_loss = torch.tensor(total_loss).mean()
    total_acc = 0
    total_auc = 0
    total_prc = 0

    performance = [acc * 100, precision * 100, recall * 100, F1, total_auc * 100, total_prc * 100,list_auc]
    # 最终选择的是最好的acc的性能部分

    emb_test_all = torch.cat(tuple(emb_test_all))
    
    return  performance # total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs,