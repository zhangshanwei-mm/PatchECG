import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms, datasets
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import torch.nn as nn
import torch ,os
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score,roc_curve, auc,precision_recall_curve

import torch.nn.functional as F
import cv2
from matplotlib.cm import get_cmap

def slide_and_cut(X,Y,window_size,stride,output_pid=False): # numpy
    out_X = []
    out_Y = []
    out_pid = [] # 
    n_sample = X.shape[0] # 
    for i in range(n_sample):
        temp_x = X[i]
        temp_y = Y[i]
        for j in range(0,temp_x.shape[1]-window_size,stride):
            padded_array = np.zeros_like(temp_x) # 新建一个同维度的
            padded_array[:,j:j+window_size] = temp_x[:,j:j+window_size] 
            out_X.append(padded_array)
            out_Y.append(temp_y)
            out_pid.append(i) # 返回唯一标识符
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)
    
def slide_and_cut_to_13channal(X,Y,window_size,stride,output_pid=False): # 输入为numpy,n,12,length
    length = X.shape[2] # length
    out_X = []
    out_Y = []
    out_pid = [] # 唯一标识符
    n_sample = X.shape[0] # 样本数量
    for i in range(n_sample):
        temp_x = X[i]
        temp_y = Y[i]
        for j in range(0,length-window_size,stride):
            padded_array = np.zeros((12,length)) # 新建一个同维度的全0
            padded_array[:,j:j+window_size] = temp_x[:,j:j+window_size] # (12,length)
            # np.savetxt("./taoutput.csv", padded_array, delimiter=',', fmt='%r')
            channel_1 = padded_array.reshape(1,12*length) # (1,12 x length)
            print(channel_1.shape)
            
            channel_2 = np.zeros((1,12*length)) # (1,12 x length)
            channel_2[:,j:j+window_size] = 1
            
            channel_3 = np.zeros((1,12*length))
            channel_3[:,j+1*length:j+1*length+window_size] = 1
            
            channel_4 = np.zeros((1,12*length))
            channel_4[:,j+2*length:j+2*length+window_size] = 1
            
            channel_5 = np.zeros((1,12*length))
            channel_5[:,j+3*length:j+3*length+window_size] = 1
            
            channel_6 = np.zeros((1,12*length))
            channel_6[:,j+4*length:j+4*length+window_size] = 1
            
            channel_7 = np.zeros((1,12*length))
            channel_7[:,j+5*length:j+5*length+window_size] = 1
            
            channel_8 = np.zeros((1,12*length))
            channel_8[:,j+6*length:j+6*length+window_size] = 1
            
            channel_9 = np.zeros((1,12*length))
            channel_9[:,j+7*length:j+7*length+window_size] = 1
            
            channel_10 = np.zeros((1,12*length))
            channel_10[:,j+8*length:j+8*length+window_size] = 1
            
            channel_11 = np.zeros((1,12*length))
            channel_11[:,j+9*length:j+9*length+window_size] = 1
            
            channel_12 = np.zeros((1,12*length))
            channel_12[:,j+10*length:j+10*length+window_size] = 1
            
            channel_13 = np.zeros((1,12*length))
            channel_13[:,j+11*length:j+11*length+window_size] = 1
            
            data_temp = np.vstack((channel_1,
                              channel_2,
                              channel_3,
                              channel_4,
                              channel_5,
                              channel_6,
                              channel_7,
                              channel_8,
                              channel_9,
                              channel_10,
                              channel_11,
                              channel_12,
                              channel_13))
                        
             
            out_X.append(data_temp)
            out_Y.append(temp_y)
            out_pid.append(i) # 返回唯一标识符
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)

def mask_13channal(signal):
    """
        signal:(12,length)
        mask_window_size: 随机mask长度
        mask_start: mask的起始位置
        每一个通道都随机mask且
    """
    # mask_value = [100, 200, 300, 400, 500]
    length = 1000 # signal 的长度
    # 随机生成12个值
    # mask_window_size = np.random.choice(mask_value, size=12) # 12
    mask_window_size = np.random.randint(0, 500, size=12) # 12个0-500之间的数字
    mask_start = []
    for i in range(12):
        mask_start.append(np.random.randint(0, length-mask_window_size[i]))
    
    channel_1 = np.array(signal[0]).reshape(1000)
    channel_1[mask_start[0]:mask_start[0]+mask_window_size[0]] = 0
    
    channel_2 = np.array(signal[1]).reshape(1000)
    channel_2[mask_start[1]:mask_start[1]+mask_window_size[1]] = 0
    
    channel_3 = np.array(signal[2]).reshape(1000)
    channel_3[mask_start[2]:mask_start[2]+mask_window_size[2]] = 0
    
    channel_4 = np.array(signal[3]).reshape(1000)
    channel_4[mask_start[3]:mask_start[3]+mask_window_size[3]] = 0
    
    channel_5 = np.array(signal[4]).reshape(1000)
    channel_5[mask_start[4]:mask_start[4]+mask_window_size[4]] = 0
    
    channel_6 = np.array(signal[5]).reshape(1000)
    channel_6[mask_start[5]:mask_start[5]+mask_window_size[5]] = 0
    
    channel_7 = np.array(signal[6]).reshape(1000)
    channel_7[mask_start[6]:mask_start[6]+mask_window_size[6]] = 0
    
    channel_8 = np.array(signal[7]).reshape(1000)
    channel_8[mask_start[7]:mask_start[7]+mask_window_size[7]] = 0
    
    channel_9 = np.array(signal[8]).reshape(1000)
    channel_9[mask_start[8]:mask_start[8]+mask_window_size[8]] = 0
    
    channel_10 = np.array(signal[9]).reshape(1000)
    channel_10[mask_start[9]:mask_start[9]+mask_window_size[9]] = 0
    
    channel_11 = np.array(signal[10]).reshape(1000)
    channel_11[mask_start[10]:mask_start[10]+mask_window_size[10]] = 0
    
    channel_12 = np.array(signal[11]).reshape(1000)
    channel_12[mask_start[11]:mask_start[11]+mask_window_size[11]] = 0
    
    channel_signal = np.hstack((channel_1,
                                channel_2,
                                channel_3,
                                channel_4,
                                channel_5,
                                channel_6,
                                channel_7,
                                channel_8,
                                channel_9,
                                channel_10,
                                channel_11,
                                channel_12
                                ))
    # print(channel_signal.shape)
    mask_1 = np.ones(1000) # mask
    mask_1[mask_start[0]:mask_start[0]+mask_window_size[0]] = 0 # 取mask
    mask_1 = np.pad(mask_1,((0,11000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_2 = np.ones(1000) # mask
    mask_2[mask_start[1]:mask_start[1]+mask_window_size[1]] = 0 # 取mask
    mask_2 = np.pad(mask_2,((1000,10000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_3 = np.ones(1000) # mask
    mask_3[mask_start[2]:mask_start[2]+mask_window_size[2]] = 0 # 取mask
    mask_3 = np.pad(mask_3,((2000,9000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_4 = np.ones(1000) # mask
    mask_4[mask_start[3]:mask_start[3]+mask_window_size[3]] = 0 # 取mask
    mask_4 = np.pad(mask_4,((3000,8000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_5 = np.ones(1000) # mask
    mask_5[mask_start[4]:mask_start[4]+mask_window_size[4]] = 0 # 取mask
    mask_5 = np.pad(mask_5,((4000,7000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_6 = np.ones(1000) # mask
    mask_6[mask_start[5]:mask_start[5]+mask_window_size[5]] = 0 # 取mask
    mask_6 = np.pad(mask_6,((5000,6000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_7 = np.ones(1000) # mask
    mask_7[mask_start[6]:mask_start[6]+mask_window_size[6]] = 0 # 取mask
    mask_7 = np.pad(mask_7,((6000,5000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_8 = np.ones(1000) # mask
    mask_8[mask_start[7]:mask_start[7]+mask_window_size[7]] = 0 # 取mask
    mask_8 = np.pad(mask_8,((7000,4000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_9 = np.ones(1000) # mask
    mask_9[mask_start[8]:mask_start[8]+mask_window_size[8]] = 0 # 取mask
    mask_9 = np.pad(mask_9,((8000,3000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_10 = np.ones(1000) # mask
    mask_10[mask_start[9]:mask_start[9]+mask_window_size[9]] = 0 # 取mask
    mask_10 = np.pad(mask_10,((9000,2000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_11 = np.ones(1000) # mask
    mask_11[mask_start[10]:mask_start[10]+mask_window_size[10]] = 0 # 取mask
    mask_11 = np.pad(mask_11,((10000,1000)),'constant',constant_values = (0,0)) # 填充完整
    
    mask_12 = np.ones(1000) # mask
    mask_12[mask_start[11]:mask_start[11]+mask_window_size[11]] = 0 # 取mask
    mask_12 = np.pad(mask_12,((11000,0)),'constant',constant_values = (0,0)) # 填充完整
    
    data = np.vstack((channel_signal,
                      mask_1,
                      mask_2,
                      mask_3,
                      mask_4,
                      mask_5,
                      mask_6,
                      mask_7,
                      mask_8,
                      mask_9,
                      mask_10,
                      mask_11,
                      mask_12))
    # np.savetxt("./taoutput.csv", data.T, delimiter=',', fmt='%r')
    # return (13,12000)
    return data
    
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

def signal_to_image(signal):
    """
        input : signal (12,1000)
        output : picture ()
    """
    
    # 创建12个子图
    fig, axes = plt.subplots(12, 1, figsize=(10, 10), sharex=True)
    for i in range(12):
        ax = axes[i]
        # 将0值替换为NaN
        ecg_plot_data = np.where(signal[i] == 0, np.nan, signal[i])
        ax.plot(ecg_plot_data, color='black', linewidth=0.5)  # 绘制黑色线条
        ax.axis('off')  # 关闭坐标轴

     # 调整子图间距
    plt.subplots_adjust(hspace=0, wspace=0, top=1, bottom=0, left=0, right=1)
        
    # 将图像渲染为RGB格式
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.savefig("./12_1000picture.png") # 保存图片
    plt.close(fig)
    return img_array

def mask_12channal(signal,random):
    """
        input:12,1000 ndarray
        random :
        output:12,1000 ndarray
        return : 12,1000
    """
    np.random.seed(random)
    length = 1000 # signal 的长度
    # 随机生成12个值
    mask_window_size = np.random.randint(0, 500, size=12)
    mask_start = []
    for i in range(12):
        mask_start.append(np.random.randint(0, length-mask_window_size[i]))
        
    for i in range(12):
        signal[i,mask_start[i]:mask_start[i]+mask_window_size[i]] = np.nan
        
    return signal

def mask_12channal_maks_and_signal(signal):
    """
        Args:
            signal:input signal
        return:
            signal,msk (same dim)
    """
    length = 1000 # signal 的长度
    msk = np.ones_like(signal)
    # 随机生成12个值
    mask_window_size = np.random.randint(0, 750, size=12)
    mask_start = []
    for i in range(12):
        mask_start.append(np.random.randint(0, length-mask_window_size[i]))
        
    for i in range(12):
        signal[i,mask_start[i]:mask_start[i]+mask_window_size[i]] = 0
        msk[i,mask_start[i]:mask_start[i]+mask_window_size[i]] = 0

    return signal,msk


def signal_to_image_raw(signal):
    """
        input : shape(12,1000)
        output : picture shape(12,1000,3)   
        将心电图转为灰度图像，数据维度不发生变化 
    """
    # 将信号数据直接转为图像维度不变化
    picture = np.zeros((signal.shape[0],signal.shape[1]),dtype=np.uint8)
    # 标准化数据
    ecg_normalized = ((signal - signal.min()) / (signal.max() - signal.min()) * 255).astype(np.uint8)
    picture = np.stack([ecg_normalized] * 3, axis=-1)
    
    # 保存图像
    # image = Image.fromarray(picture)
    # 保存图片
    # image.save('./12_1000ecg_gray1_stand.png')
    return picture

# 找最好的阈值
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

    for i in tqdm(range(n_task)):
        best_ba = -1  
        best_thresh = 0.5  
        for thresh in np.linspace(0.01, 0.99, 99):  
            pred_labels = (pred[:, i] > thresh).astype(int)
            ba = balanced_accuracy_score(gt[:, i], pred_labels)  
            if ba > best_ba:
                best_ba = ba
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)
    print("best thresholds:\n")
    print(optimal_thresholds)
    return optimal_thresholds

def my_find_optimal_thresholds(y_true,y_scores):
    """
    Args:
        y_true: Ground truth labels (numpy array)
        y_scores:Prediction score(numpy array)
    
    return:
        best_thresholds (list) ,shape (n_classes,)
    """
    n_classes = y_true.shape[1] # 11
    best_thresholds = np.zeros(n_classes)
    # 为每个标签计算最佳阈值
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_f1_index = np.argmax(f1_scores)
        best_thresholds[i] = thresholds[best_f1_index]
        print(f'Best Threshold for label {i}: {best_thresholds[i]}, Best F1-score: {f1_scores[best_f1_index]}')
    
    return best_thresholds.tolist()
    
# 画图部分
def figure_ROC(y_true,y_score):
    """
        y_true:
        y_score:
    """
    
    n_classes = y_true.shape[1] # 获取多标签的数量
    
    print("There are {} classes".format(n_classes))
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        # 计算AUC
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i+1} (AUC = {roc_auc:.2f})')
    # 绘制ROC曲线
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve for Multi-label Classification')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('./picture/roc_11labels.jpg')

def figure_Loss(loss,path):
    """
        Args:
            loss : list()
            path : 保存的路径
    """
    plt.figure()
    plt.plot(loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(path)

def figure_case_mask_origin(signal):
    """
        input : (1,12,length) ecg
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    offset = -2.8  # lead的偏移量
    cmap = get_cmap('tab20')  # 使用 'tab20' 颜色映射
    colors = cmap.colors[:12]  # 选择前 12 种颜色
    for i in range(12):
        ax.plot(signal[i] + i * offset, label=f'Lead {i+1}',color=colors[i])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('./picture/case/12-lead-ecg_origin.png')
    
def figure_case_mask_ecg_patch_vit(signal):
    """
        input : (1,12,length) ecg
        
    """
    cmap = get_cmap('tab20')  # 使用 'tab20' 颜色映射
    colors = cmap.colors[:12]  # 选择前 12 种颜色
    
    signal_data = signal
    fig, axes = plt.subplots(12, 8, figsize=(12, 36))
    for i in range(12):
        for j in range(8):
             # 计算当前子图对应的信号数据范围
            start_idx = j * 125
            end_idx = (j + 1) * 125
            temp_data = signal_data[i, start_idx:end_idx] # 250 
            # 绘制当前子图的信号数据
            axes[i, j].plot(temp_data,color=colors[i])  
            axes[i, j].set_xticks([])
            axes[i, j].set_xlim(0, 124)  # 设置 x 轴范围为 0 到 249
            axes[i, j].set_yticks([])

            
            
            # 保存当前子图为单独的图像文件
            fig_single = plt.figure(figsize=(4, 4))
            ax_single = fig_single.add_subplot(111)
            ax_single.plot(temp_data, color=colors[i])
            ax_single.set_xticks([])
            ax_single.set_xlim(0, 124)
            ax_single.set_yticks([])
            # ax_single.set_title(f"Signal {i+1}, Segment {j+1}")
            fig_single.tight_layout()
            fig_single.savefig(f"./picture/case/signal_{i+1}_segment_{j+1}.png")
            plt.close(fig_single)
            
    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图像
    plt.savefig('./picture/case/12-lead-ecg.png')

    
# loss
# 没什么效果,效果一般
class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss. 

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2

    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

from torch.autograd import Variable
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
        
        
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # 分别计算正负例的概率
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # 非对称裁剪
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)  # 给 self.xs_neg 加上 clip 值

        # 先进行基本交叉熵计算
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
                
            # 以下 4 行相当于做了个并行操作
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


# KD loss
class KdLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(self, outputs, labels, teacher_outputs):
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / self.T, dim=1),
                                                      F.softmax(teacher_outputs / self.T, dim=1)) * (
                          self.alpha * self.T * self.T) + F.binary_cross_entropy_with_logits(outputs, labels) * (
                          1. - self.alpha)
        return kd_loss

# 保存模型
def save_checkpoint(f1_macro, model, optimizer, epoch):
    print('Model Saving...')
    model_state_dict = model.state_dict()
    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'F1_macro': f1_macro,
    }, os.path.join('checkpoints_picture_resnet_v2', 'picture_checkpoint_resnet_'+str(epoch)+'.pth')) # 保存路径

def read_boxes_mask_lable(mask_path):
    """
        args:
            pic_path :
            mask_path : .npz
        
        retrun :
            dic :
                bounding_boxes : xmin, ymin, xmax, ymax shape (13,4) or (12,4)
                labels : shape(12) or (13)
                'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II(full)'
                0,1 .....12
    """
    mask = np.load(mask_path+'-0.npz')['mask'].astype(int)
    tr = np.transpose(mask, (0, 2, 1))
    ma = np.array(tr,dtype=np.uint8)
    bounding_boxes = []
    labels = list()
    
    for i in range(13):
    # 找到连通区域
        contours, _ = cv2.findContours(ma[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
            bounding_boxes.append([xmin, ymin, xmax, ymax])
            
    # 输出结果,给每一个boxes加上label
    for i, box in enumerate(bounding_boxes):
        # print(f'Box {i}: xmin={box[0]}, ymin={box[1]}, xmax={box[2]}, ymax={box[3]}')
        if i==0:
            labels.append('I')
        if i==1:
            labels.append('II')
        if i==2:
            labels.append('III')
        if i==3:
            labels.append('AVR')
        if i==4:
            labels.append('AVL')
        if i==5:
            labels.append('AVF')
        if i==6:
            labels.append('V1')
        if i==7:
            labels.append('V2')
        if i==8:
            labels.append('V3')
        if i==9:
            labels.append('V4')
        if i==10:
            labels.append('V5')
        if i==11:
            labels.append('V6')
        if i==12:
            labels.append('II(full)')
    
    # print(labels)
    dic = {} # return 字典的格式
    labels_dic = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II(full)'] # 一共13个类
    label_to_index = {label: idx for idx, label in enumerate(labels_dic)}
    # print("Label to index mapping:")
    # print(label_to_index) # {'I': 0, 'II': 1, 'III': 2, 'AVR': 3, 'AVL': 4, 'AVF': 5, 'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11, 'II(full)': 12}
    
    num_classes = len(labels_dic)
    int_to_one_hot = {idx: np.eye(num_classes)[idx] for idx in range(num_classes)}
    # 创建标签到one-hot向量的映射
    label_to_one_hot = {label: int_to_one_hot[idx] for label, idx in label_to_index.items()}
    # print(label_to_one_hot)

    dx = []
    for i in range(len(labels)):
        # print(label_to_one_hot[labels[i]])
        dx.append(list(label_to_one_hot[labels[i]]))
    # print(dx)
    
    # a = np.array(bounding_boxes)
    # b = np.array(dx)
    # print(a.shape)
    # print(b.shape)
    dic['boxes'] = bounding_boxes
    dic['labels'] = dx
    # print(dic)
    return dic

def encode_labels_lead(data, label_to_index):
# 初始化零矩阵，形状为[数据长度, 标签数量]
    target = torch.zeros((len(data), len(label_to_index)), dtype=torch.float32)

    # 填充矩阵
    for i ,item in enumerate(data):
        # print(i)
        # print(item)
        if item in label_to_index:
            print(item)
            print(label_to_index[i])
            target[i, label_to_index[i]] = 1
    
    return target

def read_boxes_labels(mask_path):
    """
        input : .npz path
        output : 
            dic : 
                "boxes" : list[sample_n,4]
                "labels" : list(sample_n)
                
        comment :
            labels : 'I': 0, 'II': 1, 'III': 2, 'AVR': 3, 'AVL': 4, 'AVF': 5, 
            'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11, 'II(full)': 12
    """
    mask = np.load(mask_path+'-0.npz')['mask'].astype(int)
    tr = np.transpose(mask, (0, 2, 1))
    ma = np.array(tr,dtype=np.uint8)
    bounding_boxes = []
    labels = list()
    
    for i in range(13):
    # 找到连通区域
        contours, _ = cv2.findContours(ma[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
            bounding_boxes.append([xmin, ymin, xmax, ymax])
            
    for i, box in enumerate(bounding_boxes):
        labels.append(i)
        
    dic_label = {}
    dic_label['boxes'] = bounding_boxes
    dic_label['labels'] = labels
    l_box = len(dic_label['boxes'])
    l_label = len(dic_label['labels'])
    if l_box !=l_label:
        print("numbers of boxes != numbers of labels")
    # print(f"numbers of boxes{l_box},numbers of labels{l_label}")
    return dic_label



if __name__ == "__main__":
    # x = 1
    # print(x)
    # a = mask_13channal(x)
    
    # ecg_data = np.random.rand(12, 10)  # 这里用随机数据作为示例
    # print(ecg_data)
    # # 1. 切割数据为 [12, 4, 250]
    # ecg_data_sliced = ecg_data.reshape(12, 2, 5)
    
    
    a = np.zeros((3,4))
    
    print(not np.all(a==0))
    if not np.all(a==0):
        print('all is zero')
    
    
    
    