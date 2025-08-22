import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args, learning_rate):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))

    if args.lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, pred_len):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, pred_len)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, pred_len)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, pred_len):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + f'checkpoint_{pred_len}.pth')
        self.val_loss_min = val_loss
        

def plt_ecg_12lead(data):
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
    plt.savefig("./ecg_12lead.png")