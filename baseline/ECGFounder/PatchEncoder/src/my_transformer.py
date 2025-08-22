"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from scipy import signal
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    roc_auc_score, f1_score
from helper_code import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
# from tensorboardX import SummaryWriter
# from torchsummary import summary
import pickle
import math
from vit_transformer import *
from S3 import S3
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # mask = []
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]\
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print(f"scale {self.scale}")


        # ==============================绘制attention case
        # 增加大小
        # q1, k1, v1 = qkv[0] * 10000000000000000000000000000000000000.0, qkv[1] * 10000000000000000000000000000000000000.0, qkv[2] * 10000000000000000000000000000000000000.0
        # attn_1 = (q1 @ k1.transpose(-2, -1)) * self.scale

        # print(q1[0])
        # attn_min = attn_1.min(dim=-1, keepdim=True)[0]
        # attn_max = attn_1.max(dim=-1, keepdim=True)[0]
        # attn_norm = (attn_1 - attn_min) / (attn_max - attn_min + 1e-8)
        # torch.save(attn_norm,"./case_study/attention/last_attention_norm.pt")


        # raw_attn = attn_1.clone().detach() # 保存注意力权重
        # print("ran_atten")
        # print(raw_attn.shape) #
        # print(raw_attn[0,0,:,:])

        # # 保存最后一层的attention 
        # torch.save(raw_attn,"./case_study/attention/last_attention.pt") # 全为0？这个attention



        # ==============================绘制attention case


        # print("========")
        # mask 掉缺失的值，全部降为-1e9 就不影响结果了
        mask = torch.all(x == 0, dim=(2)) # B,num_patches+1
        attention_mask = mask.unsqueeze(1).unsqueeze(-1) # [B,1,1,num_patches+1] 256,1,1,181
        # print(attention_mask.shape)
        
        attn = torch.where(attention_mask == 0, torch.full_like(attn, -1e9), attn)# softmax
        # scores.masked_fill_(attn_mask, -1e9)
        
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) # b,
        # print(x.shape)


        # torch.save(x,"./case_study/attention/last_attention_proj.pt") # 保存第一层的映射
        # exit()

        
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Learnable2DRelativePositionalEmbedding(nn.Module):
    def __init__(self, num_positions_h, num_positions_w, embedding_dim):
        super(Learnable2DRelativePositionalEmbedding, self).__init__()
        
        # 计算相对位置的最大范围
        max_relative_distance_h = num_positions_h - 1
        max_relative_distance_w = num_positions_w - 1
        
        # 创建可学习的相对位置嵌入矩阵
        self.relative_embeddings_h = nn.Embedding(2 * max_relative_distance_h + 1, embedding_dim)
        self.relative_embeddings_w = nn.Embedding(2 * max_relative_distance_w + 1, embedding_dim)
        
        # 初始化嵌入矩阵
        nn.init.uniform_(self.relative_embeddings_h.weight, -0.1, 0.1)
        nn.init.uniform_(self.relative_embeddings_w.weight, -0.1, 0.1)
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        
        # 生成相对位置索引
        range_height = torch.arange(height, device=x.device)
        range_width = torch.arange(width, device=x.device)
        
        relative_coords_h = range_height.view(-1, 1) - range_height.view(1, -1)
        relative_coords_w = range_width.view(-1, 1) - range_width.view(1, -1)
        
        # 将相对位置索引映射到嵌入矩阵的索引范围
        relative_coords_h = relative_coords_h + (height - 1)
        relative_coords_w = relative_coords_w + (width - 1)
        
        # 获取相对位置嵌入向量
        relative_embed_h = self.relative_embeddings_h(relative_coords_h)  # (height, height, embedding_dim)
        relative_embed_w = self.relative_embeddings_w(relative_coords_w)  # (width, width, embedding_dim)
        
        # 将嵌入向量扩展到与输入x相同的形状
        relative_embed_h = relative_embed_h.unsqueeze(0).unsqueeze(3).expand(batch_size, -1, -1, width, -1)
        relative_embed_w = relative_embed_w.unsqueeze(0).unsqueeze(1).expand(batch_size, height, -1, -1, -1)
        
        # 将高度和宽度的相对嵌入向量相加
        relative_pos_embed = relative_embed_h + relative_embed_w  # (batch_size, height, width, height, width, embedding_dim)
        
        return relative_pos_embed

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

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None,
                 num_patch = 15,channel = 12
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        
        
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = num_patch * channel # [12 x 15]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # [1,180+1,768] no positional 
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02) 
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        
        # 输入
        # [B,channel x num_patch,dim] 256,180,768
        # print("==============")
        # print(x.shape) 
        # print("==============")
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # 添加一个class token 
        # print("==============")
        # print(x.shape) # [B, 181, 768]
        # print("==============")
        
        x = self.pos_drop(x + self.pos_embed) 
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # print(x.shape) # [batch,11] 
        return x

class DualPositionalEmbedding(nn.Module):
    def __init__(self, num_channels, num_patches, dim):
        """
        双维度可学习位置嵌入

        Args:
            num_channels (int): 输入的通道数
            num_patches (int): 输入的patch数量
            dim (int): 输入张量的最后一维的特征维度
        """
        super(DualPositionalEmbedding, self).__init__()

        # 为channel维度创建可学习的position embedding，形状为 (1, num_channels, 1, dim)
        self.channel_pos_embedding = nn.Parameter(torch.randn(1, num_channels, 1, dim))

        # 为numPatch维度创建可学习的position embedding，形状为 (1, 1, num_patches, dim)
        self.patch_pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, dim))

    def forward(self, x):
        """
        前向传播，将位置嵌入添加到输入中

        Args:
            x (torch.Tensor): 输入特征，形状为 (batch_size, num_channels, num_patches, dim)

        Returns:
            torch.Tensor: 加入位置嵌入后的张量
        """
        # 将channel和patch的position embedding分别加到输入张量的对应维度上
        x = x + self.channel_pos_embedding + self.patch_pos_embedding

        return x
    
class VisionTransformer_2dembedding(nn.Module):
    def __init__(self, num_classes=11,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None,
                 num_patch = 15,channel = 12
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        
        
        super(VisionTransformer_2dembedding, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = num_patch * channel # [12 x 15]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # [1,180+1,768] no positional 
        
        # 2d positional embedding
        self.channel_pos_embedding = nn.Parameter(torch.zeros(1, channel, 1, embed_dim))
        self.patch_pos_embedding = nn.Parameter(torch.zeros(1, 1, num_patch, embed_dim))
        
        
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02) # no embedding
        nn.init.trunc_normal_(self.channel_pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.patch_pos_embedding, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # self.apply(_init_vit_weights)

    def forward_features(self, x):
        # 输入 [B,12,15,768]
        B,C,N,D = x.shape
        x = x+self.channel_pos_embedding+self.patch_pos_embedding 
        x = x.reshape(B,C*N,D)
        
        # add s3 layers
        
        # print("==============")
        # print(x.shape) # [B,180,768]
        # print("==============")
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # 添加一个class token 

        
        # x = self.pos_drop(x + self.pos_embed) # no embedding
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # print(x.shape) # [batch,11] 
        return x

class VisionTransformer_2dembedding_v2(nn.Module):
    def __init__(self, num_classes=11,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None,
                 num_patch = 15,channel = 12
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        
        
        super(VisionTransformer_2dembedding_v2, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = num_patch * channel # [12 x 15]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # [1,180+1,768] no positional 
        
        # 2d positional embedding temporal embedding and lead embedding 
        self.channel_pos_embedding = nn.Parameter(torch.zeros(1, channel, 1, embed_dim))
        self.patch_pos_embedding = nn.Parameter(torch.zeros(1, 1, num_patch, embed_dim))
        
        # S3
        self.s3_layers = S3(num_layers=3, initial_num_segments=4, shuffle_vector_dim=1, segment_multiplier=2) # S3 

        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # no embedding
        nn.init.trunc_normal_(self.channel_pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.patch_pos_embedding, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # self.apply(_init_vit_weights)

    def forward_features(self, x):
        # 输入 [B,12,15,768]
        B,C,N,D = x.shape
        # 记录一下为0的mask数据
        mask = torch.all(x == 0, dim=(3)) 
        mask = mask.unsqueeze(-1)
        # ==============================
        x = x+self.channel_pos_embedding+self.patch_pos_embedding # origin

        # x = x + self.channel_pos_embedding # wo temporal 
        # x # wo lead and temporal embedding
        # x = x + self.patch_pos_embedding # wo lead 
        # x = x +self.pos_embed # 1d embedding
       
        
        # ==============================
        
        # 1d embedding 
        # x = x.reshape(B,C*N,D)
        # x = x+self.pos_embed
        # x = x.reshape(B,C,N,D)
        # # ==============================
        # print(mask.shape)
        # print(x.shape)
        
        # =====================
        
        x = x * (~mask)
        x = x.reshape(B,C*N,D) # [B,C*N,D] B,180,768
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # add s3 layers
        
        # print(x.shape) # [B,181,768]
        x = self.s3_layers(x)
        
        # 添加一个class token
        # x = self.pos_drop(x + self.pos_embed) # no embedding
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # print(x.shape) # [batch,11] 
        return x
def _init_vit_weights(m):
        nn.init.ones_(m.weight)


class CombinedModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        # x = torch.from_numpy(a).to("cuda:0")
        # tensor = torch.from_numpy(a)
        # x = tensor.cuda(3)
        # print(x.shape)
        # input [B,C,N,P]
        mask = torch.isnan(x).all(dim=-1) # B,C,N [B,12,15]
        # y[torch.isnan(y)] = 0 # 将所有nan 填充为0
        # x = y
        x = torch.where(torch.isnan(x), torch.tensor(0.0), x)
        
        temp_x = x.view(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]) # B x 12 x15
        # add 2channel mask
        
        channel_2_mask = torch.zeros_like(x)
        channel_2_mask[torch.isnan(x)] = 0.0
        channel_2_mask[~torch.isnan(x)] = 1.0
        channel_2_temp = channel_2_mask.reshape(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3])
        PatchEncode_output = self.model1(torch.cat((temp_x.unsqueeze(1), channel_2_temp.unsqueeze(1)), dim=1))
        
        output_x = PatchEncode_output.squeeze(1).view(x.shape[0],x.shape[1],x.shape[2],768)
        output_x[mask] = 0
        
        return self.model2(output_x)