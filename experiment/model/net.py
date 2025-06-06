"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tltorch import *
import math
from sparsemax import Sparsemax



class Forward_Attention_sparse(nn.Module):
    def __init__(self, patches, dim, attn_drop=0):
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.dim2 = int(1*dim)
        self.qw = nn.Linear(dim, self.dim2, bias=False)
        self.kw = nn.Linear(dim, self.dim2, bias=False)
        self.att_drop=nn.Dropout(attn_drop)
        self.bn=nn.BatchNorm2d(self.patches)
        self.softmax=nn.Softmax(dim=2)
        self.sparsemax=Sparsemax(dim=2)




    def forward(self, x):
        B, N, C = x.shape
        assert (C == self.dim and N == self.patches)
        Q=self.bn( self.qw(input=x).unsqueeze(-1) ).squeeze(-1)
        K=self.bn(self.kw(input=x).unsqueeze(-1) ).squeeze(-1)
        Q = F.normalize(Q, p=2, dim=2)
        K = F.normalize(K, p=2, dim=2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = self.att_drop(scores)
        scores = self.sparsemax(scores)
        return scores


class Forward_Attention_soft(nn.Module):
    def __init__(self, patches, dim, attn_drop=0.2):
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.qw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)
        self.att_drop=nn.Dropout(attn_drop)
        self.bn=nn.BatchNorm2d(self.patches)
        self.softmax=nn.Softmax(dim=2)
        # self.sparsemax=Sparsemax(dim=2)




    def forward(self, x):
        B, N, C = x.shape
        assert (C == self.dim and N == self.patches)
        Q=self.bn( self.qw(input=x).unsqueeze(-1) ).squeeze(-1)
        V=self.bn(self.vw(input=x).unsqueeze(-1) ).squeeze(-1)

        scores = torch.matmul(Q, V.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = self.att_drop(scores)
        scores = self.softmax(scores)
        return scores


class Forward_Multihead_Attention_sparse(nn.Module):
    def __init__(self, patches, embeded_dim, key_size, num_heads, attn_drop=0.3):
        super().__init__()
        self.embeded_dim = embeded_dim
        self.patches = patches
        self.num_heads = num_heads
        self.qw = nn.Linear(embeded_dim, key_size, bias=False)
        self.kw = nn.Linear(embeded_dim, key_size, bias=False)

        self.qv_head_dim = key_size//num_heads

        self.att_drop=nn.Dropout(attn_drop)
        self.bn=nn.BatchNorm2d(self.patches)
        self.sparsemax = Sparsemax(dim=1)
        self.softmax= nn.Softmax(dim=1)


        self.dim_multihead_concate = int(num_heads * patches)

        self.bn = nn.BatchNorm2d(self.num_heads)






    def forward(self, x):
        B, N, C = x.shape
        q = self.qw(input=x)
        k = self.kw(input=x)
        Q = self.bn(q.reshape(B, N, self.num_heads, self.qv_head_dim).transpose(1, 2))
        K = self.bn(k.reshape(B, N, self.num_heads, self.qv_head_dim).transpose(1, 2))

        Q = F.normalize(Q, p=2, dim=3)
        K = F.normalize(K, p=2, dim=3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embeded_dim)

        scores = scores.transpose(1, 2).reshape(B, N, self.dim_multihead_concate)

        scores = self.att_drop(scores)
        scores = self.sparsemax(scores)
        return scores





class FieldFormer_TAP(nn.Module):
    """
    20*20*20->20*20*20
    """

    def __init__(self, dropout_rate=0.01, depth=1):
        super(FieldFormer_TAP, self).__init__()
        e1, e2, e3=int(5), int(5), int(20)
        self.stride = 3
        self.depth = depth
        self.subtensor_size = e1

        self.x = int((20-e1)/self.stride+1)

        self.patches = int((20-e1)/self.stride+1)**3
        self.dim = e2**3

        self.attention_blocks = nn.ModuleList([Forward_Attention_sparse(patches=self.patches, dim=self.dim, attn_drop=dropout_rate) for _ in range(self.depth)])

        self.feature_dim = int((20-e1)/self.stride+1)**2

        self.decoder1 = TCL(input_shape=[self.feature_dim, self.feature_dim, self.feature_dim], rank=[e3, e3, e3])
        self.drop = nn.Dropout(p=dropout_rate)

    def forward_features(self,x):
        for blk in self.attention_blocks:
            x = blk(x)
        return x, x


    def get_U(self):
        U_list =[]
        for i in self.decoder1.parameters():
            U_list.append(i.detach())
        return U_list


    @staticmethod
    def cut_tensor_into_sliding_patches(original_tensor, subtensor_size, stride):
        original_tensor = original_tensor.squeeze(0)
        oritensor_szie = original_tensor.shape
        out_shape = int((oritensor_szie[0] - subtensor_size[0]) / stride + 1)

        num_patch = out_shape ** 3
        result_tensor = torch.empty((num_patch,) + subtensor_size)


        count = 0
        for i in range(0, oritensor_szie[0] - subtensor_size[0] + 1, stride):
            for j in range(0, oritensor_szie[1] - subtensor_size[1] + 1, stride):
                for k in range(0, oritensor_szie[2] - subtensor_size[2] + 1, stride):
                    subtensor = original_tensor[i:i + subtensor_size[0], j:j + subtensor_size[1],
                                k:k + subtensor_size[2]]
                    result_tensor[count] = subtensor
                    count += 1
        return result_tensor.unsqueeze(0)


    def forward(self, patches):
        x1 = patches.cuda()
        x2 = x1.squeeze(0)
        x2 = x2.contiguous().view(self.patches, -1) 
        x2 = x2.unsqueeze(0) 
        x3, att_map = self.forward_features(x2)


        x3 = x3.view(self.x,self.x,self.x,self.x,self.x,self.x)
        x3 = x3.permute(0,3,1,4,2,5)
        x4 = x3.reshape(1,self.feature_dim,self.feature_dim,self.feature_dim)

        x4 = self.drop(x4)


        core = x4.squeeze(0)

        x5 = self.decoder1(x4)

        return torch.tanh(x5), att_map



class FieldFormer_MHTAP(nn.Module):
    """
    20*20*20->20*20*20
    """

    def __init__(self, dropout_rate=0.3, depth=1, num_heads_tuple=(2, 2, 2)):
        super(FieldFormer_MHTAP, self).__init__()
        e1,e2=int(5), int(5)
        self.depth = depth
        self.subtensor_size = e1
        self.stride = 3
        self.patches = int((20-e1)/self.stride+1)**3
        self.N = int((20-e1)/self.stride+1)
        self.dim = int(e2**3)
        self.dim2 = int(64)
        self.num_head1 = num_heads_tuple[0]
        self.num_head2 = num_heads_tuple[1]
        self.num_head3 = num_heads_tuple[2]
        self.num_heads = self.num_head1*self.num_head2*self.num_head3

        self.attention_blocks = nn.ModuleList([Forward_Multihead_Attention_sparse(patches=self.patches, embeded_dim=self.dim, key_size = int(self.num_heads*self.dim2), num_heads= self.num_heads, attn_drop=dropout_rate ) for _ in range(self.depth)])
        self.feature_dim = int((20-e1)/self.stride+1)**2
        #self.encoder = TCL(input_shape=[20, 20, 20], rank=[self.feature_dim, self.feature_dim, self.feature_dim])
        e3 = int(20)
        #self.decoder1 = TCL(input_shape=[self.num_head1*self.feature_dim, self.num_head2*self.feature_dim, self.num_head3*self.feature_dim], rank=[e3, e3, e3])
        self.decoder2 = TCL(input_shape=[self.num_head1*self.N *self.N, self.num_head2*self.N *self.N, self.num_head3*self.N *self.N], rank=[e3, e3, e3])

        self.drop = nn.Dropout(p=dropout_rate)

    def forward_features(self,x):
        for blk in self.attention_blocks:
            x = blk(x)
        return x, x




    @staticmethod
    def cut_tensor_into_sliding_patches(original_tensor, subtensor_size, stride):
        original_tensor = original_tensor.squeeze(0)
        oritensor_szie = original_tensor.shape
        out_shape = int((oritensor_szie[0] - subtensor_size[0]) / stride + 1)
        num_patch = out_shape ** 3
        result_tensor = torch.empty((num_patch,) + subtensor_size)
        count = 0
        for i in range(0, oritensor_szie[0] - subtensor_size[0] + 1, stride):
            for j in range(0, oritensor_szie[1] - subtensor_size[1] + 1, stride):
                for k in range(0, oritensor_szie[2] - subtensor_size[2] + 1, stride):
                    subtensor = original_tensor[i:i + subtensor_size[0], j:j + subtensor_size[1],
                                k:k + subtensor_size[2]]
                    result_tensor[count] = subtensor
                    count += 1
        return result_tensor.unsqueeze(0)


    def forward(self, patches):
        x1 = patches.cuda()
        x2 = x1.squeeze(0)
        x2 = x2.contiguous().view(self.patches, -1)  
        x2 = x2.unsqueeze(0) 

        x3, att_map = self.forward_features(x2)
        x4 = x3.view(self.num_head1, self.num_head2, self.num_head3, self.N, self.N, self.N, self.N, self.N, self.N )
        x4 = x4.permute(0, 3, 6, 1, 4, 7, 2, 5, 8)
        x4 = x4.reshape(1, self.num_head1*self.N *self.N, self.num_head2*self.N *self.N, self.num_head3*self.N *self.N)


        x4 = self.drop(x4)

        x5 = self.decoder2(x4)
        return torch.tanh(x5), att_map










class TNN(nn.Module):
    def __init__(self):
        super(TNN, self).__init__()

        one = torch.tensor(1.0, requires_grad=False)
        self.input = one.cuda()

        self.f1 = 5
        self.f2 = 10
        self.linear = nn.Linear(1, self.f1*self.f1*self.f1, bias=None)
 
        self.decoder1 = TCL(input_shape=[self.f1, self.f1, self.f1], rank=[self.f2, self.f2, self.f2])
        self.decoder2 = TCL(input_shape=[self.f2, self.f2, self.f2], rank=[20, 20, 20])

    def forward(self):
        initial_core = self.linear(self.input.unsqueeze(0))
        #initial_core = self.initial_core
        x = self.decoder1(initial_core.view(1,self.f1,self.f1,self.f1))
        x = torch.relu(x)
        x = self.decoder2(x)
        return torch.tanh(x), initial_core









def total_variation(images): #torch.Size([1, 20, 20, 20])
    sum_axis = None
    pixel_dif1 = images[0,1:, :, :] - images[0,:-1, :, :]
    pixel_dif2 = images[0,:, 1:, :] - images[0,:, :-1, :]
    pixel_dif3 = images[0,:, :, 1:] - images[0,:, :, :-1]
    tot_var = (
        torch.sum(torch.abs(pixel_dif1), axis=sum_axis) +
        torch.sum(torch.abs(pixel_dif2), axis=sum_axis) +
        torch.sum(torch.abs(pixel_dif3), axis=sum_axis) )
    return tot_var








def loss_fn_mse(outputs, observation_truth, observation_tensor,  add_TV_regu=False):
    pred=(observation_tensor*outputs)
    observation_truth=observation_truth
    num_obser=torch.sum(observation_tensor)
    if add_TV_regu:
        alpha = 1e-8
        return torch.sum((pred-observation_truth)**2)/num_obser + alpha*total_variation(outputs)
    else:
        return torch.sum((pred-observation_truth)**2)/num_obser







if __name__ == '__main__':
    pass

