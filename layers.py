import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(4*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def relu_bt(self, x):
        threshold = torch.norm(x,p=float("inf")).clone().detach()
        return - torch.threshold(-F.leaky_relu(x),-threshold,-threshold)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        h = self.relu_bt(h)
        N = h.size()[0]
        
        agg = self.relu_bt(torch.add(h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)))
        diff = self.relu_bt(torch.sub(h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)))
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1), agg, diff], dim=1).view(N, -1, 4 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return self.relu_bt(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
