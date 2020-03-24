import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


    
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class GraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    
    def __init__(self, in_features, out_features, dropout, alpha, adj, attention=True,last=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.attention = attention
        self.last = last
        self.adj = adj
        
        self.edge = adj._indices()   

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        #self.bias = nn.Parameter(torch.zeros(size=(adj.shape[0], out_features)))
        #nn.init.xavier_normal_(self.bias.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 4*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def relu_bt(self, x):
        threshold = torch.norm(x,p=float("inf")).clone().detach()
        return - torch.threshold(-F.leaky_relu(x),-threshold,-threshold)
    
    def gam(self, x, epsilon=1e-6):
        return F.relu6(x+3)/3 + epsilon
    
    
    def forward(self, input):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        h = torch.mm(input,self.W)
        #h = torch.add(h,self.bias)
        # h: N x out
        assert not torch.isnan(h).any()

        if not(self.attention):
            # if this layer is not last layer,
            return h
        else:
            # Self-attention on the nodes - Shared attention mechanism
            #h = self.relu_bt(h)
            agg = self.relu_bt(torch.add(h[self.edge[0, :], :], h[self.edge[1, :], :]))       
            diff = self.relu_bt(torch.sub(h[self.edge[0, :], :], h[self.edge[1, :], :]))
            edge_h = torch.cat([h[self.edge[0, :], :], h[self.edge[1, :], :], agg, diff], dim=1).t()

            edge_e = torch.exp(-self.relu_bt(self.a.mm(edge_h).squeeze()))
            #edge_e = self.relu_bt(self.a.mm(edge_h).squeeze())
            assert not torch.isnan(edge_e).any()
            # edge_e: E
            
            e_rowsum = self.special_spmm(self.edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
            # e_rowsum: N x 1

            edge_e = self.dropout(edge_e)
            # edge_e: E

            h_prime = self.special_spmm(self.edge, edge_e, torch.Size([N, N]), h)
            assert not torch.isnan(h_prime).any()
            # h_prime: N x out
            
            h_prime = h_prime.div(e_rowsum+1e-16)
            # h_prime: N x out
            assert not torch.isnan(h_prime).any()
            if not(self.last):
                return h_prime
            else:
                return self.relu_bt(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

