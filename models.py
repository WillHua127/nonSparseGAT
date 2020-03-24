import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha,
                                                 adj=adj,
                                                 attention=False,
                                                last=False) for _ in range(nheads)]
        self.attentions[int(nheads/2)] = GraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha,
                                                 adj=adj,
                                                 attention=True,
                                                last=False)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha,
                                             adj=adj,
                                             attention=True,
                                            last=True)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x)
        return F.log_softmax(x, dim=1)

