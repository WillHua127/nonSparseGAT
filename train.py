from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_dataset
from models import GAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='gcn_onepath', help='network model.') 
parser.add_argument('--public', type=int, default=1, help='split data') 
parser.add_argument('--dataset', type=str, default='citeseer', help='prefix identifying training data. cora, pubmed, citeseer.') 
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=6e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=16, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--runtimes', type=int, default=20, help='Runtime')
parser.add_argument('--identifier', type=int, default=1234567, help='Identifier for the job')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_dataset(args.dataset)

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch, model, features, labels, adj, idx_train, idx_val, optimizer):
    t = time.time()    
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features)
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val),
          'loss_test: {:.4f}'.format(loss_test.data.item()),
          'acc_test: {:.4f}'.format(acc_test),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item(), acc_test.data.item()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Train model
best_tests = []
for runtime in range(args.runtimes):
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha,
                adj=adj)
    print("MODEL_BUILT")
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
    t_total = time.time()
    test_acc = []
    val_loss = []
    bad_counter = 0
    best_test = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss, acc = train(epoch, model, features, labels, adj, idx_train, idx_val, optimizer)
        val_loss.append(loss)
        test_acc.append(acc)

        if val_loss[-1] < best:
            best = val_loss[-1]
            best_test = test_acc[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    best_tests.append(best_test)
    print("The best test accuracy this tuntime : ",best_test)
    del model, optimizer
print("The average test accuracy : ", np.mean(best_tests), "The test variance : ", np.var(best_tests), "The test standard deviation : ", np.std(best_tests))
script = open("%d.txt" % args.identifier, 'w'); script.write("%e" % np.mean(best_tests)); script.close()
