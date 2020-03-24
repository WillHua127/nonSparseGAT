import numpy as np
import random
import json
import sys
import os
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import math
import torch 

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_dataset(dataset, dense=False):
    dense_adj, features, labels = torch.load("%s_dense_adj.pt" % dataset), torch.load("%s_features.pt" % dataset), torch.load("%s_labels.pt" % dataset)

    y=140
    s_val = labels[np.random.choice(labels.shape[0], 500, replace=False)]
    if dataset == 'citeseer':
        y=120
    elif dataset == 'pubmed':
        y=60
    elif dataset == 'nell.0.1':
        y=6600
    elif dataset == 'nell.0.01':
        y=660
    elif dataset == 'nell.0.001':
        y=66
    idx_train = range(y)
    idx_val = range(y, len(s_val) + 500)
    idx_test = range(len(s_val) + 500, len(s_val) + 1500)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    if dense:
        return dense_adj, features, labels, idx_train, idx_val, idx_test
    else:
        indices = torch.nonzero(dense_adj).t(); values = dense_adj[indices[0], indices[1]]
        adj = torch.sparse.FloatTensor(indices, values, dense_adj.size()).clone()
        del dense_adj, indices, values
        return adj, features, labels, idx_train, idx_val, idx_test
    
def random_walk_sampling_with_fly_back(complete_graph, fly_back_prob):
    complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
    # giving unique id to every node same as built-in function id
    for n, data in complete_graph.nodes(data=True):
        complete_graph.node[n]['id'] = n

    nr_nodes = len(complete_graph.nodes())
    upper_bound_nr_nodes_to_sample = nr_nodes

    index_of_first_random_node = random.randint(0, nr_nodes-1)
    sampled_graph = nx.Graph()

    sampled_graph.add_node(complete_graph.node[index_of_first_random_node]['id'])

    iteration = 1
    edges_before_t_iter = 0
    curr_node = index_of_first_random_node
    while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
        edges = [n for n in complete_graph.neighbors(curr_node)]
        index_of_edge = random.randint(0, len(edges) - 1)
        chosen_node = edges[index_of_edge]
        sampled_graph.add_node(chosen_node)
        sampled_graph.add_edge(curr_node, chosen_node)
        choice = np.random.choice(['prev','neigh'], 1, p=[fly_back_prob,1-fly_back_prob])
        if choice == 'neigh':
            curr_node = chosen_node
        iteration=iteration+1

        if iteration % 100 == 0:
            if ((sampled_graph.number_of_edges() - edges_before_t_iter) < 2):
                curr_node = random.randint(0, nr_nodes-1)
            edges_before_t_iter = sampled_graph.number_of_edges()

    return sampled_graph
    
def load_data(prefix):
    print('Loading {} dataset...'.format(prefix))
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./dataset/ind.{}.{}".format(prefix, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
        
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./dataset/ind.{}.test.index".format(prefix))   
    test_idx_range = np.sort(test_idx_reorder)

    if prefix == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        
    if prefix == 'nell.0.1':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-allx.shape[0], :] = ty
        ty = ty_extended
    
    if prefix == 'nell.0.01':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-allx.shape[0], :] = ty
        ty = ty_extended
    
    if prefix == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-allx.shape[0], :] = ty
        ty = ty_extended
        
    g = nx.DiGraph(graph)
    g = random_walk_sampling_with_fly_back(g, 0.15)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    
    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = nx.adjacency_matrix(g,nodelist=sorted(g.nodes()))
    adj = sp.csr_matrix(adj, dtype=np.float32)[:,list(np.where(np.sum(labels,1)==1)[0])][list(np.where(np.sum(labels,1)==1)[0]),:]
    adj = sp.coo_matrix(adj,dtype=np.float32)
    adj = normalize(adj + sp.eye(adj.shape[0])) 

    # build symmetric adjacency matrix
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    return adj, features, labels, 



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

