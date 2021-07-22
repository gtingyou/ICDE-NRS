import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import scipy.sparse
import argparse

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import sparse_mx_to_torch_sparse_tensor
from pygcn.layers import GraphConvolution
# from pygcn.models import GCN


class GCN_1layer(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN_1layer, self).__init__()
    
        self.gc1 = GraphConvolution(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


# Load data
def load_news_data(path, EXP_W_NAME):
    print('[INFO] Loading dataset from %s' %(path))
    
    nodes_files = pd.read_csv("%s/cross_news_network_nodes.csv" %(path), header=None).to_numpy()
    features = nodes_files[:, 1:]
    idx = nodes_files[:, 0]
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_files = pd.read_csv("%s/cross_news_network_edges-%s.csv" %(path, EXP_W_NAME), header=None).to_numpy()
    edges_head_tail = edges_files[:, 0:2]
    edges_weight = edges_files[:, -1]
    edges = np.array(list(map(idx_map.get, edges_head_tail.flatten())), 
                     dtype=np.int32).reshape(edges_head_tail.shape)
    
    adj = scipy.sparse.coo_matrix((edges_weight, (edges[:, 0], edges[:, 1])),
                                  shape=(idx.shape[0], idx.shape[0]),
                                  dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features, dtype=np.float32))
    X = torch.LongTensor(edges)
    Y = torch.FloatTensor(np.array(edges_weight, dtype=np.float32))
    
    return idx_map, adj, features, X, Y



def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    
    x1 = X[:, 0]
    x2 = X[:, 1]
    criterion = torch.nn.CosineSimilarity()
    y = criterion(output[x1], output[x2])
    mse_loss = torch.nn.MSELoss()
    loss_train = mse_loss(y, Y)
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


# def generate_gcn_embeddings(model, adj, features):
#     model.eval()
#     output = model(features, adj)
#     return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--today_date", type=str, required=True)
    parser.add_argument("--time_version", type=int, required=True, 
                        help='Time which the version of network is generate within a day')
    args = parser.parse_args()
    
    TODAY_DATE = args.today_date
    EXP_W = 0.1
    EXP_W_NAME = str(EXP_W).replace('.','')
    TIME_VERSION = str(args.time_version)
    
    torch.cuda.empty_cache()
    
    
    no_cuda = False
    fastmode = False
    seed = 42
    epochs = 60
    lr = 0.001
    weight_decay = 5e-4
    dropout = 0.1
    
    # Load data
    cross_news_network_path = "./cross-news-network/%s/%s" %(TODAY_DATE, TIME_VERSION)
    idx_map, adj, features, X, Y = load_news_data(cross_news_network_path, EXP_W_NAME)
    
    # Init model
    model = GCN_1layer(nfeat=features.shape[1],
                       nclass=768,
                       dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('\n\n', model, '\n\n')
    
    # Cuda
    if not no_cuda and torch.cuda.is_available():
        print('[INFO] Use Cuda!')
        model.cuda()
        features, adj, X, Y = features.cuda(), adj.cuda(), X.cuda(), Y.cuda()
    
    # Train gcn model
    start = time.time()
    for epoch in range(epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start))
 
    
    # Save gcn model
    gcn_model_output_path = "./gcn-model/%s/%s" %(TODAY_DATE, TIME_VERSION)
    if not os.path.exists(gcn_model_output_path):
        os.makedirs(gcn_model_output_path)
    
    print("[INFO] Save GCN model: %s" %(gcn_model_output_path))
    torch.save(model, os.path.join(gcn_model_output_path, 'gcn_model.pt'))
    
    
    
    
    
#     gcn_emb = generate_gcn_embeddings(model, adj, features)
#     print("[INFO] GCN embedding shape", gcn_emb.shape)
#     print("[INFO] Save GCN embedding: %s" %(gcn_emb_output_path))
#     np.savez(gcn_emb_output_path + 'gcn_emb.npz', gcn=gcn_emb.detach().cpu().numpy())
    
    
    
    
    
    



