import sqlite3
import numpy as np
import pandas as pd
import scipy.sparse
import re
import torch
from torch import Tensor

from utils import select_nodes_table, pytorch_cos_sim
from utils import get_media_name_via_NewsIndex, get_media_name_via_NewsURL, clean_media_NewsURL



class UserNetwork():
    def __init__(self, DB_PATH, TODAY_DATE):
        self.DB_PATH = DB_PATH
        self.TODAY_DATE = TODAY_DATE
        
    
    def construct_user_interests_network(self, user_history_NewsIndex, user_history_NewsEmb, THRES):
        """ This function construct user interests network 
            via construct a similarity matrix using user_history sbert embedding
        Args:
            THRES: a threshold to filter similarity edges between 2 nodes
        Returns:
            user_interests_network (list): a list of edges represents the user interests network 
            [('TVBS_20210530_115', 'TVBS_20210530_115'),
             ('TVBS_20210530_115', 'TVBS_20210530_34'), 
             ...]
        """
        def construct_sparse_cs_matrix(emb, THRES):
            from sklearn.metrics import pairwise_distances
            from scipy.spatial.distance import cosine
            emb = np.array(emb)
            cs_matrix = 1 - pairwise_distances(emb, metric="cosine") # pytorch_cos_sim(emb, emb)
            THRES_cs_matrix = np.where(cs_matrix<THRES, np.zeros_like(cs_matrix), cs_matrix)
#             THRES_cs_matrix = THRES_cs_matrix.cpu().detach().numpy()
            sparse_THRES_cs_matrix = scipy.sparse.csc_matrix(THRES_cs_matrix) 
            return sparse_THRES_cs_matrix

        def sparse_matrix_to_dense_edges(matrix_index, sparse_matrix):
            assert len(matrix_index) == sparse_matrix.shape[0]
            data = sparse_matrix.data
            indptr = sparse_matrix.indptr
            indices = sparse_matrix.indices
            dense_edges = []
            for i in range(sparse_matrix.shape[0]): # 第 i 列
                d = data[indptr[i]: indptr[i+1]] # 第 i 列所有非零元素
                col = indices[indptr[i]:indptr[i+1]] # 第 i 列所有非零元素的 行index
                for idx, j in enumerate(col): # 第 i 列，第 j 行
                    if i<=j: # symmetric matrix，只取上半三角形
                        dense_edges.append( (matrix_index[i], matrix_index[j]) )
            return sorted(dense_edges)
        
#         user_history_NewsEmb = torch.tensor(user_history_NewsEmb).cuda()
        user_browsed_sparse_cs_matrix = construct_sparse_cs_matrix(user_history_NewsEmb, THRES)
        user_interests_network = sparse_matrix_to_dense_edges(user_history_NewsIndex, user_browsed_sparse_cs_matrix)
        return user_interests_network
    
    def generate_user_interests_network_cluster(self, user_interests_network):
        x = []
        for c in user_interests_network:
            if [c[0]] not in x:
                x.append([c[0]])
        for c in user_interests_network:
            if c[0]==c[1]:
                continue
            for k in x:
                if c[0] in k and c[1] not in k:
                    k.append(c[1])

        cluster = [x[0]]
        for i in range(len(x)):
            if i==0:
                continue
            for j, k in enumerate(cluster):
                new_cluster = False
                intersect = list(set(x[i]).intersection(set(k)))
                if intersect==[]:
                    new_cluster = True   
                else:
                    k = k + x[i]
                    k = list(set(k))
                    cluster[j] = k
                    break
            if new_cluster == True:
                cluster.append(x[i])
        return cluster
    
    def print_user_interests_network_cluster(self, user_interests_cluster):
        for i, cluster in enumerate(user_interests_cluster):
            print('\n----------cluster %d-----------\n' %(i+1))
            for k in cluster:
                NewsIndex = k
                media_name = get_media_name_via_NewsIndex(NewsIndex)
                x = select_nodes_table(self.DB_PATH, media_name+'news', '*', "NewsIndex=='%s'" %(NewsIndex))
                if x==[]:
                    continue
                x = x[0]
                print(x[0], x[4], x[3])
                
    def plot_user_interests_network(self, user_interests_network, save_figure=False):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()
        G.add_edges_from(user_interests_network)
        print('User interests network: # of nodes %d, # of edges %d' %(G.number_of_edges(), G.number_of_nodes()) )
        
        plt.figure(figsize=(15,15))
        pos = nx.spring_layout(G)
        nx.draw(G, with_labels=True, node_size=30, font_size=10)
        if save_figure == True:
            plt.savefig("./user_interests_graph.png", format="PNG")
        plt.show()
                