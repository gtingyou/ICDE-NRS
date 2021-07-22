import os
import re
import time
import datetime
import torch
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import argparse
import multiprocessing as mp
import GPUtil

from sentence_transformers import SentenceTransformer, util
from single_news_network import SingleNewsNetwork
from run_isorank import IsoRank




def sbert_embedding(MEDIA_nodes, sbert_model):
    MEDIA_doc = [k[-1] for k in MEDIA_nodes]
    MEDIA_emb = sbert_model.encode(MEDIA_doc, convert_to_tensor=True)
    return MEDIA_emb

def construct_sparse_document_similarity_matrix(ALL_emb, THRES):
    def filter_document_similarity_score(document_cs_matrix):
        zero = torch.zeros_like(document_cs_matrix)
        THRES_document_cs_matrix = torch.where(document_cs_matrix<THRES, zero, document_cs_matrix)
        return THRES_document_cs_matrix
    
    document_cs_matrix = util.pytorch_cos_sim(ALL_emb, ALL_emb)
    THRES_document_cs_matrix = filter_document_similarity_score(document_cs_matrix)
    sparse_document_matrix = scipy.sparse.csc_matrix(THRES_document_cs_matrix.cpu().detach().numpy()) # torch tensor to numpy array
    return sparse_document_matrix





def construct_inter_edges(ALPHA):
    def normalize_IsoRank_score(isorank_score):
        minima = isorank_score.min()
        maxima = isorank_score.max()
        normalized_isorank_score = (isorank_score - minima)/(maxima - minima)
        return normalized_isorank_score
    
    
    ### Step 1. Initialize IsoRank
    EXP_W_NAME = str(ALPHA).replace('.','')
    isorank = IsoRank('%s-%s' %(TODAY_DATE, EXP_W_NAME), ALL_nodes, sparse_document_matrix)

    ### Step 2. Create IsoRank input files
    for i in range(len(MEDIAS)):
        for j in range(len(MEDIAS)):
            if i<j:
                print(MEDIAS[i], MEDIAS[j])
                isorank.create_evals_files(MEDIAS[i], MEDIAS[j])
    isorank.create_tab_files('CHT', CHT_edges)
    isorank.create_tab_files('CNA', CNA_edges)
    isorank.create_tab_files('LBT', LBT_edges)
    isorank.create_tab_files('SETN', SETN_edges)
    isorank.create_tab_files('TVBS', TVBS_edges)
    isorank.create_tab_files('UDN', UDN_edges)
    isorank.create_inp_file(MEDIAS)
    isorank.create_sh_file(ALPHA)

    ### Step 3. Execute IsoRank.sh
    start = time.time()
    isorank.run_isorank_exe()
    print('[INFO] Running IsoRank time %d sec' %(time.time() - start) )
    
    ### Step 4. Process tmp_match-score.txt
    isorank_output_path = isorank.output_path
    df = pd.read_csv(isorank_output_path+'/tmp_match-score.txt', header=None, sep=' ')
    normalized_df = pd.concat([df.iloc[:, 0:2], normalize_IsoRank_score(df[2])], axis=1) # Due to low score, apply normalization
    normalized_df = normalized_df[normalized_df[2] > 0.1] # After normalization, delete score smaller than 0.1
    normalized_isorank_score = normalized_df.values.tolist()
    inter_edges = sorted(normalized_isorank_score)
    return inter_edges





def construct_single_media_intra_edges(MEDIA_nodes, MEDIA_edges, MEDIA_emb, BETA):
    def calculate_single_network_common_neighbors(MEDIA_nodes, MEDIA_edges):
        """Computes Common neighbors similarity matrix for an adjacency matrix"""
        sparse_cn_matrix = scipy.sparse.lil_matrix((len(MEDIA_nodes), len(MEDIA_nodes)))
        sparse_min_degree_matrix = scipy.sparse.lil_matrix((len(MEDIA_nodes), len(MEDIA_nodes)))
        MEDIA_nodes_NewsIndex = [k[0] for k in MEDIA_nodes]
        
        G = nx.Graph()
        G.add_edges_from(MEDIA_edges)
        adj = nx.adjacency_matrix(G)
        A = np.squeeze(np.asarray(adj.todense()))
        vertices = list(G.nodes)
        
        for line in A:
            AdjList = line.nonzero()[0] # column indices with nonzero values
            k_deg = len(AdjList)
            d = np.log(1.0/k_deg) # row i's AA score
            #add i's score to the neighbor's entry
            for i in range(len(AdjList)):
                for j in range(len(AdjList)):
                    if i>j:
                        NewsIndex1 = vertices[AdjList[i]]
                        NewsIndex2 = vertices[AdjList[j]]
                        id1 = MEDIA_nodes_NewsIndex.index(NewsIndex1)
                        id2 = MEDIA_nodes_NewsIndex.index(NewsIndex2)
                        sparse_cn_matrix[(id1, id2)] = sparse_cn_matrix[(id1, id2)] + 1
                        sparse_cn_matrix[(id2, id1)] = sparse_cn_matrix[(id2, id1)] + 1
                        if sparse_min_degree_matrix[(id1, id2)]==0 and sparse_min_degree_matrix[(id2, id1)]==0:
                            min_degree = min(G.degree(NewsIndex1), G.degree(NewsIndex2))
                            sparse_min_degree_matrix[(id1, id2)] = min_degree
                            sparse_min_degree_matrix[(id2, id1)] = min_degree
        sparse_common_neighbor_matrix = np.nan_to_num(sparse_cn_matrix / sparse_min_degree_matrix)
        # filter intra (common neighbor) edges, if intra score < 0.1, then replace with 0
        sparse_common_neighbor_matrix = np.where(sparse_common_neighbor_matrix>0.1, sparse_common_neighbor_matrix, 0)
        sparse_common_neighbor_matrix = scipy.sparse.lil_matrix(sparse_common_neighbor_matrix)
        return sparse_common_neighbor_matrix
    
    def merge_intra_topology_document_similarity_matrix(sparse_intra_topology_matrix, sparse_intra_document_matrix, BETA):
        topology_matrix = sparse_intra_topology_matrix.todense()
        document_matrix = sparse_intra_document_matrix.todense()
        total_matrix = BETA*topology_matrix + (1-BETA)*document_matrix
        # if intra score < BETA, then replace with 0 (i.e. Keep the score with both document/topology score)
        total_matrix = np.where(total_matrix>BETA, total_matrix, 0)
        return total_matrix
    
    def sparse_matrix_to_edges(MEDIA_nodes, sparse_intra_matrix):
        assert len(MEDIA_nodes) == len(sparse_intra_matrix)
        MEDIA_nodes_NewsIndex = [k[0] for k in MEDIA_nodes]
        edges = []
        for i in range(len(sparse_intra_matrix)):
            for j in range(len(sparse_intra_matrix)):
                if i>j:
                    s = sparse_intra_matrix[i][j]
                    if s!=0:
                        edges.append([MEDIA_nodes_NewsIndex[i], MEDIA_nodes_NewsIndex[j], s])
                elif i==j:
                    s = sparse_intra_matrix[i][j]
                    edges.append([MEDIA_nodes_NewsIndex[i], MEDIA_nodes_NewsIndex[j], s])
        return edges
    
    start_time = time.time()
    sparse_intra_topology_matrix = calculate_single_network_common_neighbors(MEDIA_nodes, MEDIA_edges)
    sparse_intra_document_matrix = construct_sparse_document_similarity_matrix(MEDIA_emb, DOC_THRES)
    torch.cuda.empty_cache() # clear gpu memory
    sparse_intra_matrix = merge_intra_topology_document_similarity_matrix(sparse_intra_topology_matrix, 
                                                                          sparse_intra_document_matrix, 
                                                                          BETA)
    intra_edges = sparse_matrix_to_edges(MEDIA_nodes, sparse_intra_matrix)
    end_time = time.time()
    print('[INFO] Single news Intra edges process time %d sec' %(end_time - start_time))
    return intra_edges

def construct_intra_edges(BETA):
    def normalize_intra_edges(intra_edges):
        df = pd.DataFrame(intra_edges)
        intra_edges_score = df[2]
        minima = intra_edges_score.min()
        maxima = intra_edges_score.max()
        normalized_intra_edges_score = (intra_edges_score - minima)/(maxima - minima)
        normalized_df = pd.concat([df.iloc[:, 0:2], normalized_intra_edges_score], axis=1)
        return normalized_df.values.tolist()
    
    CHT_intra_edges = construct_single_media_intra_edges(CHT_nodes, CHT_edges, CHT_emb, BETA)
    CNA_intra_edges = construct_single_media_intra_edges(CNA_nodes, CNA_edges, CNA_emb, BETA)
    LBT_intra_edges = construct_single_media_intra_edges(LBT_nodes, LBT_edges, LBT_emb, BETA)
    SETN_intra_edges = construct_single_media_intra_edges(SETN_nodes, SETN_edges, SETN_emb, BETA)
    TVBS_intra_edges = construct_single_media_intra_edges(TVBS_nodes, TVBS_edges, TVBS_emb, BETA)
    UDN_intra_edges = construct_single_media_intra_edges(UDN_nodes, UDN_edges, UDN_emb, BETA)
    
    intra_edges = CHT_intra_edges+CNA_intra_edges+LBT_intra_edges+SETN_intra_edges+TVBS_intra_edges+UDN_intra_edges
    intra_edges = normalize_intra_edges(intra_edges)
    return intra_edges





def construct_cross_news_network_nodes(ALL_nodes, ALL_emb):
    assert len(ALL_nodes) == ALL_emb.shape[0]
    cross_news_network_nodes = []
    ALL_nodes_NewsIndex = [k[0] for k in ALL_nodes]
    ALL_nodes_features = ALL_emb.cpu().detach().numpy()
    for i in range(len(ALL_nodes)):
        cross_news_network_nodes.append([ALL_nodes_NewsIndex[i]] + ALL_nodes_features[i].tolist())
    return cross_news_network_nodes

def construct_cross_news_network_edges(inter_edges, intra_edges):
    return sorted(inter_edges + intra_edges)

def output_cross_news_network_nodes(cross_news_network_nodes, cross_news_network_output_path):
    print('[INFO] Cross news network nodes %d' %len(cross_news_network_nodes))
    
    pd.DataFrame(cross_news_network_nodes).to_csv(cross_news_network_output_path + '/cross_news_network_nodes.csv', 
                                                  encoding='utf_8_sig', 
                                                  index=None,
                                                  header=None)
    
def output_cross_news_network_edges(cross_news_network_edges, cross_news_network_output_path, EXP_W_NAME):
    print('[INFO] Cross news network edges %d' %len(cross_news_network_edges))
    
    pd.DataFrame(cross_news_network_edges).to_csv(cross_news_network_output_path + '/cross_news_network_edges-%s.csv' %(EXP_W_NAME), 
                                                  encoding='utf_8_sig', 
                                                  index=None, 
                                                  header=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--today_date", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--num_days", type=int, default=7)
    parser.add_argument("--doc_thres", type=float, default=0.9)
#     parser.add_argument("--alpha", type=float, default=0.1)
#     parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--time_version", type=int, required=True, 
                        help='Time which the version of network is generate within a day')
    args = parser.parse_args()
    
    print(args)
    
    # Params
    DB_PATH = args.db_path
    TODAY_DATE = args.today_date # datetime.datetime.now().strftime("%Y-%m-%d")
    NUM_DAYS = args.num_days
    DOC_THRES = args.doc_thres
    MEDIAS = sorted(['CHT', 'CNA', 'LBT', 'SETN', 'TVBS', 'UDN'])
    EXP_WEIGHTS = [0.1] # [0.01, 0.1, 0.5]
#     ALPHA = args.alpha
#     BETA = args.beta
    TIME_VERSION = str(args.time_version)
    
    
    cross_news_network_output_path = os.path.join(os.getcwd(), 'cross-news-network', TODAY_DATE, TIME_VERSION)
    if not os.path.exists(cross_news_network_output_path):
        os.makedirs(cross_news_network_output_path)
    
    
    
    
    
    total_start_t = time.time()
    
    # Initialize SBERT model
    sbert_model_save_path = '/work/u4839782/SBERT-models/'
    sbert_model_name = 'training_news_multiGPU_bert-base-chinese-2021-03-08_15-53-17'
    sbert_model = SentenceTransformer(os.path.join(sbert_model_save_path, sbert_model_name))

    # Read single news network from 6 medias
    CHT_nodes, CHT_edges = SingleNewsNetwork('CHT', DB_PATH, TODAY_DATE, NUM_DAYS).construct_news_network()
    LBT_nodes, LBT_edges = SingleNewsNetwork('LBT', DB_PATH, TODAY_DATE, NUM_DAYS).construct_news_network()
    UDN_nodes, UDN_edges = SingleNewsNetwork('UDN', DB_PATH, TODAY_DATE, NUM_DAYS).construct_news_network()
    CNA_nodes, CNA_edges = SingleNewsNetwork('CNA', DB_PATH, TODAY_DATE, NUM_DAYS).construct_news_network()
    SETN_nodes, SETN_edges = SingleNewsNetwork('SETN', DB_PATH, TODAY_DATE, NUM_DAYS).construct_news_network()
    TVBS_nodes, TVBS_edges = SingleNewsNetwork('TVBS', DB_PATH, TODAY_DATE, NUM_DAYS).construct_news_network()

    # Inference SBERT embedding
    CHT_emb = sbert_embedding(CHT_nodes, sbert_model)
    CNA_emb = sbert_embedding(CNA_nodes, sbert_model)
    LBT_emb = sbert_embedding(LBT_nodes, sbert_model)
    SETN_emb = sbert_embedding(SETN_nodes, sbert_model)
    TVBS_emb = sbert_embedding(TVBS_nodes, sbert_model)
    UDN_emb = sbert_embedding(UDN_nodes, sbert_model)

    ALL_nodes = CHT_nodes + CNA_nodes + LBT_nodes + SETN_nodes + TVBS_nodes + UDN_nodes
    ALL_edges = CHT_edges + CNA_edges + LBT_edges + SETN_edges + TVBS_edges + UDN_edges
    ALL_emb = torch.cat((CHT_emb, CNA_emb, LBT_emb, SETN_emb, TVBS_emb, UDN_emb), 0) # axis = 0

    # Document similarity matrix via SBERT embedding
    sparse_document_matrix = construct_sparse_document_similarity_matrix(ALL_emb, DOC_THRES)
    torch.cuda.empty_cache() # clear gpu memory
    
    # Cross news network nodes
    cross_news_network_nodes = construct_cross_news_network_nodes(ALL_nodes, ALL_emb)
    output_cross_news_network_nodes(cross_news_network_nodes, cross_news_network_output_path)
    
    # Cross news network edges with different weight
    for i, EXP_W in enumerate(EXP_WEIGHTS):
        ALPHA = BETA = EXP_W
        EXP_W_NAME = str(EXP_W).replace('.','')
        print('\n\n------------------------------NETWORK_NAME %s-------------------------------\n\n' %EXP_W_NAME)
        # Inter edges
        inter_edges = construct_inter_edges(ALPHA)
        # Intra edges
        intra_edges = construct_intra_edges(BETA)
        # Cross news network edges
        cross_news_network_edges = construct_cross_news_network_edges(inter_edges, intra_edges)
        output_cross_news_network_edges(cross_news_network_edges, cross_news_network_output_path, EXP_W_NAME)
        
    
    # Multi process
#     def mp_worker(w):
#         ALPHA = BETA = w
#         NETWORK_TYPE = str(w).replace('.','')
#         print('\n\n------------------------------NETWORK_NAME %s-------------------------------\n\n' %NETWORK_TYPE)
#         # Inter edges
#         inter_edges = construct_inter_edges(ALPHA)
#         # Intra edges
#         intra_edges = construct_intra_edges(BETA)
#         # Cross news network edges
#         cross_news_network_edges = construct_cross_news_network_edges(inter_edges, intra_edges)
#         output_cross_news_network_edges(cross_news_network_edges, cross_news_network_output_path, NETWORK_TYPE)
    
#     mp_list = []
#     for w in EXP_WEIGHTS: 
#         p = mp.Process(target=mp_worker, args=(w, ))
#         mp_list.append(p)
#     for p in mp_list:
#         p.start()
#     for p in mp_list:
#         p.join()
    
    print('[INFO] Cross news network construction total process time %d sec' %(time.time() - total_start_t) )
    
    
    
    