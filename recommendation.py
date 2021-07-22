import re
import os
import sys
import time
import math
import datetime
import random
import numpy as np
import pandas as pd
import scipy.sparse
import multiprocessing as mp

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import torch
from pygcn.utils import sparse_mx_to_torch_sparse_tensor

from user_network import UserNetwork
from utils import select_nodes_table, pytorch_cos_sim
from utils import get_media_name_via_NewsIndex, get_media_name_via_NewsURL, clean_media_NewsURL

import sshtunnel
from sshtunnel import SSHTunnelForwarder
from mysql.connector import pooling


ssh_host = '140.114.55.4'
ssh_username = 'user'
ssh_password = 'Nthuieem@42198'
database_username = 'NRS'
database_password = '70933980'
database_name = 'NRS'
localhost = '127.0.0.1'


def open_ssh_tunnel(verbose=False):
    """Open an SSH tunnel and connect using a username and password.
    :param verbose: Set to True to show logging
    :return tunnel: Global SSH tunnel connection
    """
    if verbose:
        sshtunnel.DEFAULT_LOGLEVEL = logging.DEBUG
    global tunnel
    tunnel = SSHTunnelForwarder(
        (ssh_host, 22),
        ssh_username = ssh_username,
        ssh_password = ssh_password,
        remote_bind_address = ('140.114.55.4', 3306)
    )
    tunnel.start()

def close_ssh_tunnel():
    """Closes the SSH tunnel connection."""
    tunnel.close





def check_user_history_url(url):
    media_name = get_media_name_via_NewsURL(url)
    if media_name==None: # Check url belong to news media
        return None
    url = clean_media_NewsURL(media_name, url)
    if url==None: # Check url correctness of each news media
        return None
    x = select_nodes_table(DB_PATH, media_name+'news', 'NewsIndex', "NewsURL LIKE '%{}%'".format(url))
    if x==[]: # Check is parsed or not
#         print('News not found in sql: %s' %url)
        return None
    NewsIndex = x[0][0]
    if NewsIndex not in NewsIndex_list: # Check in cross-news-network
#         print('News not found in cross-news-network: %s' %url)
        return None
    return NewsIndex

def select_user_history_from_lab_sql(TODAY_DATE, USER_NAME, EXP_DAYS):
    QUERY_DATE = (datetime.datetime.strptime(TODAY_DATE, "%Y-%m-%d") - 
                  datetime.timedelta(days=EXP_DAYS)).strftime('%Y-%m-%d')
    sql = "SELECT NewsURL FROM UserRead WHERE User='%s' AND TodayDate>='%s'" %(USER_NAME, QUERY_DATE)
    con = dbpool.get_connection()
    cur = con.cursor(buffered=True)
    cur.execute(sql)
    user_history_url = cur.fetchall()
    cur.close()
    con.close()
    
    user_history = []
    for k in user_history_url:
        url = k[0]
        x = check_user_history_url(url)
        if x!=None:
            user_history.append(x)
    return user_history

def insert_recommendation_to_lab_sql(USER_NAME, TODAY_DATE, final_recommendation):
    con = dbpool.get_connection()
    cur = con.cursor(buffered=True)
    
    user = USER_NAME
    today_date = TODAY_DATE
    for i, k in enumerate(final_recommendation):
        NewsIndex = k[0]
        NewsDate = k[1]
        NewsTitle = k[2]
        NewsContext = k[3]
        
        check_exist_sql = """SELECT * FROM UserRecommendation WHERE USER='%s'AND TodayDate='%s' AND NewsIndex='%s'
                            AND NewsDate='%s' AND NewsTitle='%s'""" %(user, today_date, NewsIndex, NewsDate, NewsTitle) 
        cur.execute(check_exist_sql)
        check_exist=False if cur.fetchall()==[] else True
        if check_exist==False:        
            sql = '''INSERT INTO UserRecommendation (USER,TodayDate,NewsIndex,NewsDate,NewsTitle,NewsContext) 
                VALUES ('%s','%s','%s','%s','%s','%s')''' %(user, today_date, NewsIndex, NewsDate, NewsTitle, NewsContext)
            cur.execute(sql)
        
    con.commit()
    cur.close()
    con.close()






def generate_single_news_recommendation(NewsIndex, gcn_similarity_matrix, TOP_N):
    """
    Requires:
        idx_NewsIndex_map, idx_list, NewsIndex_list, gcn_similarity_matrix
    """
    recommendation_list = []
    NewsIndex_idx = idx_NewsIndex_map[NewsIndex]
    recommend_idx_list = gcn_similarity_matrix[NewsIndex_idx].argsort()[::-1][:TOP_N*2]
    
    for idx in recommend_idx_list:
        recommend_NewsIndex = NewsIndex_list[idx_list.index(idx)]
        if recommend_NewsIndex==NewsIndex: 
            continue
        media_name = get_media_name_via_NewsIndex(recommend_NewsIndex)
        x = select_nodes_table(DB_PATH, media_name+'news', '*', "NewsIndex=='%s'" %(recommend_NewsIndex))
        if x==[]: 
            continue
        x = x[0]
        score = gcn_similarity_matrix[NewsIndex_idx][idx]
        # [NewsIndex, NewsDate, NewsTitle, NewsContext, sim_score]
        recommendation_list.append([x[0], x[3], x[4], x[5], float(score)])
        
#         recommendation_list = sorted(recommendation_list, key=lambda x:x[2], reverse=True) # sort by NewsDate
#         recommendation_list = [k for k in recommendation_list if k[2]==TODAY_DATE]
    return recommendation_list[:TOP_N]

def generate_cluster_recommendation(cluster, REC_LEN):
    recommendation_list = []
    for NewsIndex in cluster:
        recommendation_list += generate_single_news_recommendation(NewsIndex, gcn_similarity_matrix, TOP_N)
 
    cluster_recommendation = []
    for k in recommendation_list:
        if k[1] not in [n[1] for n in cluster_recommendation]:
            cluster_recommendation.append(k)
            
    cluster_recommendation = sorted(cluster_recommendation, key=lambda x:x[2], reverse=True) # sort by NewsDate
    cluster_recommendation = cluster_recommendation[:REC_LEN]
    
#     for r in cluster_recommendation:
#         print('%s %s %s %s' %(r[0], r[1], r[2], r[-1]))
    return cluster_recommendation

def generate_user_recommendation(user_history, user_interests_cluster, TOTAL_REC_LEN):
    """
    TOTAL_REC_LEN: Expected final recommendation list length
    SCALE: Impotance of an interests cluster (i.e. the propotion within all news)
    cluster_rec_len: # of rec news for interests cluster i = len(news in cluster i)/len(total user history news)
    """
    SCALE = len(user_history) / TOTAL_REC_LEN 
    total_len = 0
    user_recommendation = []
    for i, cluster in enumerate(user_interests_cluster):
        CLUSTER_REC_LEN = math.ceil(len(cluster) / SCALE)
        user_recommendation += generate_cluster_recommendation(cluster, CLUSTER_REC_LEN)
#         print('\n----------cluster %d-----------\n' %(i+1))
#         print('Original cluster len %d, recommend %d news' %(len(cluster), CLUSTER_REC_LEN))
        total_len += CLUSTER_REC_LEN
        if total_len >= TOTAL_REC_LEN:
            break

    user_recommendation = user_recommendation[:TOTAL_REC_LEN]
    return user_recommendation






def main(DB_PATH, TODAY_DATE, USER_NAME, EXP_DAYS):
    user_history = select_user_history_from_lab_sql(TODAY_DATE, USER_NAME, EXP_DAYS)
    if len(user_history)==0:
        print('[Warning] %s have no history news' %USER_NAME)
        return
    else:
        print('%s read list length %d' %(USER_NAME, len(user_history)))
    
    user_history_NewsIndex = []
    user_history_NewsEmb = []
    for NewsIndex in user_history:
        idx = idx_NewsIndex_map[NewsIndex]
        user_history_NewsIndex.append(NewsIndex)
        user_history_NewsEmb.append(NewsEmb_list[idx].tolist())
        
    UN = UserNetwork(DB_PATH, TODAY_DATE)
    user_interests_network = UN.construct_user_interests_network(user_history_NewsIndex, 
                                                                 user_history_NewsEmb, 
                                                                 THRES)
    user_interests_cluster = UN.generate_user_interests_network_cluster(user_interests_network)
    UN.print_user_interests_network_cluster(user_interests_cluster)
    user_recommendation = generate_user_recommendation(user_history, user_interests_cluster, TOTAL_REC_LEN)
    
    final_recommendation = user_recommendation
    print('%s recommendation list length %d' %(USER_NAME, len(final_recommendation)) )
#     insert_recommendation_to_lab_sql(USER_NAME, TODAY_DATE, final_recommendation)
#     return final_recommendation

    
    
if __name__ == '__main__':
    DB_PATH = './db/NewsNetwork_ch_20210619.db'
    TODAY_DATE = '2021-06-19'
    USERS = []          # Experiment user list
    EXP_DAYS = 1        # 1d / 3d / 5d
    THRES = 0.75        # User interests network edges thres
    TOTAL_REC_LEN = 50  # Total rec len
    TOP_N = 5           # Single news rec len
    
    USERS = ['李孟熹', '楊怡芳', '林勁甫', '胡筱郁', '許育珠', '鄭建澤', '徐家琇', '王麗喬', '方宥鈞', '葉人維', '邵建喜']
    
    
    open_ssh_tunnel(verbose=False)
    dbpool = pooling.MySQLConnectionPool(
        pool_size=5,
        pool_reset_session=True,
        host='127.0.0.1',
        database=database_name,
        user=database_username,
        password=database_password,
        port=tunnel.local_bind_port
    )
    
    s = time.time()
    
    # Load GCN emb
    gcn_emb_save_path = './gcn-emb/%s.npz' %(TODAY_DATE)
    gcn_emb = np.load(gcn_emb_save_path)['gcn']
    # print(gcn_emb.shape)
    def normalization(matrix):
        minima, maxima = matrix.min(), matrix.max()
        return (matrix - minima)/(maxima - minima)
    gcn_similarity_matrix = 1-pairwise_distances(gcn_emb, metric="cosine")
    gcn_similarity_matrix = normalization(gcn_similarity_matrix)
    
    # Load cross-news-network
    cross_news_network_path = './cross-news-network/%s/' %(TODAY_DATE)
    cross_news_network_nodes = pd.read_csv("%s/cross_news_network_nodes.csv" 
                                           %(cross_news_network_path), header=None).to_numpy()
    NewsEmb_list = cross_news_network_nodes[:, 1:]
    NewsIndex_list = cross_news_network_nodes[:, 0].tolist()
    idx_NewsIndex_map = {j: i for i, j in enumerate(NewsIndex_list)}
    idx_list = list(idx_NewsIndex_map.values())

    print('Pre Process time %d sec' %(time.time() - s) )
    
    
    
    def users_2_batch(before, batch_size):
        c, batch, after = 0, [], []
        for i, k in enumerate(before):
            c += 1
            batch.append(k)
            if (i+1) == len(before):
                after.append(batch)
                return after, len(after)
            if c == batch_size:
                after.append(batch)
                c, batch = 0, []
        return after, len(after)

    USERS_BATCH, num_of_batch = users_2_batch(USERS, batch_size=3)
    print(USERS_BATCH, num_of_batch)


    for batch_idx in range(num_of_batch):
        print('\n---------- Batch %d ----------\n' %(batch_idx+1) )
        s_batch_time = time.time()
        USERS_BATCH_NAMES = USERS_BATCH[batch_idx]

        mp_list = []
        for USER_NAME in USERS_BATCH_NAMES: 
            p = mp.Process(target=main, args=(DB_PATH, TODAY_DATE, USER_NAME, EXP_DAYS, ))
            mp_list.append(p)
        for p in mp_list:
            p.start()
        for p in mp_list:
            p.join()
        print('Batch process time %d sec' %(time.time() - s_batch_time) )

    print('Total process time %d sec' %(time.time() - s) )
    
    
    
    
    close_ssh_tunnel()
    
    
    
    
    
    
    
    
