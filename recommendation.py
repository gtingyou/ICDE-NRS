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
from gcn import load_news_data, GCN_1layer

from user_network import UserNetwork
from utils import select_nodes_table, pytorch_cos_sim
from utils import get_media_name_via_NewsIndex, get_media_name_via_NewsURL, clean_media_NewsURL

import sshtunnel
from sshtunnel import SSHTunnelForwarder
from mysql.connector import pooling

ssh_host = ''
ssh_username = ''
ssh_password = ''
database_username = ''
database_password = ''
database_name = ''
localhost = ''

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



def normalization(matrix):
    minima, maxima = matrix.min(), matrix.max()
    return (matrix - minima)/(maxima - minima)

def standadization(matrix):
    mean, mu = matrix.mean(), matrix.mu()
    return (matrix - mean)/(mu)



def generate_single_news_recommendation(NewsIndex, gcn_similarity_matrix, TOP_N):
    single_news_recommendation = []
    NewsIndex_idx = idx_NewsIndex_map[NewsIndex]
    recommend_idx_list = gcn_similarity_matrix[NewsIndex_idx].argsort()[::-1][:TOP_N*3]
    
    for idx in recommend_idx_list:
        recommend_NewsIndex = NewsIndex_list[idx_list.index(idx)]
        score = gcn_similarity_matrix[NewsIndex_idx][idx]
        if recommend_NewsIndex==NewsIndex: 
            continue
        media_name = get_media_name_via_NewsIndex(recommend_NewsIndex)
        x = select_nodes_table(DB_PATH, media_name+'news', '*', "NewsIndex=='%s'" %(recommend_NewsIndex))
        if x==[]: 
            continue
        x = x[0]
        NewsIndex, NewsDate = x[0], x[3]
        NewsTitle, NewsContext = x[4].replace("'",""), x[5].replace("'","")
        if NewsTitle not in [k[2] for k in single_news_recommendation]:
#             [NewsIndex, NewsDate, NewsTitle, NewsContext, sim_score]
            single_news_recommendation.append([NewsIndex, NewsDate, NewsTitle, NewsContext, float(score)])
        
        # sort by NewsDate
        single_news_recommendation = sorted(single_news_recommendation, key=lambda x:x[1], reverse=True)
    return single_news_recommendation[:TOP_N]

def generate_cluster_recommendation(cluster, REC_LEN):
    recommendation_list = []
    for NewsIndex in cluster:
        recommendation_list += generate_single_news_recommendation(NewsIndex, gcn_similarity_matrix, TOP_N)
        
    cluster_recommendation = []
    for k in recommendation_list:
        if k[2] not in [n[2] for n in cluster_recommendation]:
            cluster_recommendation.append(k)
            
    cluster_recommendation = sorted(cluster_recommendation, key=lambda x:x[1], reverse=True) # sort by NewsDate
#     cluster_recommendation = [k for k in cluster_recommendation if k[1]==TODAY_DATE] # filter NewsDate
    cluster_recommendation = cluster_recommendation[:REC_LEN]
    
    for r in cluster_recommendation:
        print('%s %s %s %s' %(r[0], r[1], r[2], r[4]))
    return cluster_recommendation

def generate_user_recommendation(user_history, user_interests_cluster, TOTAL_REC_LEN):
    """
    TOTAL_REC_LEN: Expected final recommendation list length
    SCALE: Impotance of an interests cluster (i.e. the propotion within all news)
    CLUSTER_REC_LEN: # of rec news for interests cluster i = len(news in cluster i)/len(total user history news)
    """
    SCALE = len(user_history) / TOTAL_REC_LEN 
    total_len = 0
    recommendation_list = []
    for i, cluster in enumerate(user_interests_cluster):
        CLUSTER_REC_LEN = math.ceil(len(cluster) / SCALE)
        print('\n----------cluster %d-----------\n' %(i+1))
        print('Original cluster len %d, recommend %d news' %(len(cluster), CLUSTER_REC_LEN))
        recommendation_list += generate_cluster_recommendation(cluster, CLUSTER_REC_LEN)
        total_len += CLUSTER_REC_LEN
        if total_len >= TOTAL_REC_LEN:
            break
    
    user_history_NewsTitle = []
    for NewsIndex in user_history:
        media_name = get_media_name_via_NewsIndex(NewsIndex)
        x = select_nodes_table(DB_PATH, media_name+'news', 'NewsTitle', "NewsIndex=='%s'" %(NewsIndex))
        user_history_NewsTitle.append(x[0][0])
    user_recommendation = []
    for k in recommendation_list:
        if k[2] not in user_history_NewsTitle:
            user_recommendation.append(k)
            
    final_recommendation = []
    for k in user_recommendation:
        if k[2] not in [n[2] for n in final_recommendation] and k[0] not in [n[0] for n in final_recommendation]:
            final_recommendation.append(k)

    final_recommendation = final_recommendation[:TOTAL_REC_LEN]
    return final_recommendation



def check_user_history_url(url):
    media_name = get_media_name_via_NewsURL(url)
    if media_name==None: # Check url belong to news media
#         print('News url not belong: %s' %url)
        return None
    url = clean_media_NewsURL(media_name, url)
    if url==None: # Check url correctness of each news media
#         print('News url not correct: %s' %url)
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

def select_user_history_from_lab_sql(TODAY_DATE, USER_NAME, EXP_D):
    QUERY_DATE = (datetime.datetime.strptime(TODAY_DATE, "%Y-%m-%d") - 
                  datetime.timedelta(days=EXP_D)).strftime('%Y-%m-%d')
#     print('%s - %s' %(QUERY_DATE, TODAY_DATE))
    sql = "SELECT NewsURL FROM UserRead WHERE User='%s' AND TodayDate<'%s' AND TodayDate>='%s'" %(USER_NAME, TODAY_DATE, QUERY_DATE)
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
    if TIME_VERSION==12:
        table_name = 'UserRecommendation1'
    else:
        table_name = 'UserRecommendation2'
    print('Insert into %s' %(table_name))
    
    con = dbpool.get_connection()
    cur = con.cursor(buffered=True)
    
    count = 0
    user = USER_NAME
    today_date = TODAY_DATE
    for i, k in enumerate(final_recommendation):
        if len(k)!=5:
            continue
        NewsIndex = k[0]
        NewsDate = k[1]
        NewsTitle = k[2]
        NewsContext = k[3].encode('utf8').decode('utf8')
        ExpDays = k[4]
        
        check_exist_sql = """SELECT * FROM %s WHERE USER='%s' AND TodayDate='%s' AND NewsIndex='%s'
                            AND NewsDate='%s' AND NewsTitle='%s' AND ExpDays='%s'""" %(table_name, user, today_date, NewsIndex, NewsDate, NewsTitle, ExpDays) 
        cur.execute(check_exist_sql)
        check_exist=False if cur.fetchall()==[] else True
        if check_exist==False:
            try:
                sql = '''INSERT INTO %s (USER,TodayDate,NewsIndex,NewsDate,NewsTitle,NewsContext,ExpDays) 
                VALUES ('%s','%s','%s','%s','%s','%s','%s')''' %(table_name, user, today_date, NewsIndex, NewsDate, NewsTitle, NewsContext, ExpDays)
                cur.execute(sql)
                count+=1
            except:
                print('[Warning] Failed to insert sql')
#         else:
#             print('[Warning] Exist sql', NewsIndex)
            
    print('Success insert %d / %d' %(count, len(final_recommendation)))    
    con.commit()
    cur.close()
    con.close()




def main(DB_PATH, TODAY_DATE, USER_NAME, EXP_D):
    user_history = select_user_history_from_lab_sql(TODAY_DATE, USER_NAME, EXP_D)
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
    user_interests_network = UN.construct_user_interests_network(user_history_NewsIndex, user_history_NewsEmb, THRES)
    user_interests_cluster = UN.generate_user_interests_network_cluster(user_interests_network)
    UN.print_user_interests_network_cluster(user_interests_cluster)

    user_recommendation = generate_user_recommendation(user_history, user_interests_cluster, TOTAL_REC_LEN)
    
    final_recommendation = [[k[0], k[1], k[2], k[3], '%dd'%(EXP_D)] for k in user_recommendation]
    print('%s recommendation list length %d' %(USER_NAME, len(final_recommendation)) )
    return final_recommendation

def merge_recommendation_lists(list1, list2):
    result = list2.copy()
    list2_NewsTitle = [k[2] for k in list2]
    
    for i, k in enumerate(list1):
        if k[2] in list2_NewsTitle: # list1 news in list2
            idx = list2_NewsTitle.index(k[2])
#             print(k[0], k[-1], result[idx][-1])
            x = k[-1] + result[idx][-1]
            result[idx] = [k[0], k[1], k[2], k[3], x]
        else: # list1 news not in list2
            result.append(k)
    return result

def cluster_recommendation_lists(user_history_NewsIndex):
    user_history_NewsEmb = []
    for NewsIndex in user_history_NewsIndex:
        idx = idx_NewsIndex_map[NewsIndex]
        user_history_NewsEmb.append(NewsEmb_list[idx].tolist())
    
    UN = UserNetwork(DB_PATH, TODAY_DATE)
    user_interests_network = UN.construct_user_interests_network(user_history_NewsIndex, user_history_NewsEmb, THRES)
    user_interests_cluster = UN.generate_user_interests_network_cluster(user_interests_network)
    result = []
    for cluster in user_interests_cluster:
        for k in cluster:
            result.append(k)
    return result





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

USERS = pd.read_csv('./exp2_users.csv')['姓名'].values.tolist()
DB_PATH = './db/NewsNetwork_ch_20210629.db'
TODAY_DATE = '2021-06-29'
TIME_VERSION = 18

EXP_W = 0.1
EXP_W_NAME = str(EXP_W).replace('.','')
# EXP_D = 1           # 1d / 3d / 5d
THRES = 0.8         # User interests network edges thres
TOTAL_REC_LEN = 20  # Total rec len
TOP_N = 5           # Single news rec len

print('TODAY_DATE %s, TIME_VERSION %d, User length: %d' %(TODAY_DATE, TIME_VERSION, len(USERS)))

cross_news_network_path = "./cross-news-network/%s/%s" %(TODAY_DATE, TIME_VERSION)

gcn_model_path = "./gcn-model/%s/%s" %(TODAY_DATE, TIME_VERSION)
model = torch.load(os.path.join(gcn_model_path, 'gcn_model.pt'))
model.eval()

def generate_gcn_embeddings(model, adj, features):
    no_cuda = False
    if not no_cuda and torch.cuda.is_available():
        print('[INFO] Use Cuda!')
        model.cuda()
        features, adj = features.cuda(), adj.cuda()
    output = model(features, adj)
    return output

def normalization(matrix):
    minima, maxima = matrix.min(), matrix.max()
    return (matrix - minima)/(maxima - minima)


idx_NewsIndex_map, adj, features, X, Y = load_news_data(cross_news_network_path, EXP_W_NAME)

NewsEmb_list = features.cpu().detach().numpy()
NewsIndex_list = list(idx_NewsIndex_map.keys())
idx_list = list(idx_NewsIndex_map.values())

gcn_emb = generate_gcn_embeddings(model, adj, features)
gcn_emb = gcn_emb.cpu().detach().numpy()
gcn_similarity_matrix = 1 - pairwise_distances(gcn_emb, metric="cosine")
gcn_similarity_matrix = normalization(gcn_similarity_matrix)


for USER_NAME in USERS:
    s = time.time()
    final_recommendation_1 = main(DB_PATH, TODAY_DATE, USER_NAME, EXP_D=1)
    final_recommendation_3 = main(DB_PATH, TODAY_DATE, USER_NAME, EXP_D=3)
    final_recommendation_5 = main(DB_PATH, TODAY_DATE, USER_NAME, EXP_D=5)
    
    if final_recommendation_1==None and final_recommendation_3==None:
        continue
    elif final_recommendation_1==None:
        merge2 = merge_recommendation_lists(final_recommendation_3, final_recommendation_5) # 1d 3d 5d
    else:
        merge1 = merge_recommendation_lists(final_recommendation_1, final_recommendation_3) # 1d 3d
        merge2 = merge_recommendation_lists(merge1, final_recommendation_5) # 1d 3d 5d
    rec_merge_list = merge2
    rec_merge_NewsIndex = [k[0] for k in rec_merge_list]
    rec_NewsIndex_order = cluster_recommendation_lists(rec_merge_NewsIndex)
    final_recommendation = []
    for NewsIndex in rec_NewsIndex_order:
        idx = rec_merge_NewsIndex.index(NewsIndex)
        final_recommendation.append(rec_merge_list[idx])

close_ssh_tunnel()


