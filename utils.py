import sqlite3
import numpy as np
import pandas as pd
import scipy.sparse
import re
import torch
from torch import Tensor



def pytorch_cos_sim(a: Tensor, b: Tensor):
    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def select_nodes_table(database_name, table_name, columns, conditions):
    conn = sqlite3.connect(database_name)
    cur = conn.cursor()
    myresult = cur.execute('''SELECT {1}
                            FROM {0} WHERE {2}'''
                            .format(table_name,columns,conditions) )
    result = []
    for row in myresult:
        result.append(row)
    conn.close()    
    return result

def get_media_name_via_NewsIndex(NewsIndex):
    media_name = None
    if 'CHT' in NewsIndex: media_name = 'CHT'
    elif 'CNA' in NewsIndex: media_name = 'CNA'
    elif 'LBT' in NewsIndex: media_name = 'LBT'
    elif 'SETN' in NewsIndex: media_name = 'SETN'
    elif 'TVBS' in NewsIndex: media_name = 'TVBS'
    elif 'UDN' in NewsIndex: media_name = 'UDN'
    return media_name

def get_media_name_via_NewsURL(NewsURL):
    media_name = None
    if 'www.chinatimes.com' in NewsURL: media_name = 'CHT'
    elif 'www.cna.com.tw' in NewsURL: media_name = 'CNA'
    elif 'ltn.com.tw' in NewsURL: media_name = 'LBT'
    elif 'www.setn.com' in NewsURL: media_name = 'SETN'
    elif 'news.tvbs.com.tw' in NewsURL: media_name = 'TVBS'
    elif 'udn.com' in NewsURL: media_name = 'UDN'
    return media_name

def clean_media_NewsURL(media_name, url):
    if media_name == 'CHT':
        if '?' in url:
            url = url[:url.find('?')]
        if re.search("^(https:\/\/)+[A-Za-z0-9\.\-\/]+\/\d{14}\-\d{6}", url)!=None:
            return url
        else:
            return None
            
    elif media_name=='CNA':
        if '?' in url:
            url = url[:url.find('?')]
        if re.search("^(https:\/\/www.cna.com.tw)+[A-Za-z0-9\.\-\/]+\/\d{12}.aspx", url)!=None:
            return url
        else:
            return None
        
    elif media_name=='LBT':
        if '?' in url:
            url = url[:url.find('?')]
        if re.search("^(https:\/\/)+[A-Za-z]+(\.ltn.com.tw\/)+[A-Za-z0-9\.\-\/]+\/\d{7}", url)!=None:
            return url
        else:
            return None
        
    elif media_name=='SETN':
        if '&Area' in url:
            url = url[:url.find('&Area')]
        elif '&utm_source' in url:
            url = url[:url.find('&utm_source')]
        if re.search("^(https:\/\/www.setn.com)+[A-Za-z0-9\.\-\/]+(\?NewsID=)+\d{6}", url)!=None:
            return url
        else:
            return None
        
    elif media_name=='TVBS':
        if '?' in url:
            url = url[:url.find('?')]
        if re.search("^(https:\/\/news.tvbs.com.tw)+[A-Za-z0-9\.\-\/]+\/\d{7}", url)!=None:
            return url
        else:
            return None
        
    elif media_name=='UDN':
        if '?' in url:
            url = url[:url.find('?')]
        if re.search("^(https:\/\/)+[a-z\.\/]*(udn.com\/)+[a-z\_\.\/]+\/\d{4,6}\/\d{7}", url)!=None:
            return url
        else:
            return None
    
    