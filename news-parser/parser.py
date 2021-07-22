#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 09:09:59 2021

@author: gtingyou
"""

import time
import datetime
import emoji
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import logging

import parse_utils_db_v2




class NewsNodes(list):        
    def add_nodes(self, node: dict):
        self.append(node)
        
    def count_nodes(self):
        print('length of NewsNodes: ', len(self))
    
    def return_NewsIndex_in_NewsNodes(self, NewsURL: str):
        urls = [k['NewsURL'] for k in self]
        if NewsURL in urls:
            idx = urls.index(NewsURL)
            return self[idx]['NewsIndex']
        else:
            return None
    
#     def check_node_exist_in_NewsNodes(self, NewsURL: str):
#         urls = [k['NewsURL'] for k in self]
#         if NewsURL in urls:
#             return True
#         else:
#             return False
        
class NewsEdges(list):
    def __init__(self, ParseDate):
        self.ParseDate = ParseDate
        
    def add_edges(self, NewsIndex1, NewsIndex2):
        self.append([NewsIndex1, NewsIndex2, self.ParseDate])
        
    def count_edges(self):
        print('length of NewsEdges: ', len(self))
    
    def check_edge_exist_in_NewsEdges(self, NewsIndex1, NewsIndex2):
        edge = [NewsIndex1, NewsIndex2]
        edges = [[k[0],k[1]] for k in self] + [[k[1],k[0]] for k in self]
        if edge in edges:
            return True
        else:
            return False



class  NewsParser():
    def __init__(self, db_path: str, media_name: str, ParseDate: str, headline_news_parser, related_news_parser):
        self.db_path = db_path
        self.media_name = media_name
        self.ParseDate = ParseDate
        self.ParseDate_4_NewsIndex = ParseDate.replace('-', '')
        
        self.today_id = self.sqlite_last_today_id(self.db_path, self.media_name, self.ParseDate, self.ParseDate_4_NewsIndex)
        print('today_id: %d' %self.today_id)
        
        self.headline_news_parser = headline_news_parser
        self.related_news_parser = related_news_parser
        
        
        
    def sqlite_last_today_id(self, db_path, media_name, ParseDate, ParseDate_4_NewsIndex):
        x = parse_utils_db_v2.select_nodes_table(db_path, 
                                                 media_name+'news', 
                                                 'NewsIndex', 
                                                 "ParseDate=='%s' ORDER BY ABS(NewsIndex)" %(ParseDate))
        if x==[]:
            return 0 ## start of the day
        last_NewsIndex = x[-1][0]
        idx = last_NewsIndex.index(ParseDate_4_NewsIndex) + len(ParseDate_4_NewsIndex)
        last_today_id = last_NewsIndex[idx+1:]
        last_today_id = int(last_today_id) + 1
        return last_today_id

    
    def parse_source_layer(self, NewsNodes, NewsEdges, today_id):
        """ This function parsed the news source layer news (i.e. Hot / Latest news)
        node = {'NewsIndex':'', 'ParseDate':'', 'NewsURL':'', 'NewsDate':'', 'NewsTitle':'', 'NewsContext':'', 'Layer': 0}
        """
        
        headline_news = self.headline_news_parser()
        
        for n in headline_news:
            today_id += 1
            NewsIndex =  '%s_%s_%d' %(self.media_name, self.ParseDate_4_NewsIndex, today_id)
            node = {}
            node['Layer'] = 0
            node['NewsIndex'] = NewsIndex
            node['ParseDate'] = self.ParseDate
            node['NewsURL'] = n[0]
            node['NewsDate'] = n[1]
            node['NewsTitle'] = n[2]
            node['NewsContext'] = n[3]
            
            if NewsNodes.return_NewsIndex_in_NewsNodes(node['NewsURL'])==None:
                NewsNodes.add_nodes(node)
            
        NewsNodes.count_nodes()
        NewsEdges.count_edges()
        return NewsNodes, NewsEdges, today_id
    
    
    def parse_other_layer(self, NewsNodes, NewsEdges, today_id, layer):
        previous_layer = layer-1
        previous_layer_nodes = [k for k in NewsNodes if k['Layer']==previous_layer]
        print('previous layer %d nodes length %d' %(previous_layer, len(previous_layer_nodes)))
        
        for n in previous_layer_nodes:
            NewsIndex1 = n['NewsIndex']
            
            related_news = self.related_news_parser(n['NewsURL'])
            if len(related_news) == 0:
                print('*** No Related News ***')
                continue
                
            for r in related_news:
                r_url = r[0]
                # condition1: [] / parsed NewsIndex, condition1: None / parsed NewsIndex
                x = parse_utils_db_v2.select_nodes_table(self.db_path, self.media_name+'news', 'NewsIndex',"NewsURL like '%s%s' " %(r_url, '%'))
                condition1 = False if x==[] else True
                condition2 = False if NewsNodes.return_NewsIndex_in_NewsNodes(r)==None else True
    #             print(condition1, condition2)
                
                if condition1==False:
                    # case 1: cant find node in database, and cant find in today (add node and edge)
                    if condition2==False:
                        today_id += 1
                        NewsIndex2 =  '%s_%s_%d' %(self.media_name, self.ParseDate_4_NewsIndex, today_id)
                        node = {}
                        node['Layer'] = layer
                        node['NewsIndex'] = NewsIndex2
                        node['ParseDate'] = self.ParseDate
                        node['NewsURL'] = r[0]
                        node['NewsDate'] = r[1]
                        node['NewsTitle'] = r[2]
                        node['NewsContext'] = r[3]
                        NewsNodes.add_nodes(node)
                        print(NewsIndex1, NewsIndex2)
                        if NewsEdges.check_edge_exist_in_NewsEdges(NewsIndex1, NewsIndex2)==False:
                            NewsEdges.add_edges(NewsIndex1, NewsIndex2)
                    
                    # case 2: cant find node in database, but find in today (add edge)
                    elif condition2==True:
                        NewsIndex2 = NewsNodes.return_NewsIndex_in_NewsNodes(r)
                        print(NewsIndex1, NewsIndex2)
                        if NewsEdges.check_edge_exist_in_NewsEdges(NewsIndex1, NewsIndex2)==False:
                            NewsEdges.add_edges(NewsIndex1, NewsIndex2)
                
                # case 3: find node in database (add edge)
                elif condition1==True:
                    NewsIndex2 = x[0][0]
                    print(NewsIndex1, NewsIndex2)
                    if NewsEdges.check_edge_exist_in_NewsEdges(NewsIndex1, NewsIndex2)==False:
                        NewsEdges.add_edges(NewsIndex1, NewsIndex2)
        
        NewsNodes.count_nodes()
        NewsEdges.count_edges()
        return NewsNodes, NewsEdges, today_id
    
    
    def output_NewsNodes_NewsEdges_to_sqlite(self, NewsNodes, NewsEdges):
        start_time = datetime.datetime.now()
        
        output_nodes = [[k['NewsIndex'],k['ParseDate'],k['NewsURL'],k['NewsDate'],k['NewsTitle'],k['NewsContext']] for k in NewsNodes]
        output_edges = NewsEdges
        parse_utils_db_v2.insert_nodes_table(self.db_path, self.media_name+'news', output_nodes)
        parse_utils_db_v2.insert_edges_table(self.db_path, self.media_name+'connection', output_edges)
        
        print('Total loading database time: ',datetime.datetime.now()-start_time)
    


if __name__ == '__main__':
    print('main')
    
    
    
    
    
    
    

