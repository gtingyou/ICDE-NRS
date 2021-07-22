#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:44:39 2021

@author: gtingyou
"""

import time
import datetime
import logging
import emoji
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import argparse

import parse_utils_db_v2
from parser import NewsNodes, NewsEdges, NewsParser



def clean_setn_url(url):
    if '&utm_source' in url:
        url = url[:url.find('&utm_source')]
    elif '?utm_source' in url:
        url = url[:url.find('?utm_source')]
    elif '&Area' in url:
        url = url[:url.find('&Area')]
    return url


def setn_headline_news_parser():
    res = requests.get('https://www.setn.com/ViewAll.aspx')
    soup = BeautifulSoup(res.text,"html.parser")

    setn_headline_address = []
    setn_headline_news = []
    
    for link in soup.select('h3.view-li-title a.gt'):
        if 'https://www.setn.com'+link.get('href') not in setn_headline_address:
            if 'setn.com/' not in link.get('href'):
                r = clean_setn_url('https://www.setn.com'+link.get('href'))
            else:
                r = clean_setn_url(link.get('href'))
            
            if parse_utils_db_v2.select_nodes_table(db_path, 'SETNnews', '*', "NewsURL like '%s%s'" %(r, '%') )==[]:
                setn_headline_address.append(r)
    
    for article_url in setn_headline_address[:30]:
        print(article_url)
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        if article.text != '':
            setn_headline_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    
    return setn_headline_news


def setn_related_news_parser(url):
    print('Parse related news of: ' + url) 
    
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")    
    
    setn_related_address = []
    setn_related_news = []
    
    # 延伸閱讀
#    for link in soup.select('div.hidden-print.bottom-buffer > div#involve.extend.news-list > ul > li > a.gt'):
#        if link.get('href') not in setn_related_address:
#            setn_related_address.append('https://www.setn.com/'+link.get('href'))   
    # 大數據推薦 ---> 相關新聞
    for link in soup.select('div.extend.news-list ul li a.gt'):
        if link.get('href') not in setn_related_address:
            setn_related_address.append('https://www.setn.com/'+link.get('href'))         
            
    
    setn_related_address = [clean_setn_url(k) for k in setn_related_address]
    
    for article_url in setn_related_address:
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        if article.text != '':
            setn_related_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    
    return setn_related_news





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--db-path", default="/home/ubuntu/DailyNews/NewsNetwork_ch.db")
    parser.add_argument("-l","--num_of-layer", default=3)    
    args = parser.parse_args()
    
    # params
    db_path = args.db_path
    media_name = 'SETN'
    ParseDate = time.strftime("%Y-%m-%d", time.localtime())
    
    
    print("Start parsing time:", datetime.datetime.now())
    
    setn_NewsParser = NewsParser(db_path, 
                                media_name, 
                                ParseDate, 
                                headline_news_parser = setn_headline_news_parser, 
                                related_news_parser = setn_related_news_parser)
    
    today_id = setn_NewsParser.today_id
    setn_NewsNodes = NewsNodes()
    setn_NewsEdges = NewsEdges(ParseDate)
    
    setn_NewsNodes, setn_NewsEdges, today_id = setn_NewsParser.parse_source_layer(setn_NewsNodes, setn_NewsEdges, today_id)
    
    for layer in range(1, args.num_of_layer):
        print('\n\n---------- layer %d ----------\n\n' %layer)
        setn_NewsNodes, setn_NewsEdges, today_id = setn_NewsParser.parse_other_layer(setn_NewsNodes, setn_NewsEdges, today_id, layer)
    
    
    setn_NewsParser.output_NewsNodes_NewsEdges_to_sqlite(setn_NewsNodes, setn_NewsEdges)
    
    print("End time:", datetime.datetime.now())
    
    
    
    

