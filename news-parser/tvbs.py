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



def clean_tvbs_url(url):
    if '?' in url:
        url = url[:url.find('?')]
    return url


def tvbs_headline_news_parser():
    res = requests.get('https://news.tvbs.com.tw/')
    soup = BeautifulSoup(res.text,"html.parser")

    tvbs_headline_address = []
    tvbs_headline_news = []
    
    for link in soup.select('div.content_center_list_box > div > ul > li > a'):
        r = link.get('href')
        r = 'https://news.tvbs.com.tw' + r
        r = clean_tvbs_url(r)
        if r not in tvbs_headline_address:
            if parse_utils_db_v2.select_nodes_table(db_path, 'TVBSnews', '*', "NewsURL like '%s'" %r)==[]:
                tvbs_headline_address.append(r)
#             else:
#                 print('[INFO] URL IN DATABASE', r)
    
    for article_url in tvbs_headline_address:
        print(article_url)
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('[INFO] FAILED TO DOWNLOAD ', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        if article.text != '':
            tvbs_headline_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    return tvbs_headline_news


def tvbs_related_news_parser(url):
    print('Parse related news of: ' + url) 
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")    
    
    tvbs_related_address = []
    tvbs_related_news = []
    
    for link in soup.select('div.article_extended div ul li h2 a'):
        r = link.get('href')
        r = 'https://news.tvbs.com.tw' + r
        r = clean_tvbs_url(r)
        if r not in tvbs_related_address:
            tvbs_related_address.append(r)
    
    for article_url in tvbs_related_address:
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        if article.text != '':
            tvbs_related_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    return tvbs_related_news





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--db-path", default="/home/ubuntu/DailyNews/NewsNetwork_ch.db")
    parser.add_argument("-l","--num_of-layer", default=3)    
    args = parser.parse_args()
    
    # params
    db_path = args.db_path
    media_name = 'TVBS'
    ParseDate = time.strftime("%Y-%m-%d", time.localtime())
    
    
    print("Start parsing time:", datetime.datetime.now())
    
    tvbs_NewsParser = NewsParser(db_path, 
                                media_name, 
                                ParseDate, 
                                headline_news_parser = tvbs_headline_news_parser, 
                                related_news_parser = tvbs_related_news_parser)
    
    today_id = tvbs_NewsParser.today_id
    tvbs_NewsNodes = NewsNodes()
    tvbs_NewsEdges = NewsEdges(ParseDate)
    
    tvbs_NewsNodes, tvbs_NewsEdges, today_id = tvbs_NewsParser.parse_source_layer(tvbs_NewsNodes, tvbs_NewsEdges, today_id)
    
    for layer in range(1, args.num_of_layer):
        print('\n\n---------- layer %d ----------\n\n' %layer)
        tvbs_NewsNodes, tvbs_NewsEdges, today_id = tvbs_NewsParser.parse_other_layer(tvbs_NewsNodes, tvbs_NewsEdges, today_id, layer)
    
    
    tvbs_NewsParser.output_NewsNodes_NewsEdges_to_sqlite(tvbs_NewsNodes, tvbs_NewsEdges)
    
    print("End time:", datetime.datetime.now())
    
    
    
    
    

