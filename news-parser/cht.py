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



def clean_cht_url(url):
    if '?' in url:
        url = url[:url.find('?')]
    return url


def cht_headline_news_parser():
    cht_headline_address = []
    cht_headline_news = []
    
    for page_num in range(1, 4):
        print('cht realtimenews page number: %d' %page_num)
        res = requests.get('https://www.chinatimes.com/realtimenews/?page=%d&chdtv' %page_num)
        soup = BeautifulSoup(res.text,"html.parser")
        for link in soup.select('section.article-list h3.title a'):
            r = 'https://www.chinatimes.com' + link.get('href')
            r = clean_cht_url(r)
            if r not in cht_headline_address:
                if parse_utils_db_v2.select_nodes_table(db_path, 'CHTnews', '*', "NewsURL like '%s%s'" %(r, '%') )==[]:
                    cht_headline_address.append(r)
                else:
                    print('***URL IS IN DATABASE***', r)
            
    cht_headline_address = [clean_cht_url(k) for k in cht_headline_address]
    
    for article_url in cht_headline_address:
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
            cht_headline_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    
    return cht_headline_news


def cht_related_news_parser(url):
    print('Parse related news of: ' + url) 
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")    
    
    cht_related_address = []
    cht_related_news = []

    for link in soup.select('div.article-body div.promote-word a'):
        if link.get('href') not in cht_related_address and 'chinatimes.com/' in link.get('href'):
            cht_related_address.append(link.get('href'))            
    for link in soup.select('section.recommended-article h4.title a'):
        if link.get('href') not in cht_related_address and 'chinatimes.com/' in link.get('href'):
            cht_related_address.append(link.get('href'))
    
    cht_related_address = [clean_cht_url(k) for k in cht_related_address]
    
    for article_url in cht_related_address:
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        if article.text!='' and article.title!='':
            cht_related_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    
    return cht_related_news





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--db-path", default="/home/ubuntu/DailyNews/NewsNetwork_ch.db")
    parser.add_argument("-l","--num_of-layer", default=3)    
    args = parser.parse_args()
    
    # params
    db_path = args.db_path
    media_name = 'CHT'
    ParseDate = time.strftime("%Y-%m-%d", time.localtime())
    
    
    print("Start parsing time:", datetime.datetime.now())
    
    cht_NewsParser = NewsParser(db_path, 
                                media_name, 
                                ParseDate, 
                                headline_news_parser = cht_headline_news_parser, 
                                related_news_parser = cht_related_news_parser)
    
    today_id = cht_NewsParser.today_id
    cht_NewsNodes = NewsNodes()
    cht_NewsEdges = NewsEdges(ParseDate)
    
    cht_NewsNodes, cht_NewsEdges, today_id = cht_NewsParser.parse_source_layer(cht_NewsNodes, cht_NewsEdges, today_id)
    
    for layer in range(1, args.num_of_layer):
        print('\n\n---------- layer %d ----------\n\n' %layer)
        cht_NewsNodes, cht_NewsEdges, today_id = cht_NewsParser.parse_other_layer(cht_NewsNodes, cht_NewsEdges, today_id, layer)
    
    
    cht_NewsParser.output_NewsNodes_NewsEdges_to_sqlite(cht_NewsNodes, cht_NewsEdges)
    
    print("End time:", datetime.datetime.now())
    
    
    
    

