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



def clean_cna_url(url):
    return url


def cna_headline_news_parser():
#    res = requests.get('https://www.cna.com.tw/') # 主頁面
    res = requests.get('https://www.cna.com.tw/list/aall.aspx') # 即時新聞
    soup = BeautifulSoup(res.text,"html.parser")

    cna_headline_address = []
    cna_headline_news = []
    
    for link in soup.select('div.statement > ul > li > a'):
        r = link.get('href')
        r = clean_cna_url(r)
        if r not in cna_headline_address:
            if parse_utils_db_v2.select_nodes_table(db_path, 'CNAnews', '*', "NewsURL like '%s%s'" %(r, '%') )==[]:
                cna_headline_address.append(r)
            else:
                print('***URL IS IN DATABASE***', r)
    
    for article_url in cna_headline_address:
        print(article_url)
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        if article_date==str(None):
            r = requests.get(article_url)
            s = BeautifulSoup(r.text,"html.parser")
            try:
                article_date = s.select('div.updatetime > span')[0].text
                article_date = article_date[:article_date.find(' ')]
                year = article_date[:article_date.find('/')]
                article_date = article_date[article_date.find('/')+1:]
                month = article_date[:article_date.find('/')]
                month = "%02d" %(int(month))
                day = article_date[article_date.find('/')+1:]
                day = "%02d" %(int(day))
                article_date = '%s-%s-%s' %(year, month, day)
            except:
                print('***FAILED TO GET DATE***', article_url)
                continue
        if article.text != '':
            cna_headline_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    
    return cna_headline_news


def cna_related_news_parser(url):
    print('Parse related news of: ' + url) 
    
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")    
    
    cna_related_address = []
    cna_related_news = []
    
    for link in soup.select('div.paragraph.moreArticle a.moreArticle-link'):
        if link.get('href') not in cna_related_address:
            cna_related_address.append(link.get('href'))         
            
    cna_related_address = [clean_cna_url(k) for k in cna_related_address]
    
    for article_url in cna_related_address:
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        if '綠光人間條件一免費線上看' in article.text:
            continue
        article_date = str(article.publish_date)[:10]
        if article_date==str(None):
            r = requests.get(article_url)
            s = BeautifulSoup(r.text,"html.parser")
            try:
                article_date = s.select('div.updatetime > span')[0].text
                article_date = article_date[:article_date.find(' ')]
                year = article_date[:article_date.find('/')]
                article_date = article_date[article_date.find('/')+1:]
                month = article_date[:article_date.find('/')]
                month = "%02d" %(int(month))
                day = article_date[article_date.find('/')+1:]
                day = "%02d" %(int(day))
                article_date = '%s-%s-%s' %(year, month, day)
            except:
                print('***FAILED TO GET DATE***', article_url)
                continue
        if article.text != '':
            cna_related_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    
    return cna_related_news





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--db-path", default="/home/ubuntu/DailyNews/NewsNetwork_ch.db")
    parser.add_argument("-l","--num_of-layer", default=8)    
    args = parser.parse_args()
    
    # params
    db_path = args.db_path
    media_name = 'CNA'
    ParseDate = time.strftime("%Y-%m-%d", time.localtime())
    
    
    print("Start parsing time:", datetime.datetime.now())
    
    cna_NewsParser = NewsParser(db_path, 
                                media_name, 
                                ParseDate, 
                                headline_news_parser = cna_headline_news_parser, 
                                related_news_parser = cna_related_news_parser)
    
    today_id = cna_NewsParser.today_id
    cna_NewsNodes = NewsNodes()
    cna_NewsEdges = NewsEdges(ParseDate)
    
    cna_NewsNodes, cna_NewsEdges, today_id = cna_NewsParser.parse_source_layer(cna_NewsNodes, cna_NewsEdges, today_id)
    
    for layer in range(1, args.num_of_layer):
        print('\n\n---------- layer %d ----------\n\n' %layer)
        cna_NewsNodes, cna_NewsEdges, today_id = cna_NewsParser.parse_other_layer(cna_NewsNodes, cna_NewsEdges, today_id, layer)
    
    
    cna_NewsParser.output_NewsNodes_NewsEdges_to_sqlite(cna_NewsNodes, cna_NewsEdges)
    
    print("End time:", datetime.datetime.now())
    
    
    
    
    

