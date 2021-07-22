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



def clean_udn_url(url):
    if '?' in url:
        url = url[:url.find('?')]
    return url


def udn_headline_news_parser():
    res = requests.get('https://udn.com/news/breaknews/1/99')
    soup = BeautifulSoup(res.text,"html.parser")

    udn_headline_address = []
    udn_headline_news = []
    
    for link in soup.select('section div.story-list__text h2 a'):
        if 'stars.udn.com' in link.get('href'):
            r = link.get('href')
        else:
            r = 'https://udn.com'+link.get('href')
        r = clean_udn_url(r)
        if parse_utils_db_v2.select_nodes_table(db_path, 'UDNnews', '*', "NewsURL like '%s%s'" %(r, '%') )==[]:
            if r not in udn_headline_address:
                udn_headline_address.append(r)
    
    for article_url in udn_headline_address:
#         time.sleep(1)
        print(article_url)
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        # 如果article_date是空值，爬日期
        if article_date==str(None):
            r = requests.get(article_url)
            s = BeautifulSoup(r.text,"html.parser")
            try:
                if 'stars.udn.com' in article_url:
                    article_date = s.select('div.shareBar__info--author span')[0].text[:10]
                else:
                    article_date = s.select('time.article-content__time')[0].string[:10]
            except:
                print('***FAILED TO GET DATE***', article_url)
                continue
        if article.text!= '':
            udn_headline_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    
    return udn_headline_news


def udn_related_news_parser(url):
    print('Parse related news of: ' + url) 
    
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")    
    
    udn_related_address = []
    udn_related_news = []

    for link in soup.select('div.story-list__news div h2 a'):
        if link.get('href') not in udn_related_address and 'udn.com/' in link.get('href'):
            udn_related_address.append(clean_udn_url(link.get('href')))
    
    for article_url in udn_related_address:
        time.sleep(1)
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        # 如果article_date是空值，爬日期
        if article_date==str(None):
            r = requests.get(article_url)
            s = BeautifulSoup(r.text,"html.parser")
            try:
                if 'stars.udn.com' in article_url:
                    article_date = s.select('div.shareBar__info--author span')[0].text[:10]
                else:
                    article_date = s.select('time.article-content__time')[0].string[:10]
            except:
                print('***FAILED TO GET DATE***', article_url)
                continue
        if article.text != '':
            udn_related_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
    
    return udn_related_news





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--db-path", default="/home/ubuntu/DailyNews/NewsNetwork_ch.db")
    parser.add_argument("-l","--num_of-layer", default=3)    
    args = parser.parse_args()
    
    # params
    db_path = args.db_path
    media_name = 'UDN'
    ParseDate = time.strftime("%Y-%m-%d", time.localtime())
    
    
    print("Start parsing time:", datetime.datetime.now())
    
    udn_NewsParser = NewsParser(db_path, 
                                media_name, 
                                ParseDate, 
                                headline_news_parser = udn_headline_news_parser, 
                                related_news_parser = udn_related_news_parser)
    
    today_id = udn_NewsParser.today_id
    udn_NewsNodes = NewsNodes()
    udn_NewsEdges = NewsEdges(ParseDate)
    
    udn_NewsNodes, udn_NewsEdges, today_id = udn_NewsParser.parse_source_layer(udn_NewsNodes, udn_NewsEdges, today_id)
    
    for layer in range(1, args.num_of_layer):
        print('\n\n---------- layer %d ----------\n\n' %layer)
        udn_NewsNodes, udn_NewsEdges, today_id = udn_NewsParser.parse_other_layer(udn_NewsNodes, udn_NewsEdges, today_id, layer)
    
    
    udn_NewsParser.output_NewsNodes_NewsEdges_to_sqlite(udn_NewsNodes, udn_NewsEdges)
    
    print("End time:", datetime.datetime.now())
    
    
    
    

