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



def clean_lbt_url(url):
    return url


def lbt_headline_news_parser():
    lbt_headline_address = []
    lbt_headline_news = []

    res = requests.get('https://news.ltn.com.tw/list/breakingnews')
    soup = BeautifulSoup(res.text,"html.parser")

    for link in soup.select('div.whitecon.boxTitle ul li a'):
        r = link.get('href')
        r = clean_lbt_url(r)
        if r not in lbt_headline_address:
            if parse_utils_db_v2.select_nodes_table(db_path, 'LBTnews', '*', "NewsURL like '%s%s'" %(r, '%') )==[]:
                lbt_headline_address.append(r)
            else:
                print('***URL IS IN DATABASE***', r)

    for article_url in lbt_headline_address:
        print(article_url)
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]

        if article.text!='':
            lbt_headline_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
        else:
            article_text=''
            r = requests.get(article_url)
            s = BeautifulSoup(r.text,"html.parser")    
            for p in s.select('div.text p'):
                if p.has_attr('class')==False:
                    if p.find_all()==[]:
                        article_text+=p.text
            if article_text!='':
                lbt_headline_news.append([article_url, article_date, article.title, emoji.demojize(article_text)])
            else:
                print('***FAILED TO PARSE NEWSCONTEXT***', article_url)
    return lbt_headline_news


def lbt_related_news_parser(url):
    import random
    time.sleep(random.randint(1,10))
    
    print('Parse related news of: ' + url)
    lbt_related_address = []
    lbt_related_news = []

    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")    

    if soup.select('div.related.boxTitle a')!=[]:
        for link in soup.select('div.related.boxTitle a'):
            if link.get('href') not in lbt_related_address and 'ltn.com.tw' in link.get('href'):
                lbt_related_address.append(link.get('href'))
    else: ### for lbt finance, sports news
        for link in soup.select('ul.related.boxTitle a'):
            if link.get('href') not in lbt_related_address and 'ltn.com.tw' in link.get('href'):
                lbt_related_address.append(link.get('href'))

    lbt_related_address = [k for k in lbt_related_address if 'click?ano' not in k]
    print(len(lbt_related_address))
    for article_url in lbt_related_address:
        time.sleep(1)
        article = Article(article_url)
        try:
            article.download()
            article.parse()
        except:
            print('***FAILED TO DOWNLOAD***', article_url)
            continue
        article_date = str(article.publish_date)[:10]
        
        if article.text!='':
            lbt_related_news.append([article_url, article_date, article.title, emoji.demojize(article.text)])
        else:
            article_text=''
            r = requests.get(article_url)
            s = BeautifulSoup(r.text,"html.parser")    
            for p in s.select('div.text p'):
                if p.has_attr('class')==False:
                    if p.find_all()==[]:
                        article_text+=p.text
            if article_text!='':
                lbt_related_news.append([article_url, article_date, article.title, emoji.demojize(article_text)])
            else:
                print('***FAILED TO PARSE NEWSCONTEXT***', article_url)
    return lbt_related_news





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--db-path", default="/home/ubuntu/DailyNews/NewsNetwork_ch.db")
    parser.add_argument("-l","--num_of-layer", default=3)    
    args = parser.parse_args()
    
    # params
    db_path = args.db_path
    media_name = 'LBT'
    ParseDate = time.strftime("%Y-%m-%d", time.localtime())
    
    
    print("Start parsing time:", datetime.datetime.now())
    
    lbt_NewsParser = NewsParser(db_path, 
                                media_name, 
                                ParseDate, 
                                headline_news_parser = lbt_headline_news_parser, 
                                related_news_parser = lbt_related_news_parser)
    
    today_id = lbt_NewsParser.today_id
    lbt_NewsNodes = NewsNodes()
    lbt_NewsEdges = NewsEdges(ParseDate)
    
    lbt_NewsNodes, lbt_NewsEdges, today_id = lbt_NewsParser.parse_source_layer(lbt_NewsNodes, lbt_NewsEdges, today_id)
    
    for layer in range(1, args.num_of_layer):
        print('\n\n---------- layer %d ----------\n\n' %layer)
        lbt_NewsNodes, lbt_NewsEdges, today_id = lbt_NewsParser.parse_other_layer(lbt_NewsNodes, lbt_NewsEdges, today_id, layer)
    
    
    lbt_NewsParser.output_NewsNodes_NewsEdges_to_sqlite(lbt_NewsNodes, lbt_NewsEdges)
    
    print("End time:", datetime.datetime.now())
    
    

