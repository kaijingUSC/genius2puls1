#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup


def search_stock(key):

    news = []
    
    start_page = 'https://investing.com/search/?q=' + key
    page = requests.get(start_page, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(page.text, 'html.parser')
    
    search_link = soup.find(class_='searchSectionMain').find('a').get('href')
    stock_link = 'https://investing.com' + search_link + '-news'
    stock_page = requests.get(stock_link, headers={"User-Agent": "Mozilla/5.0"})
    stock_soup = BeautifulSoup(stock_page.text, 'html.parser')
    
    news_article = stock_soup.find_all('article')
    for each in news_article:
        if(each.find(class_='date')):
            # print(each.find(class_='date').prettify())
            # print(each.find(class_='title').prettify())
            news.append([each.find(class_='date').contents[0][3:], each.find(class_='title').contents[0]])

    return news
