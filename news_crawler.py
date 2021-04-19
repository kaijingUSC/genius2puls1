import requests
import urllib
import sys
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def get_internal_links(ur):
    info = BeautifulSoup(requests.get(ur, allow_redirects=True).content, 'html.parser').find_all("div", {
        "class": "article-card__details"})
    links = ["https://financialpost.com/"+a['href'] for each_link in info for a in each_link.find_all('a', {"class":"article-card__link"},href=True)]

    return (links)


def extract_link_of_news(k_word, n_of_page):
    news_df = pd.DataFrame()

    link = "https://financialpost.com/search/?search_text=" + k_word + "&search_text="+k_word+"&date_range=-365d&sort=score&from="+str(n_of_page*10)

    print(link)
    print(get_internal_links(link))
    news_df["internal_urls"] = get_internal_links(link)
    news_df["principal_url"] = link
    news_df["n_of_page"] = n_of_page

    return (news_df[["n_of_page", "principal_url", "internal_urls"]])


def extract_date_and_text(ur):
    url = ur

    try:
        soup = BeautifulSoup(requests.get(url).content, 'html.parser')
        g = 0

    except:
        the_type, the_value, the_traceback = sys.exc_info()
        g = 1

    if (g == 0):

        if (soup.find("section", class_="article-content__content-group").find_all("p")):
            text_of_news = soup.find("section", class_="article-content__content-group").find_all("p")
        elif (soup.find("section", class_="article-content__content-group").find("p")):
            text_of_news = soup.find("section", class_="article-content__content-group").find("p")
        else:
            text_of_news = "NO TEXT"

        final_date = soup.find("span",class_="published-date__since").text

        get_text_vec = [i.text for i in text_of_news]
        final_text = "".join(get_text_vec)


    else:
        final_date = "";final_text = ""
    print(final_date)
    print(final_text)
    return (final_date, final_text)



# print(extract_link_of_news('microsoft',1))

n_of_pages = 330
df = pd.concat([extract_link_of_news("microsoft",i) for i in range(1,n_of_pages+1)],ignore_index = True)


# a = np.vectorize(extract_date_and_text)
#
# test = df[0:10].copy()
# test["internals_dates"],test["internals_text"] = a(test["internal_urls"][0:10])
# test.head()
# print(test.head())
# print(test)
# print(a(test["internal_urls"][0:10]))


a = np.vectorize(extract_date_and_text)
df["internals_dates"],df["internals_text"] = a(df["internal_urls"])

df.loc[df["internals_text"] == "NO TEXT"]
df.loc[df["internals_text"] == ""]

df.to_pickle("data/microsoft_news_text.pkl")


df3 = pd.read_pickle("data/microsoft_news_text.pkl")
print(df3.head())