import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd

analyser = SentimentIntensityAnalyzer()
### data preparation
df = pd.read_pickle("data/microsoft_news_text.pkl")
text = df['internals_text']

def split_by_dot(x):
    return(x.split("."))

df["text_split"] = df["internals_text"].apply(split_by_dot)
print(df["text_split"])
df["time"] = pd.to_datetime(df["internals_dates"])
df = df.sort_values("time")
df = df.set_index("time")

print(df.head())

### sentiment analysis
def sentimental_analysis_by_phrase(y):
    y = list(map(lambda x: analyser.polarity_scores(x)["compound"], y))
    y = np.array(y)
    y = y[y != 0]
    return (y)

text_2 = df["text_split"][0]
print(text_2)

sen_res = sentimental_analysis_by_phrase(text_2)
print(sen_res)

df["sentimental_analysis_phrase"] = df["text_split"].apply(sentimental_analysis_by_phrase)

## get average score
df["sentimental_analysis_average"] = df["sentimental_analysis_phrase"].apply(np.mean)


def sentimental_analysis(y):
    return (analyser.polarity_scores(y)["compound"])
df["sentimental_analysis_score"] = df["internals_text"].apply(sentimental_analysis)
print(df["sentimental_analysis_score"])

plt.style.use('seaborn-dark')
df[["sentimental_analysis_average","sentimental_analysis_score"]].plot(cmap = "jet",linestyle='-',figsize = (15,10))
plt.grid()
plt.show()


df.to_pickle("data/sentiments_microsoft_news.pkl")
