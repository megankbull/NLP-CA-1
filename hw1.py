from curses import COLS
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

import re
from bs4 import BeautifulSoup
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

F_PATH = 'amazon_reviews_us_Jewelry_v1_00.tsv'

STAR_H = 'star_rating'
REVIEW_H = 'review_body'

COLS=[STAR_H, REVIEW_H]

VALID_RATS = {'1', '2', '3', '4', '5'}
SAMPLE_SIZE = 20000


df = pd.read_csv(F_PATH, sep='\t', usecols=COLS, low_memory=False)
df.dropna(inplace=True)

grouped = df.groupby(STAR_H)
rat_dfs = [grouped.get_group(rating).sample(n=SAMPLE_SIZE) for rating in VALID_RATS]

sampled = pd.concat(rat_dfs)

raw_len_avg = sampled[REVIEW_H].str.len().mean()

print(f'Average character length pre-clean: {raw_len_avg}')

def gen_clean(text):
    """
    gen text cleanup 
    incl removal: extended ws, html tags, urls
    """
    text = BeautifulSoup(text, "html.parser").text #rm html tags 
    text = re.sub(r'http\S+', r'', text)
    text = contractions.fix(text)

    for c in text: 
        if not c.isalpha():
            text = text.replace(c, ' ')

    text = re.sub(" +", " ", text)
    
    return text.lower()

sampled[REVIEW_H] = sampled[REVIEW_H].apply(gen_clean)
sampled.sort_index(inplace=True)


cl_len_avg = sampled[REVIEW_H].str.len().mean()

print(f'Average character length post-clean: {cl_len_avg}')


def rm_stops(text): 
   """
   remove stop words from text 
   """
   stops = set(stopwords.words("english"))
   sans_stops = [tok for tok in word_tokenize(text) if tok not in stops]
   return " ".join(sans_stops).strip()

sampled[REVIEW_H] = sampled[REVIEW_H].apply(rm_stops)


wnl = WordNetLemmatizer()

def lemmatize(text): 
   lemmas = [wnl.lemmatize(w) for w in word_tokenize(text)]
   return " ".join(lemmas)
   
sampled[REVIEW_H] = sampled[REVIEW_H].apply(lemmatize)

preproc_len_avg = sampled[REVIEW_H].str.len().mean()

print(f'Average character length after preproc: {preproc_len_avg}')