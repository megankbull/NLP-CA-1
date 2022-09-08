# ! python3 -m pip install bs4 
# ! python3 -m pip install pandas 
# ! python3 -m pip install nltk
# ! python3 -m pip install textacy
# ! python3 -m pip install sklearn
# ! python3 -m pip install contractions

# python version 3.10.6

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) 
nltk.download('punkt', quiet=True) 

from textacy.preprocessing import remove, normalize, replace

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression

import warnings 
import contractions

warnings.filterwarnings('ignore')

# GLOBALS 

F_PATH = 'amazon_reviews_us_Jewelry_v1_00.tsv'

STAR_H = 'star_rating'
REVIEW_H = 'review_body'

COLS=[STAR_H, REVIEW_H]

VAL_STARS = {'1', '2', '3', '4', '5'}

WNL = WordNetLemmatizer()

def read_data(f_path=F_PATH):

    df = pd.read_csv(f_path, sep='\t', usecols=COLS, low_memory=False)
    df.dropna(inplace=True)
    return df

def get_sample(df, s_size=20000):

    grouped = df.groupby(STAR_H)
    rat_dfs = [grouped.get_group(rating).sample(n=s_size) for rating in VAL_STARS]
    return pd.concat(rat_dfs) 

def gen_clean(text):
    """
    gen text cleanup 
    incl removal: extended ws, html tags, urls
    """
    text = BeautifulSoup(text, "html.parser").text #rm html tags 
    text = replace.urls(text, '')
    text = contractions.fix(text)
    text = remove.punctuation(text)
    text = normalize.whitespace(text)
    return text.lower()
   
def rm_stops(text): 
    """
    remove stop words from text 
    """
    stops = set(stopwords.words("english"))
    sans_stops = [tok for tok in word_tokenize(text) if tok not in stops]
    return " ".join(sans_stops).strip()

def lemmatize(text): 
    lemmas = [WNL.lemmatize(w) for w in word_tokenize(text)]
    return " ".join(lemmas).strip()

def print_report(test_labels, test_pred):
    classific_dict = classification_report(test_labels, test_pred, output_dict=True)
    for k, v in classific_dict.items():
        if k in VAL_STARS or k == 'macro avg':
            print(f"{v['precision']}, {v['recall']}, {v['f1-score']}")
    print()
    
def main(): 
    df = read_data()
    sampled = get_sample(df)

    sampled[REVIEW_H] = sampled[REVIEW_H].apply(gen_clean)
    sampled[REVIEW_H] = sampled[REVIEW_H].apply(rm_stops)
    sampled[REVIEW_H] = sampled[REVIEW_H].apply(lemmatize)

    v = TfidfVectorizer(use_idf=False) 
    feat = v.fit_transform(sampled[REVIEW_H])
    X_train, X_test, train_labels, test_labels = train_test_split(feat, sampled[STAR_H], test_size=0.2, random_state=42)
    
    p = Perceptron(random_state=42, class_weight='balanced', max_iter=20, n_iter_no_change=3)
    p.fit(X_train, train_labels)
    p_pred = p.predict(X_test)
    print_report(test_labels, p_pred)

    svm = LinearSVC(penalty='l1', dual=False, random_state=42, max_iter=300)
    svm.fit(X_train, train_labels)
    svm_pred = svm.predict(X_test)
    print_report(test_labels, svm_pred)

    lr = LogisticRegression(random_state=42, max_iter=400, class_weight='balanced', solver='sag')
    lr.fit(X_train, train_labels)
    lr_pred = lr.predict(X_test)
    print_report(test_labels, lr_pred)

    nb = MultinomialNB(alpha=1, fit_prior=False)
    nb.fit(X_train, train_labels)
    nb_pred = nb.predict(X_test)
    print_report(test_labels, nb_pred)

if __name__ == "__main__": 
    main()