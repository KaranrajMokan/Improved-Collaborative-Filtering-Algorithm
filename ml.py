#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:58:16 2020

@author: karanrajmokan
"""

import pandas as pd
pd.options.mode.chained_assignment = None 


import gensim
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem import PorterStemmer
import numpy as np
np.random.seed(2018)



def lemmatize_stemming(text):
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result




data = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
data_text = data['reviews.text']
data_text['index'] = data_text.index
documents = data_text

doc_sample = documents[0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

print("\n\n")
processed_docs = documents.astype(str).map(preprocess)
print(processed_docs[:10])

print("\n\n")
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

print("\n\n")
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
print(dictionary)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4310]
