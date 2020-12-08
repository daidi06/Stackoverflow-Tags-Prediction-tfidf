#! /usr/bin/env python3
# coding: utf-8

import os
import string
import numpy as np
import pandas as pd
import en_core_web_sm

from joblib import load
from gensim.utils import SaveLoad, tokenize

#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


# model retenu : regression logistique sur features tf-idf

def main():
    pass

class PredictTags():
    def __init__(self):
        
        self.df_tags = pd.Series(load('log_reg_tfidf_tags.joblib'))
        self.mystopwords = set(stopwords) | set(string.punctuation)
        self.bigram_model = SaveLoad.load('bigram_model')
        self.trigram_model = SaveLoad.load('trigram_model')
        self.nlp = en_core_web_sm.load()
        self.vectorizer = load('vectorizer_tfidf.joblib')
        self.log_reg_model = load('log_reg_tfidf.joblib')
        
    def text2tokens(self, text):
        # tokenize, normalize and filter number
        tokens = list(tokenize(text, lowercase=True))
        # stop words
        tokens = [x for x in tokens if x not in self.mystopwords]
        return tokens

    def make_bigrams(self, tokens):
        return self.bigram_model[tokens]
	
    def make_trigrams(self, tokens):
        return self.trigram_model[tokens]

    def lemmatize(self, tokens):
        text = ' '.join(tokens)
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc]
        return tokens

    def vectorize(self, tokens):
        post_tokens = ' '.join(tokens)
        return self.vectorizer.transform([post_tokens])

    def text2tfidf(self, text):
        tokens = self.text2tokens(text)
        tokens = self.make_bigrams(tokens)
        tokens = self.make_trigrams(tokens)
        tokens = self.lemmatize(tokens)
        tfidf = self.vectorize(tokens)
        return tfidf

    def text2tags(self, text):
        tfidf = self.text2tfidf(text)
        prediction = self.log_reg_model.predict_proba(tfidf)
        p = np.mean(prediction[0])+2*np.std(prediction[0])
        index = np.where(prediction[0] > p)
        tags = self.df_tags.iloc[index]
        return tags

if __name__ == '__main__':
    # appel du script
    main()
else:
    # appel du module
    pass
