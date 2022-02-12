#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 08:25:58 2021

@author: r21sscott
exploring the possibilty of using topic modelling for theme analysis
followed along with this article: https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/

edit 8/6/2021:
    basic LDA topic model
    realized unsupervised learning probably was not the best approach
    could be explored in the future though
"""
#for preprocessing
import nltk
from nltk.probability import FreqDist
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
#importing Gensim
import gensim
from gensim import corpora

def main():
    '''Prepare documents'''
    file = open('/mnt/linuxlab/home/r21sscott/REU_Project/FEDERALIST No. 42.txt', 'r') #path depends on user
    rawtxt = file.read()
    file.close()
    corpus = rawtxt.split('\n\n')
    doc_clean = [clean(doc).split() for doc in corpus]
    doc_clean = [removeFreq(doc, rawtxt).split() for doc in doc_clean]
    doc_clean = [ele for ele in doc_clean if ele != []] #get rid of empty sets
    #print(doc_clean)
    '''#preparing document-term matrix
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    #creating the object for LDA model using gensim library
    lda = gensim.models.ldamodel.LdaModel
    #running and training LDA model on the document term matrix
    ldamodel = lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)
    #results
    print(ldamodel.print_topics(num_topics=5, num_words=5))'''
    
    

def clean(doc):
    '''cleaning and preprocessing'''
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    num_free = ''.join(c for c in punc_free if not c.isdigit())
    small_free = " ".join([i for i in num_free.split() if len(i) > 4]) #form that German paper- words less than 3 noramlly have no effect on the text
    normalized = " ".join(lemma.lemmatize(word) for word in small_free.split())
    return normalized

def removeFreq(doc, corpus):
    #removes the least and most frequent words to see if that makes the topics/themes more accurate
    tokens = word_tokenize(corpus)
    fdist = FreqDist(tokens)
    least_freq = " ".join([i for i in doc if fdist[i] > 3])
    return least_freq
    

    
    
    

main()
