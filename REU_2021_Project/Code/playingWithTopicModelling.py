#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:21:18 2021

exploring topic modeling with a single fed paper

@author: r21sscott

edit 8/6/2021:
    this is like half finished;
    DOES NOT WORK AND IS NOT COMPLETE
    could be explored in the future
"""
import nltk, re, pprint
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
#from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import collections
'''
Notes- following along with 'Topic Moedlling:Going Beyond Token Outputs'
by Lowri williams
ignoring lemmatisation'''
def main():
    '''open file and store raw data into variable'''
    file = open('/mnt/linuxlab/home/r21sscott/REU_Project/FEDERALIST No. 42.txt', 'r') #chose to first work with No.42 because it is credited as the most cited Fed Paper
    rawtxt = file.read()
    file.close()
    tokens = tokenize(rawtxt)
    filtered_tokens =cleanUpTxt(tokens)
    print(filtered_tokens)

    
    
    

def tokenize(rawtxt):
    '''split into paragraphs'''
    txt = rawtxt.split('\n\n')
    '''tokenize text'''
    tokens= []
    for x in range(len(txt)):
        tokens.append(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", txt[x]))
    return tokens

def cleanUpTxt(tokens):
    '''make all words lowercase and remove stop words'''
    for w in range(len(tokens)):
        for y in range(len(tokens[w])):
            tokens[w][y] = tokens[w][y].lower()
    stop_words = stopwords.words('english')
    stop_pun = [',', ';', '.', '?', '!', ':', '"']
    '''this is not done'''
    filtered_tokens=[]
    for w in tokens:
        if w not in stop_words:
            if w not in stop_pun:
                filtered_tokens.append(w)
    return filtered_tokens

main()