#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 13:08:58 2021

last edited 8/6/2021
@author: Savannah Scott
Prepare essays to be analyzed by model
-breaks down essays into documents
-adds docuements to csv file
"""
import csv
import pandas as pd
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
def main():
    addAnaData()
    #the commented out section was suppose to import and use the models in this file but
    #it did not work; to actually predict theme go to model2.py or any other model file
    '''df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/analyze55')
    df['clean_document'] = df['Document'].apply(lambda x: finalPreprocess(x))
    #vectorization
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 2),
                            stop_words='english')
    fitted_vectorizer = tfidf.fit(df.clean_document)
    tfidf_vectorizer_vectors = fitted_vectorizer.transform(df.clean_document)
    test1 = 'FEDERALIST No 55'
    #features = tfidf.fit_transform(df.clean_document).toarray()
    #print(len(features))
    filename4 = '/mnt/linuxlab/home/r21sscott/REU_Project/Class_Models/randForest.sav'
    loaded_model = pickle.load(open(filename4, 'rb'))
    #y_pred = loaded_model.predict(features)
    print(loaded_model.predict((fitted_vectorizer.transform([test1]))))'''
    
def addAnaData():
    '''takes raw text file and adds it to analytical dataset- depends on model
    text broken into sentence or phrase chunks'''
    filex = open('/mnt/linuxlab/home/r21sscott/REU_Project/FEDERALIST No. 55.txt', 'r') #file path depends on user
    #reads in file
    rawtxt = filex.read()
    filex.close()
    #divides file into paragraphs
    pars = rawtxt.split('\n\n')
    #replaces all puncutuation that makes up phrases with a '.'; no newlines
    for x in range(len(pars)):
        pars[x] = pars[x].replace(';', '.')
        pars[x] = pars[x].replace(':', '.')
        pars[x] = pars[x].replace('!', '.')
        pars[x] = pars[x].replace('?', '.')
        pars[x] = pars[x].replace('\n', ' ')      
    txt = [] #list of documents
    #this loop divides the paragraphs into sentences or phrases
    for x in range(len(pars)):
        if x < 5:
            pars[x] = pars[x].replace('.', '')
            txt.append(pars[x])
        else:
            temp = pars[x].split('.')
            for y in range(len(temp)):
                temp[y] = temp[y].strip()
                txt.append(temp[y])
    txt = list(filter(None,txt))  #get ride of None
    #make new and apoend to csv file
    with open('/mnt/linuxlab/home/r21sscott/REU_Project/analyzeLinearSVC', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(['Document']) #not needed if adding to existing file
        for x in range(len(txt)):
            writer.writerow([txt[x]])
        file.close()

'''start preprocessing'''
'''common text preprocessing'''
def preprocess(text):
    text = text.lower() #lowercase text
    text = text.strip() #remove leading/trailing whitespace
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text) #replaces punctuation with space
    text = re.sub(r'\[0-9]*\]',' ', text) #removes digits
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) #matches any digit from 0 to 100000..., \D matches non-digits
    return text

'''stopword removal'''
def stopword(string):
    a=[i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a) 

'''stemming'''
'''initialize the stemmer'''
snow = SnowballStemmer('english')
def stemming(string):
    a = [snow.stem(i) for i in word_tokenize(string)]
    return " ".join(a)

'''lemmatization'''
'''initialize the lemmatizer'''
wl = WordNetLemmatizer()
def get_wordnet_pos(tag):
     '''Map NLTK position tags'''
     if tag.startswith('J'):
         return wordnet.ADJ
     elif tag.startswith('V'):
         return wordnet.VERB
     elif tag.startswith('N'):
         return wordnet.NOUN
     elif tag.startswith('R'):
         return wordnet.ADV
     else:
         return wordnet.NOUN

def lemmatizer(string):
    '''tokenize the sentence'''
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) #get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # map the position tag and lemmatize the word/token
    return " ".join(a)

def finalPreprocess(string):
    '''calls all preprocessing into one'''
    return lemmatizer(stopword(preprocess(string)))
'''end preprocessing'''    
main()