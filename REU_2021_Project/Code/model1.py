#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:39:08 2021

@author: r21sscott
create and test supervised topic model to identify themes
following this article: https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e
but can and will try other models later 

edit 8/9/2021:
    this file was mainly used to add documents to the training data 
    and to look at EDA (specifcally class distribution).
    Model part was not really used and does not work perfectly (first attempt)
    the preprocessing however was useful and employed in the rest of the models.
"""
#text preprocessing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import random
from random import seed
from random import randint
#for label ecoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
#import gensim
#from gensim.models import Word2Vec #Word2Vec is mostly used for huge datasets

def main():
    #randomPapers()  
    #addTrainData()  
    train_set = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    print(train_set.shape)
    #print(train_set)
    eda(train_set)
    
    #only really used to add and explore data
    #uncomment to run model I don't think this really works
    '''train_set['clean_document'] = train_set['Document'].apply(lambda x: finalPreprocess(x))
    #label encoding themes
    le = LabelEncoder()
    train_set['Theme_N'] = le.fit_transform(train_set['Theme'])  
    print(train_set['Theme_N'])
    #splitting the training dataset into train and test
    x_train, x_val, y_train, y_val = train_test_split(train_set['clean_document'], train_set['Theme_N'], test_size=0.2, shuffle = True)
    #TF-IDF Vectorization
    #convert x_train to vector- fit and transform
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    x_train_vectors_tfidf = tfidf_vectorizer.fit_transform(x_train)
    #only transform x_test
    x_val_vectors_tfidf = tfidf_vectorizer.transform(x_val)'''
    #building logic Regression model
    '''#fitting the classification model
    lr_tfidf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
    lr_tfidf.fit(x_train_vectors_tfidf, y_train) #model
    #Predict y value for test dataset
    y_predict = lr_tfidf.predict(x_val_vectors_tfidf)
    y_prob = lr_tfidf.predict_proba(x_val_vectors_tfidf)[:,1]
    print(classification_report(y_val, y_predict))
    print('Confusion Matrix:',confusion_matrix(y_val, y_predict))
    roc_auc = multiclassrocauc(y_val, y_predict, average='macro')
    print(roc_auc)
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    sns.heatmap(confusion_matrix(y_val, y_predict), annot=True)'''

def multiclassrocauc(y_val, y_predict, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_val)
    y_val = lb.transform(y_val)
    y_predict = lb.transform(y_predict)
    return roc_auc_score(y_val, y_predict, average=average)
    
    
    
    

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

#used to find a random paper- ended up not using
def randomPapers():
    '''this was the process of assigning which papers to train, validation, or test sets
    order went l (train) first, y (validation), p (test'''
    '''seed(1)
    y = [6, 12, 17, 21, 22, 23, 26, 40, 46, 47, 48, 66, 67, 74, 75, 79, 85]
    l = [1, 2, 3, 4, 5, 9, 13, 14, 16, 18, 24, 25, 27, 28, 29, 30, 32, 33, 35, 37, 38, 39, 41, 42, 43, 45, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 68, 70, 71, 72, 73, 76, 78, 81, 83, 84]
    p = []
    z = 1
    for x in range(700):
        value = randint(1, 85)
        if value not in l:
            if value not in y:
                if value not in p:
                    if z < 18:
                        p.append(value)
                        z = z + 1
    p = sorted(p)
    print(p)
    print(len(p))'''
    #print(randint(1, 85))

def addTrainData():
    '''takes raw text file and adds it to training dataset
    text broken into sentence or phrase chunks'''
    #filepath depends on user; change to specify what what you want
    filex = open('/mnt/linuxlab/home/r21sscott/REU_Project/Raw_Data_Essays/FEDERALIST No. 57.txt', 'r') 
    rawtxt = filex.read()
    filex.close()
    #trigrams(rawtxt)
    #make rows and columns for csv file
    rawtxt = rawtxt.replace(';', '.')
    rawtxt = rawtxt.replace(':', '.')
    rawtxt = rawtxt.replace('!', '.')
    rawtxt = rawtxt.replace('?', '.')
    txt = rawtxt.split('.')
    for x in range(len(txt)):
        txt[x] = txt[x].replace('\n', ' ')
        txt[x] = txt[x].strip()
    #make new and apoend to csv file
    with open('/mnt/linuxlab/home/r21sscott/REU_Project/trainData', 'a') as file:
        writer = csv.writer(file)
        #writer.writerow(['Theme', 'Document']) #uncomment if new file
        for x in range(len(txt)):
            writer.writerow(['n/a', txt[x]])
        file.close()
        
def eda(train_set):
    '''Exploratory Data Analysis (EDA)'''
    #num_words = [len(s.split()) for s in train_set['Document']]
    #print(np.median(num_words))
    
    #class distribution
    x = train_set['Theme'].value_counts()
    print(x)
    #sns.barplot(x.index,x)
    
    #missing values
    #train_set.isna().sum()
    #word count
    '''train_set['word_count'] = train_set['Document'].apply(lambda x: len(str(x).split()))
    print('mean word count for none:' +str(train_set[train_set['Theme']=='none']['word_count'].mean()))
    print('mean word count for weakness of articles of Confederation:' +str(train_set[train_set['Theme']=='weakness of articles of Confederation']['word_count'].mean()))
    print('mean word count for legislative:' +str(train_set[train_set['Theme']=='legislative']['word_count'].mean()))
    print('mean word count for executive:' +str(train_set[train_set['Theme']=='executive']['word_count'].mean()))
    print('mean word count for interest-group theory:' +str(train_set[train_set['Theme']=='interest-group theory']['word_count'].mean()))
    print('mean word count for republican principles:' +str(train_set[train_set['Theme']=='republican principles']['word_count'].mean()))
    print('mean word count for strong federal government:' +str(train_set[train_set['Theme']=='strong federal government']['word_count'].mean()))
    print('mean word count for federalism:' +str(train_set[train_set['Theme']=='federalism']['word_count'].mean()))
    print('mean word count for checks and balances:' +str(train_set[train_set['Theme']=='checks and balances']['word_count'].mean()))
    print('mean word count for nationalist outlook:' +str(train_set[train_set['Theme']=='nationalist outlook']['word_count'].mean()))
    print('mean word count for natural rights:' +str(train_set[train_set['Theme']=='natural rights']['word_count'].mean()))
    print('mean word count for human nature:' +str(train_set[train_set['Theme']=='human nature']['word_count'].mean()))
    print('mean word count for union:' +str(train_set[train_set['Theme']=='union']['word_count'].mean()))
    print('mean word count for separation of powers:' +str(train_set[train_set['Theme']=='separation of powers']['word_count'].mean()))
    print('mean word count for judicial:' +str(train_set[train_set['Theme']=='judicial']['word_count'].mean())) '''
    
    #unique word count- could repeat for all 
    '''train_set['unique_word_count'] = train_set['Document'].apply(lambda x: len(set(str(x).split())))
    print('mean unique word count for none:' +str(train_set[train_set['Theme']=='none']['unique_word_count'].mean()))
    print('mean unique word count for weakness of articles of Confederation:' +str(train_set[train_set['Theme']=='weakness of articles of Confederation']['unique_word_count'].mean()))
    print('mean unique word count for legislative:' +str(train_set[train_set['Theme']=='legislative']['unique_word_count'].mean()))
    print('mean unique word count for executive:' +str(train_set[train_set['Theme']=='executive']['unique_word_count'].mean()))
    print('mean unique word count for interest-group theory:' +str(train_set[train_set['Theme']=='interest-group theory']['unique_word_count'].mean()))
    print('mean unique word count for republican principles:' +str(train_set[train_set['Theme']=='republican principles']['unique_word_count'].mean()))
    print('mean unique word count for strong federal government:' +str(train_set[train_set['Theme']=='strong federal government']['unique_word_count'].mean()))
    print('mean unique word count for federalism:' +str(train_set[train_set['Theme']=='federalism']['unique_word_count'].mean()))
    print('mean unique word count for checks and balances:' +str(train_set[train_set['Theme']=='checks and balances']['unique_word_count'].mean()))
    print('mean unique word count for nationalist outlook:' +str(train_set[train_set['Theme']=='nationalist outlook']['unique_word_count'].mean()))
    print('mean unique word count for natural rights:' +str(train_set[train_set['Theme']=='natural rights']['unique_word_count'].mean()))
    print('mean unique word count for human nature:' +str(train_set[train_set['Theme']=='human nature']['unique_word_count'].mean()))
    print('mean unique word count for union:' +str(train_set[train_set['Theme']=='union']['unique_word_count'].mean()))
    print('mean unique word count for separation of powers:' +str(train_set[train_set['Theme']=='separation of powers']['unique_word_count'].mean()))
    print('mean unique word count for judicial:' +str(train_set[train_set['Theme']=='judicial']['unique_word_count'].mean()))'''
        
main()
        
    
    
