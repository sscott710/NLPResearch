#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:37:51 2021

@author: r21sscott
Based on for model: https://sabber.medium.com/classifying-yelp-review-comments-using-cnn-lstm-and-visualize-word-embeddings-part-2-ca137a42a97d
glove model: https://sabber.medium.com/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
edit 8/9/2021:
    neural network models 4 and 5
    4- no pretrained word embeddings
    5- glove word embeddings
"""
from numpy.random import seed
seed(0)
import tensorflow
tensorflow.random.set_seed(0)
#text preprocessing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE

vocabulary_size = 20000
def main():
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    df['clean_document'] = df['Document'].apply(lambda x: finalPreprocess(x))
    
    #encode labels
    Y = df['Theme'].values
    
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(df['clean_document'])
    sequences = tokenizer.texts_to_sequences(df['clean_document'])
    data = pad_sequences(sequences, maxlen=50)
    
    #filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model4.hdf5' 
    
    #oversampling
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    
    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2)
    #oversampling; comment out if using OG dataset
    X_train, y_train = smote.fit_resample(X_train, y_train)
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    #model 4- train_test_split method
    '''model_conv = create_conv_model()
    checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, mode='max')
    model_conv.fit(X_train, y_train, validation_split=0.2, epochs = 10, batch_size=64, callbacks=[checkpoint])
    
    accr = model_conv.evaluate(X_test, y_test)
    print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0], accr[1]))'''
    
    #mdoel 5- train_test_split method
    #Creating a model using Glove for word embeddings
    embeddings_index = dict()
    f = open('/mnt/linuxlab/home/r21sscott/REU_Project/glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    embedding_matrix = np.zeros((vocabulary_size, 100)) #Og 100
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    
    ## create model
    model_glove = Sequential()
    model_glove.add(Embedding(vocabulary_size, 100, input_length=50, weights=[embedding_matrix], trainable=False))
    model_glove.add(Dropout(0.2))
    model_glove.add(Conv1D(64, 5, activation='relu'))
    model_glove.add(MaxPooling1D(pool_size=4))
    model_glove.add(LSTM(100))
    model_glove.add(Dense(15, activation='softmax'))
    model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    filepath2='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model5.hdf5'
    checkpoint = ModelCheckpoint(filepath=filepath2, verbose=1, save_best_only=True, mode='max')
    
    ## Fit train data 
    model_glove.fit(X_train, y_train, validation_split=0.2, epochs = 10, batch_size=64, callbacks=[checkpoint])
    
    #evualuate model
    accr = model_glove.evaluate(X_test, y_test)
    print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    
    '''#evaulate best model
    b_model = load_model(filepath2)
    metrics = b_model.evaluate(X_test, y_test)
    print("evaultion of best model")
    print("{}: {}".format(b_model.metrics_names[0], metrics[0]))
    print("{}: {}".format(b_model.metrics_names[1], metrics[1]))'''
    
    #Model 4- using k-fold instead of train-test-split
    '''
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    fold_no = 1
    cvscores = []
    cvloss = []
    #X = df['clean_document'].values 
    Y = df['Theme'].values
    filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model4.hdf5'
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(df['clean_document'])
    sequences = tokenizer.texts_to_sequences(df['clean_document'])
    X = pad_sequences(sequences, maxlen=50)
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    y = df['Theme'].values
    #X_ROS, Y_ROS = ros.fit_resample(X, y) #wromg
    for train, test in kf.split(X, y):
        x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        #oversampling
        x_train, y_train = smote.fit_resample(x_train, y_train)'''
        
        
    '''y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)
    
        #model
        model_conv = create_conv_model()
        checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, mode='max')
        model_conv.fit(x_train, y_train, validation_split=0.2, epochs = 10, batch_size=64, callbacks=[checkpoint])
        
        scores = model_conv.evaluate(x_test, y_test)
        print("%s: %.2f%%" % (model_conv.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (model_conv.metrics_names[0], scores[0]))
        cvscores.append(scores[1] * 100)
        cvloss.append(scores[0])
    print("mean accuracy and loss:")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))'''
    
    #Model5- using k-fold instead of train-test-split
    '''
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    fold_no = 1
    cvscores = []
    cvloss = []
    #X = df['clean_document'].values 
    Y = df['Theme'].values
    filepath2='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model5.hdf5'
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(df['clean_document'])
    sequences = tokenizer.texts_to_sequences(df['clean_document'])
    X = pad_sequences(sequences, maxlen=50)
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    #y = df['Theme'].values
    #X_ROS, Y_ROS = ros.fit_resample(X, y)
    for train, test in kf.split(X, Y):
        x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        #oversampling
        x_train, y_train = smote.fit_resample(x_train, y_train)'''
        

       
    '''y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)
    
        #model
        model_glove = Sequential()
        model_glove.add(Embedding(vocabulary_size, 100, input_length=250, weights=[embedding_matrix], trainable=False))
        model_glove.add(Dropout(0.4)) #0.3
        model_glove.add(Conv1D(64, 4, activation='relu')) #OG 3
        model_glove.add(MaxPooling1D(pool_size=4))
        model_glove.add(LSTM(32))
        model_glove.add(Dropout(0.4))
        model_glove.add(Dense(32, activation='relu')) #64
        model_glove.add(Dropout(0.4))
        model_glove.add(Dense(15, activation='softmax'))
        model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint(filepath=filepath2, verbose=1, save_best_only=True, mode='max')
        history = model_glove.fit(x_train, y_train, validation_split=0.2, epochs = 10, batch_size=64, callbacks=[checkpoint])
        
        scores = model_glove.evaluate(x_test, y_test)
        print("%s: %.2f%%" % (model_glove.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (model_glove.metrics_names[0], scores[0]))
        cvscores.append(scores[1] * 100)
        cvloss.append(scores[0])
    print("mean accuracy and loss:")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))'''

def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, 128, input_length=250)) #OG middle num 100
    model_conv.add(Dropout(0.3)) #0.3
    model_conv.add(Conv1D(64, 4, activation='relu')) #og 64, 4 - 100
    model_conv.add(MaxPooling1D(pool_size=4)) #og 4
    model_conv.add(LSTM(32)) #OG 100
    model_conv.add(Dropout(0.3))
    model_conv.add(Dense(32, activation='relu')) #64
    model_conv.add(Dropout(0.3))
    model_conv.add(Dense(15, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

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
