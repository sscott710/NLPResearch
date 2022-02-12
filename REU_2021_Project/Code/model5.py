#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:44:38 2021

@author: r21sscott
https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35 
model of above but with preprocessing of model 3

edit 8/9/2021:
    neural network model 3
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
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, LeakyReLU, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import pandas as pd
import tensorflow as tf
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
STOPWORDS = set(stopwords.words('english'))

#hyperparameters
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

def main():
    #read in traning data
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    
    #preprocess documents
    df['clean_document'] = df['Document'].apply(lambda x: finalPreprocess(x))
    
    #max number of words to be used (most frequent)
    max_nb_words = 50000
    
    #max number of words per document
    max_sequence_length = 250
    
    #fixed
    embedding_dim = 100
    
    #train_test_split method
    #make data usable for nn
    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['clean_document'].values)
    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))
    #truncate and pad input sequnces so all same length
    X = tokenizer.texts_to_sequences(df['clean_document'].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    #print('Shape of data tensor:', X.shape)
    
    #convert categorical labels to numbers
    Y = df['Theme'].values
    #print('shape of label tensor:', Y.shape)
    
    #oversampling
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    
    #train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)
    #use below if oversmapling; if not comment out
    X_train, Y_train = smote.fit_resample(X_train, Y_train)
    Y_train = pd.get_dummies(Y_train)
    Y_test = pd.get_dummies(Y_test)
    
    #model 
    filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model3.hdf5'
    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(15, activation='softmax')
    ])
    #print(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_epochs = 10
    checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, mode='max')
    history = model.fit(X_train, Y_train, epochs=num_epochs,validation_split=0.10, verbose=2, callbacks=[checkpoint])
    
    
    accr = model.evaluate(X_test, Y_test)
    print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    
    #using a loaded model
    '''#evaulate best model
    b_model = load_model(filepath)
    metrics = b_model.evaluate(X_test, Y_test)
    print("{}: {}".format(b_model.metrics_names[0], metrics[0]))
    print("{}: {}".format(b_model.metrics_names[1], metrics[1]))'''

    #using k-fold instead of train-test-split
    '''#using k-fold instead of train-test-split
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    fold_no = 1
    #max number of words to be used (most frequent)
    max_nb_words = 50000
    #max number of words per document
    max_sequence_length = 250
    #fixed
    embedding_dim = 100
    cvscores = []
    cvloss = []
    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['clean_document'].values)
    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))
    #truncate and pad input sequnces so all same length
    X = tokenizer.texts_to_sequences(df['clean_document'].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    Y = df['Theme'].values
    filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model3.hdf5'
    #oversampling
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    #X_ROS, Y_ROS = ros.fit_resample(X, Y)
    for train, test in kf.split(X, Y):
        x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        x_train, y_train = smote.fit_resample(x_train, y_train)'''
        
    '''y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)
    
        #model
        model = tf.keras.Sequential([
            # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Flatten(),
            # use ReLU in place of tanh function since they are very good alternatives of each other.
            tf.keras.layers.Dense(32, activation='relu'), #OG embedding_dim, ...
            # Add a Dense layer with 6 units and softmax activation.
            # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
            tf.keras.layers.Dense(15, activation='softmax')
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        num_epochs = 10
        checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, mode='max')
        history = model.fit(x_train, y_train, epochs=num_epochs,validation_split=0.10, verbose=2, callbacks=[checkpoint])
        
        scores = model.evaluate(x_test, y_test)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))
        cvscores.append(scores[1] * 100)
        cvloss.append(scores[0])
    print("mean accuracy and loss:")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))'''

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
    
