#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:08:26 2021

@author: r21sscott
https://medium.com/swlh/step-by-step-building-a-multi-class-text-classification-model-with-keras-f78a0209a61a
Doesn't work--can come back to it if need be'

edit 8/9/2021:
    model does not work; would be interesting to come back to it
    error line 94: ValueError: Shapes (None, 15) and (None, 62, 15) are incompatible
"""
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout1D,Conv1D,MaxPooling1D,GRU,BatchNormalization
from tensorflow.keras.layers import Input,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import spacy
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras
def main():
    #load training data
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    
    #preprocess training data
    df['clean_document'] = df['Document'].apply(lambda x: finalPreprocess(x))
    
    #tokenize documents
    embedding_dim = 100
    tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['clean_document'].values)
    word_index = tokenizer.word_index
    
    X = tokenizer.texts_to_sequences(df['clean_document'].values)
    X = pad_sequences(X, maxlen=250)
    
    #soecial word embeddings
    nlp = spacy.load("en_core_web_lg")
    text_embedding = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        text_embedding[i] = nlp(word).vector
    
    #build nn model
    model = Sequential()
    model.add(Embedding(input_dim=text_embedding.shape[0],
                        output_dim=text_embedding.shape[1],
                        weights=[text_embedding],
                        trainable=False))
    model.add(SpatialDropout1D(0.5))
    model.add(Conv1D(300, kernel_size=3,
                     kernel_regularizer=regularizers.l2(0.00001),
                     padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(100, dropout=0.5, 
                                 recurrent_dropout=0.5, return_sequences=True))) #not sure about 100-random
    model.add(SpatialDropout1D(0.5))
    model.add(Conv1D(300, kernel_size=3,
                     kernel_regularizer=regularizers.l2(0.00001),
                     padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(100, dropout=0.5, 
                                 recurrent_dropout=0.5, return_sequences=True))) #not sure about 100-random
    model.add(Dense(15,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',
                   metrics=['accuracy'])
    #print(model.summary())
    
    callbacks = [
        ReduceLROnPlateau(),
        EarlyStopping(patience=4),
        ModelCheckpoint(filepath='model-simple.h5', save_best_only=True)
    ]
    
    y = pd.get_dummies(df['Theme'].values)
    #print('shape of label tensor:', y.shape)
    #y = to_categorical(y, num_classes=15)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    
    history = model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_split=0.1,
              callbacks=callbacks, shuffle=True)
    
    simple_model = keras.models.load_model('model-simple.h5')
    metrics = simple_model.evaluate(X_test, Y_test)
    print("{}: {}".format(simple_model.metrics_names[0], metrics[0]))
    print("{}: {}".format(simple_model.metrics_names[1], metrics[1]))
    

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