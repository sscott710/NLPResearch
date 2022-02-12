#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:23:10 2021

@author: r21sscott
https://shrikar.com/deep-learning-with-keras-and-python-for-multiclass-classification/
another model for text classification
edit 8/9/2021:
    this model is complete under model4n.py
    This is unfinished code.
"""
import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

def main():
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    
    #convert tags to integers
    df['target'] = df.Theme.astype('category').cat.codes
    
    #look at word distribution
    df['num_words'] = df.Document.apply(lambda x : len(x.split()))
    bins=[0,50,75, np.inf]
    df['bins']=pd.cut(df.num_words, bins=[0,100,300,500,800, np.inf], labels=['0-100', '100-300', '300-500','500-800' ,'>800'])
    word_distribution = df.groupby('bins').size().reset_index().rename(columns={0:'counts'})
    #print(word_distribution.head())
    
    #set num of classes and target variable
    num_class = len(np.unique(df.Theme.values))
    y = df['target'].values
    

    #tokenize input
    MAX_LENGTH = 250
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.Document.values)
    doc_seq = tokenizer.texts_to_sequences(df.Document.values)
    doc_seq_padded = pad_sequences(doc_seq, maxlen=MAX_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(doc_seq_padded, y, test_size=0.05)
    vocab_size = len(tokenizer.word_index) +1
    
    #deep learning model: simple5 epochs got 0.5625; with 10 epochs got 0.625
    inputs = Input(shape=(MAX_LENGTH, ))
    embedding_layer = Embedding(vocab_size,
                                128,
                                input_length=MAX_LENGTH)(inputs)
    x = Flatten()(embedding_layer)
    x = Dense(32, activation='relu')(x)

    predictions = Dense(num_class, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    #model.summary()
    filepath="weights-simple.hdf5"
    checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
                        shuffle=True, epochs=10, callbacks=[checkpointer])
    
    predicted = model.predict(X_test)
    predicted = np.argmax(predicted, axis=1)
    print(accuracy_score(y_test, predicted))
    
    '''#RNN- accuracy = 0.46875; suppose to get better
    inputs = Input(shape=(MAX_LENGTH, ))
    embedding_layer = Embedding(vocab_size,
                                128,
                                input_length=MAX_LENGTH)(inputs)

    x = LSTM(64)(embedding_layer)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(num_class, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    #print(model.summary())
    
    filepath="weights.hdf5"
    checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
                        shuffle=True, epochs=10, callbacks=[checkpointer])
    
    model.load_weights('weights.hdf5')
    predicted = model.predict(X_test)
    predicted = np.argmax(predicted, axis=1)
    print(accuracy_score(y_test, predicted))'''
    


    
    
    
    

    
    
    
    
    
    
    
                
main()
