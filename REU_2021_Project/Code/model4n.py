from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:23:10 2021

@author: r21sscott
https://shrikar.com/deep-learning-with-keras-and-python-for-multiclass-classification/
another model for text classification
edit 8/9/2021:
    neural network model 2
    pay attention to commnted/uncommented approches
"""
from numpy.random import seed
seed(0)
import tensorflow
tensorflow.random.set_seed(0)
import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
#text preprocessing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from IPython.core.display import display, HTML
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
display(HTML("<style>.container { width:100% !important; }</style>"))

def main():
    #read in training data
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    
    #proprocessing
    df['clean_document'] = df['Document'].apply(lambda x: finalPreprocess(x))
    
    #convert tags to integers
    df['target'] = df.Theme.astype('category').cat.codes
    
    #eda
    '''#look at word distribution
    df['num_words'] = df.clean_document.apply(lambda x : len(x.split()))
    bins=[0,50,75, np.inf]
    df['bins']=pd.cut(df.num_words, bins=[0,100,300,500,800, np.inf], labels=['0-100', '100-300', '300-500','500-800' ,'>800'])
    word_distribution = df.groupby('bins').size().reset_index().rename(columns={0:'counts'})
    #print(word_distribution.head())'''
    
    #using train_test_split to evaluate model
    #simply comment/uncomment to use desired approach
    '''#set num of classes and target variable
    num_class = len(np.unique(df.Theme.values))
    y = df['target'].values
    
    #oversampling
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)

    #tokenize input
    MAX_LENGTH = 250
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.clean_document.values)
    doc_seq = tokenizer.texts_to_sequences(df.clean_document.values)
    doc_seq_padded = pad_sequences(doc_seq, maxlen=MAX_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(doc_seq_padded, y, test_size=0.2)
    X_train, y_train = ros.fit_resample(X_train, y_train)

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
    filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model2.hdf5'
    checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
                        shuffle=True, epochs=10, callbacks=[checkpointer])
    
    predicted = model.predict(X_test)
    predicted = np.argmax(predicted, axis=1)
    print(accuracy_score(y_test, predicted))'''
    
    #using k-fold instead of train-test-split
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    fold_no = 1
    cvscores = []
    cvloss = []
    #X = df['clean_document'].values 
    df['target'] = df.Theme.astype('category').cat.codes
    Y = df['target'].values
    MAX_LENGTH = 250
    #num_class = len(np.unique(df.Theme.values))
    filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model2.hdf5'
    #make data usable for nn
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.clean_document.values)
    doc_seq = tokenizer.texts_to_sequences(df.clean_document.values)
    X = pad_sequences(doc_seq, maxlen=MAX_LENGTH)
    vocab_size = len(tokenizer.word_index) +1
    #oversampling
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    #X_ROS, Y_ROS = ros.fit_resample(X, Y) #wrong
    for train, test in kf.split(X, Y):
        x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        x_train, y_train = smote.fit_resample(x_train, y_train)
          
        #model
        inputs = Input(shape=(MAX_LENGTH, ))
        embedding_layer = Embedding(vocab_size,
                                    128,
                                input_length=MAX_LENGTH)(inputs) #OG 128
        x = Flatten()(embedding_layer)
        x = Dense(96, activation='relu')(x) #OG 64

        predictions = Dense(15, activation='softmax')(x)
        model = Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])
        
        filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model2.hdf5' #depends on user
        checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        history = model.fit([x_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
                            shuffle=True, epochs=10, callbacks=[checkpointer])
        
        #scores = model.evaluate(doc_seq_padded_te, y_test)
        
        predicted = model.predict(x_test)
        predicted = np.argmax(predicted, axis=1)
        acc = accuracy_score(y_test, predicted)
        print(acc)
        cvscores.append(acc * 100)
        loss = hamming_loss(y_test,predicted)
        print(loss)
        cvloss.append(loss)
        
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        #print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
        #cvscores.append(scores[1] * 100)
        #cvloss.append(scores[0])
    print("mean accuracy and loss:")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))

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