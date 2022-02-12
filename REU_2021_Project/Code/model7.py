from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:31:35 2021

@author: r21sscott
from https://blog.mimacom.com/text-classification/ 
edit 8/9/2021:
    neural network model 7
"""
from numpy.random import seed
seed(0)
import tensorflow
tensorflow.random.set_seed(0)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
#text preprocessing
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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Activation, Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE

def main():
    #read in training data
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    #preprocess data
    df['clean_document'] = df['Document'].apply(lambda x: finalPreprocess(x))
    
    #max number of words to be used (most frequent)
    max_nb_words = 50000
    #max number of words per document
    max_sequence_length = 250
    #fixed
    embedding_dim = 100
    
    #model 7- train_test_split method
    #prepare data for nn
    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['clean_document'].values)
    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))
    #truncate and pad input sequnces so all same length
    X = tokenizer.texts_to_sequences(df['clean_document'].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    
    #convert categorical labels to numbers
    Y = df['Theme'].values
    
    #oversampling
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #oversampling; comment out if using OG data
    x_train, y_train = smote.fit_resample(x_train, y_train)
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    filter_length = 300
    model = Sequential()
    model.add(Embedding(5000, 20, input_length=180))
    model.add(Dropout(0.1))
    model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(15))
    model.add(Activation('sigmoid'))

    filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model6.hdf5'
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    callbacks = [
        ReduceLROnPlateau(),
        EarlyStopping(patience=4),
        ModelCheckpoint(filepath=filepath, save_best_only=True)
    ]

    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=callbacks)
    accr = model.evaluate(x_test, y_test)
    print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    '''#evaulate best model
    simple_model = keras.models.load_model(filepath)
    metrics = simple_model.evaluate(x_test, y_test)
    print("{}: {}".format(simple_model.metrics_names[0], metrics[0]))
    print("{}: {}".format(simple_model.metrics_names[1], metrics[1]))'''
    
    #Model 7- using k-fold instead of train-test-split
    '''
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    fold_no = 1
    cvscores = []
    cvloss = []
    #X = df['clean_document'].values 
    #Y = df['Theme'].values
    filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model6.hdf5'
    filter_length = 300 #OG 300
    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['clean_document'].values)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) +1
    X = tokenizer.texts_to_sequences(df['clean_document'].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    y = df['Theme'].values
    #X_ROS, Y_ROS = ros.fit_resample(X, y) #wrong
    for train, test in kf.split(X, y):
        x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]
        #oversampling
        x_train, y_train = smote.fit_resample(x_train, y_train)'''

        
    '''y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)
    
        #model
        model = Sequential()
        #input layers
        model.add(Embedding(vocab_size, 128, input_length=250)) #middle num OG 50
        #hidden layers
        model.add(Dropout(0.5)) #OG 0.3
        model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1)) #Og middle number is 3
        model.add(GlobalMaxPool1D())
        model.add(Flatten())
        
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu')) #OG 64
        model.add(Dropout(0.5))    
        
        #ouput layers
        model.add(Dense(15))
        model.add(Activation('relu')) #could change to softmax
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy']) #maybe change to categorical_cross
        
        callbacks = [
            ReduceLROnPlateau(),
            EarlyStopping(patience=4),
            ModelCheckpoint(filepath=filepath, save_best_only=True)
            ]
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=callbacks) #OG 20 epochs; batch size 32
        
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
