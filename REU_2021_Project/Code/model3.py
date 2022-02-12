#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:21:40 2021

@author: r21sscott
using neural networks for multi-class text classification
RNN with LTSM architecture
https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
edit 8/9/2021:
    neural network model 1
    most things in main to pay attention to commenting
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
from keras.models import Sequential
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
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
#import plotly.graph_objs as go
#import plotly.plotly as py
#import cufflinks
from IPython.core.interactiveshell import InteractiveShell
#import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
#from plotly.offline import iplot
#cufflinks.go_offline()
#cufflinks.set_config_file(world_readable=True, theme='pearl')
#evaluation
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import RandomOverSampler, SMOTE
#from mlxtend.evaluate import confusion_matrix
def main():
    #read in training data
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    
    #preprocess documents
    df['clean_document'] = df['Document'].apply(lambda x: finalPreprocess(x))
    
    #max number of words to be used (most frequent)
    max_nb_words = 50000
    
    #max number of words per document
    max_sequence_length = 250
    
    #fixed
    embedding_dim = 100 #OG 100
    
    #this is the train/test/split method instead of stratified k-fold
    #uncomment out to use and comment out kfold below to use
    '''tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['clean_document'].values)
    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))
    #truncate and pad input sequnces so all same length
    X = tokenizer.texts_to_sequences(df['clean_document'].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    #print('Shape of data tensor:', X.shape)
    
    #convert categorical labels to numbers
    #Y = pd.get_dummies(df['Theme'].values)
    Y = df['Theme'].values
    #df['category_id'] = pd.get_dummies(df['Theme'].values())
    #category_id_df = df[['Theme', 'category_id']].drop_duplicates()
    #print('shape of label tensor:', Y.shape)
    
   
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    
    #train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)
    X_train, Y_train = smote.fit_resample(X_train, Y_train)
    Y_train = pd.get_dummies(Y_train)
    Y_test = pd.get_dummies(Y_test)

    
    #model
    model = Sequential()
    model.add(Embedding(max_nb_words, embedding_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    #model.add(LeakyReLU(alpha=0.2)) i added
    #model.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5)))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #just loss='categorical_crossentropy' got error
    #print(model.summary())
    epochs = 10 #reset = 10
    batch_size = 64 #reset = 64
    
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.20,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    
    #Evaluation
    accr = model.evaluate(X_test, Y_test)
    print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0], accr[1]))'''
    
    #using k-fold instead of train-test-split
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    fold_no = 1
    cvscores = []
    cvloss = []
    #X = df['clean_document'].values 
    #Y = df['Theme'].values
    filepath='/mnt/linuxlab/home/r21sscott/REU_Project/NN_Models/model1.hdf5'
    #make data usable for model
    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['clean_document'].values)
    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))
    #truncate and pad input sequnces so all same length
    X = tokenizer.texts_to_sequences(df['clean_document'].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    #random oversmapling method
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    y = df['Theme'].values
    #X_ROS, Y_ROS = ros.fit_resample(X, y)
    for train, test in kf.split(X, y):
        #split data into train and test
        x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]
        #oversmaple training data; comment out if using OG dataset
        x_train, y_train = smote.fit_resample(x_train, y_train)
                
        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)
      
        #model
        model = Sequential()
        model.add(Embedding(max_nb_words, embedding_dim, input_length=x_train.shape[1]))
        model.add(SpatialDropout1D(0.3)) #OG 0.3
        model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(15, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, mode='max')
        history = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0,validation_split = 0.2, callbacks=[checkpoint])
        scores = model.evaluate(x_test, y_test)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))
        cvscores.append(scores[1] * 100)
        cvloss.append(scores[0])
    print("mean accuracy and loss:")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))
    
    #looking at confusion matrix
    #could not get to work
    '''y_pred = tf.argmax(model.predict(X_test))
    #labels cannot be one-hot encoded
    Y_rounded = tf.argmax(Y_test)
    classes = df.Theme.unique()
    
    confusion = confusion_matrix(Y_rounded, y_pred, labels=classes)
    np.set_printoptions(precision=2)
    #print('Confusion Matrix\n')
    print(confusion)
    
    def plot_confusion_matrix(cm, classes, normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
    
    plt.figure()
    #plot_confusion_matrix(confusion, classes=classes, title='Confusionn matrix')'''
    
    
    
    #visualize
    '''fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(confusion, annot=True, cmap="Blues", fmt='d',
            xticklabels=df.Theme.unique(), 
            yticklabels=df.Theme.unique())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("CONFUSION MATRIX - Model3\n", size=16);'''
    #plot_confusion_matrix()
    
    
    
    
    #this is for train_test_split method
    #do not think this works but not sure- stopped using a while ago
    '''print('\nAccuracy: {:.2f}\n'.format(accuracy_score(Y_rounded, y_pred)))
    
    print('Micro Precision: {:.2f}\n'.format(precision_score(Y_rounded, y_pred, average='micro')))
    print('Micro Recall: {:.2f}\n'.format(recall_score(Y_rounded, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(Y_rounded, y_pred, average='micro')))
    
    print('Macro Precision: {:.2f}\n'.format(precision_score(Y_rounded, y_pred, average='macro')))
    print('Macro Recall: {:.2f}\n'.format(recall_score(Y_rounded, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(Y_rounded, y_pred, average='macro')))
    
    print('Weighted Precision: {:.2f}\n'.format(precision_score(Y_rounded, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}\n'.format(recall_score(Y_rounded, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}\n'.format(f1_score(Y_rounded, y_pred, average='weighted')))
    
    print('\nClassification Report\n')
    print(classification_report(Y_rounded, y_pred, target_names=df['Theme'].unique()))'''
    

    

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