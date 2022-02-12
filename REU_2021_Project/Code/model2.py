#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:41:09 2021

@author: r21sscott
following along to a multiclass text classification example
from https://www.kaggle.com/selener/multi-class-text-classification-tfidf
model 1 was binary and ultimately confused me even more 
edit 8/9/2021:
    this file explores implemnting pre-existing classification algorithms,
    oversampling, and the use of ensemble classifiers
"""
import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import csv
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
#text preprocessing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling import KMeansSMOTE 
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold

def main():
    #load training data
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/trainData')
    #print(df.shape)
    
    #preprocess documents
    df['clean_document'] = df['Document'].apply(lambda x: finalPreprocess(x))
    
    #encode themes
    df['category_id'] = df['Theme'].factorize()[0]
    le = LabelEncoder()
    enc = OneHotEncoder(sparse=False)
    
    #df['category_id'] = pd.get_dummies(df['Theme'].values)
    category_id_df = df[['Theme', 'category_id']].drop_duplicates()
    #category_id_df = df[['Theme', 'hots']].drop_duplicates()
    
    #dictionaries for future use
    category_to_id = dict(category_id_df.values)
    #print(category_to_id)
    id_to_category = dict(category_id_df[['category_id', 'Theme']].values)
    #id_to_category = dict(category_id_df[['hots', 'Theme']].values)
    
    #makes bar graph of the distribution of themes in the training data
    #uncomment for it to work
    '''#EDA
    fig = plt.figure(figsize=(8,6))
    colors = ['grey','grey','grey','grey','grey','grey','grey','grey','grey',
        'grey','grey','grey','darkblue', 'darkblue', 'darkblue']
    df.groupby('Theme').clean_document.count().sort_values().plot.barh(
        ylim=0, color=colors, title= 'NUMBER OF DOCUMENTS FOR EACH THEME\n')
    plt.xlabel('Number of ocurrences', fontsize = 10);'''
    
    #vectorization
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 2),
                            stop_words='english')
    features = tfidf.fit_transform(df.clean_document).toarray()
    
    #different method of vectorization explored
    #didn't produce better results so it stays commented out 
    '''countvec = CountVectorizer()
    features = countvec.fit_transform(df.clean_document).toarray()
    tvec = TfidfVectorizer()
    features = tvec.fit_transform(df.clean_document).toarray()'''
    
    labels = df.category_id
    
    #dealing with unbalanced data
    #explored different methods and oversampling and undersampling
    #random oversampling method
    ros = RandomOverSampler(random_state=777)
    #X_ROS, Y_ROS = ros.fit_resample(features, labels) #do not do this its the wrong way leads to overfitting
    #random undersampling method
    rus = RandomUnderSampler(random_state=777)
    #X_RUS, Y_RUS = rus.fit_resample(features, labels) #do not do this its the wrong way leads to overfitting
    #SMOTE
    smote = SMOTE(random_state=777,k_neighbors=5)
    #X_smote, Y_smote = smote.fit_resample(features, labels) #do not do this its the wrong way leads to overfitting
    
    #different label ecoding method
    #labels = pd.get_dummies(df.Theme)
    #print(labels)
    
    #print('Each of the %d complaints is represented by %d feaures (TF-IDF score of unigrams and bigrams)' %(features.shape))
    #finding the three most correlated terms with each of the product categories
    #show unigrams and bigrams in data
    n=3
    for Theme, category_id in sorted(category_to_id.items()): 
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        #feature_names = np.array(.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) ==1]
        bigrams = [v for v in feature_names if len(v.split(' ')) ==2]
        '''print("\n==> %s:" %(Theme))
        print(" *Most Correlated Unigrams are: %s" %(', '.join(unigrams[-n:])))
        print(" *Most Correlated Bigrams are: %s" %(', '.join(bigrams[-n:])))'''
        
    #original cross-validation method
    #worked well on original dataset but could not figure out how to use it 
    #with oversampling; worth exploring om how to implement that
    #splitting data into train and test sets
    '''X = df['clean_document'] #Collection of documents
    y = df['Theme'] # target/labels we want to predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    #Models
    models = [
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0)
        ]
    #5 cross-validation
    CV = 5
    cv_df = pd.DataFrame(index=range(CV*len(models)))
    
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        #accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        accuracies = cross_val_score(model, X_ROS, Y_ROS, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    #comparison of model performance
    mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
    std_accuracy = cv_df.groupby('model_name').accuracy.std()
    
    acc = pd.concat([mean_accuracy, std_accuracy], axis=1, ignore_index=True)
    acc.columns = ['Mean Accuracy', 'Standard deviation']
    print(acc)'''
    
    '''plt.figure(figsize=(8,5))
    sns.boxplot(x='model_name', y='accuracy',
                data=cv_df,
                color = 'lightblue',
                showmeans=True)
    plt.title("MEAN ACCURACY (cv = 5)\n", size=14);'''
    
    #model evaluation- train on oversmapled data; test on OG dataset
    X_train, X_test, y_train, y_test, indicies_train, indices_test = train_test_split(features,labels, df.index,
                                                                                      test_size = 0.2,random_state=1)
    #X_trainr, X_testr, y_trainr, y_testr = train_test_split(X_ROS, Y_ROS, test_size = 0.2, random_state=1) #DO NOT DO
    #only use follwoing if using oversampling
    X_trainr, y_trainr = smote.fit_resample(X_train, y_train)
   
    #the following section involves training, testing all sklearn models used in this project
    #uncomment the model you specifically want to work with

    #pre-exisiting sklearn classifiers- 4 models
    #LinearSVC model
    '''model = LinearSVC()
    model.fit(X_train, y_train)
    filename = '/mnt/linuxlab/home/r21sscott/REU_Project/Class_Models/linearsvc.sav' #filepath depends on user
    pickle.dump(model, open(filename, 'wb'))
    #loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(X_test)
    #result = loaded_model.score(X_test, y_test)
    #print(result)
    #in-depth analysis of model performance
    print('\t\t\t\tCLASSIFICATION METRICS\n')
    print(metrics.classification_report(y_test, y_pred))'''
    
    #just exploring the output and predictions of the model
    #only uncomment if want to save a file of the predicted vs. real results
    '''with open('/mnt/linuxlab/home/r21sscott/REU_Project/results', 'a') as file:
            writer = csv.writer(file)
            for x in range(len(y_pred)):
                if y_pred[x] == y_test.iloc[x]:
                    writer.writerow([y_pred[x], y_test.iloc[x],'Correct'])
                else:
                    writer.writerow([y_pred[x], y_test.iloc[x],'Incorrect'])'''
        
    #explored confusion matrix but the number of classes made it too big/messy
    '''conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df.Theme.values, yticklabels=category_id_df.Theme.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()'''
 
    #Logistic Regression model
    '''model2 = LogisticRegression()
    model2.fit(X_trainr, y_trainr)
    #filename2 = '/mnt/linuxlab/home/r21sscott/REU_Project/Class_Models/logreg.sav' #filepath depends on user
    #pickle.dump(model2, open(filename2, 'wb'))
    #loaded_model = pickle.load(open(filename2, 'rb'))
    #y_pred = loaded_model.predict(X_test)
    #result = loaded_model.score(X_test, y_test)
    #print(result)
    y_pred2 = model2.predict(X_test)
    #classification report
    print('\t\t\t\tCLASSIFICATION METRICS\n')
    print(metrics.classification_report(y_test, y_pred2))'''
    
    #Multinomial Niave Bayes model
    '''model3 = MultinomialNB()
    model3.fit(X_trainr, y_trainr)
    #filename3 = '/mnt/linuxlab/home/r21sscott/REU_Project/Class_Models/multiNB.sav' #filepath depends on user
    #pickle.dump(model3, open(filename3, 'wb'))
    #loaded_model = pickle.load(open(filename3, 'rb'))
    #y_pred = loaded_model.predict(X_test)
    #result = loaded_model.score(X_test, y_test)
    #print(result)
    y_pred3 = model3.predict(X_test)
    #classification report
    print('\t\t\t\tCLASSIFICATION METRICS\n')
    print(metrics.classification_report(y_test, y_pred3))'''
    
    #Random Forest Classifier model
    '''model4 = RandomForestClassifier()
    model4.fit(X_trainr, y_trainr)
    filename4 = '/mnt/linuxlab/home/r21sscott/REU_Project/Class_Models/randForest.sav' #filepath depends on user
    pickle.dump(model4, open(filename4, 'wb'))
    #loaded_model = pickle.load(open(filename4, 'rb'))
    #y_pred = loaded_model.predict(X_test)
    #result = loaded_model.score(X_test, y_test)
    #print(result)
    y_pred4 = model4.predict(X_test)
    #classification report
    print('\t\t\t\tCLASSIFICATION METRICS\n')
    print(metrics.classification_report(y_test, y_pred4))'''
    
    #using models as analytic tools (perfroming on unseen data) - 2 models used
    '''
    #LinearSVC Model- analytical
    filename = '/mnt/linuxlab/home/r21sscott/REU_Project/Class_Models/linearsvc.sav'
    df2 = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/analyzeLinearSVC')
    df2['clean_document'] = df2['Document'].apply(lambda x: finalPreprocess(x))
    unseenDocs = tfidf.transform(df2.clean_document).toarray()
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(unseenDocs)
    pred_labels=[]
    for x in range(len(y_pred)):
        temp = id_to_category.get(y_pred[x])
        pred_labels.append(temp)   
    df_csv = pd.DataFrame(data=pred_labels, columns=['Theme'])
    df_csv['Document'] = df2['Document']
    df_csv['Paper'] = 55
    df_csv.to_csv('/mnt/linuxlab/home/r21sscott/REU_Project/analyzeLinearSVC')'''
    
    '''#Random Forest Classifier-  analytical
    filename4 = '/mnt/linuxlab/home/r21sscott/REU_Project/Class_Models/randForest.sav'
    df2 = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/analyzeRandFor')
    df2['clean_document'] = df2['Document'].apply(lambda x: finalPreprocess(x))
    unseenDocs = tfidf.transform(df2.clean_document).toarray()
    loaded_model = pickle.load(open(filename4, 'rb'))
    y_pred = loaded_model.predict(unseenDocs)
    pred_labels=[]
    for x in range(len(y_pred)):
        temp = id_to_category.get(y_pred[x])
        pred_labels.append(temp)   
    df_csv = pd.DataFrame(data=pred_labels, columns=['Theme'])
    df_csv['Document'] = df2['Document']
    df_csv['Paper'] = 55
    df_csv.to_csv('/mnt/linuxlab/home/r21sscott/REU_Project/analyzeRandFor')'''
    
    #exploring ensemble classifiers- 4 models
    #Voting Classifier
    '''model = LinearSVC()
    model.fit(X_train, y_train)
    #model2 = LogisticRegression()
    #model2.fit(X_train, y_train)
    #model3 = MultinomialNB()
    #model3.fit(X_train, y_train)
    model4 = RandomForestClassifier()
    model4.fit(X_train, y_train)
    eclf = VotingClassifier(estimators=[('lsvc', model), ('rf', model4)],
                                        voting='hard')
    eclf.fit(X_train, y_train)
    y_predv = eclf.predict(X_test)
    #classification report
    print('\t\t\t\tCLASSIFICATION METRICS\n')
    print(metrics.classification_report(y_test, y_predv))'''
    
    #Bagging Classifier
    '''modelb = BaggingClassifier()
    modelb.fit(X_trainr, y_trainr)
    y_predb = modelb.predict(X_test)
    #classification report
    print('\t\t\t\tCLASSIFICATION METRICS\n')
    print(metrics.classification_report(y_test, y_predb))'''
    
    #XGB Classifier
    '''modelbo = XGBClassifier()
    modelbo.fit(X_trainr, y_trainr)
    y_predb = modelbo.predict(X_test)
    #classification report
    print('\t\t\t\tCLASSIFICATION METRICS\n')
    print(metrics.classification_report(y_test, y_predb))'''
    
    #Gradient boosting
    '''modelgb = GradientBoostingClassifier()
    modelgb.fit(X_trainr, y_trainr)
    y_predgb = modelgb.predict(X_test)
    #classification report
    print('\t\t\t\tCLASSIFICATION METRICS\n')
    print(metrics.classification_report(y_test, y_predgb))'''
    
    #using stratified k-fold evaluation
    '''model1 = GradientBoostingClassifier()
    scores = score_model(model1, features, labels)
    print(scores)
    model1Score = np.mean(scores)
    print(model1Score)'''
    
#stratified k-fold that works on all data
def score_model(model, features, labels):
    cv = StratifiedKFold(n_splits=5)
    ros = RandomOverSampler(random_state=777)
    smote = SMOTE(random_state=777,k_neighbors=5)
    
    scores = []
    
    for train, val in cv.split(features, labels):
        X_train_fold, y_train_fold = features[train], labels[train]
        X_val_fold, y_val_fold = features[val], labels[val]
        X_train_fold_upsample, y_train_fold_upsample = smote.fit_resample(X_train_fold, y_train_fold) #oversampling
        model_obj = model.fit(X_train_fold_upsample, y_train_fold_upsample)
        #model_obj = model.fit(X_train_fold, y_train_fold) #uncomment if want to work with OG dataset
        score = model_obj.score(X_val_fold, y_val_fold)
        scores.append(score)
    return np.array(scores)
                                      
        

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
    
    