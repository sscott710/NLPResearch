#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:17:36 2021

@author: r21sscott
This code will be an attempt to hard code the themes of The Federalist Papers.
Note: this will probaly be very inefficient and not the most sophisticated. I am looking for a proof of concept before
implemnting more AI/machine learning type algorithms
Objectives:
    create sufficeint lists of words for each theme
    clean those up so they can work well with tokens
    identify themes in the essay
    visualize work/ make it easy to check
    
edit 8/6/2021:
    very simple rule-based approach; it is very ineffiecient but shows the importance of context and
    automating token assignment
"""
import nltk, re, pprint
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.transforms import IdentityTransform


def main():
    file = open('/mnt/linuxlab/home/r21sscott/REU_Project/FEDERALIST No. 42.txt', 'r') #chose to first work with No.42 because it is credited as the most cited Fed Paper
    rawtxt = file.read()
    file.close()
    tokens = tokenize(rawtxt)
    filtered_tokens =cleanUpTxt(tokens)
    themes = theme()
    themeTokens = identify(filtered_tokens, themes)
    visual(themeTokens)


def tokenize(rawtxt):
    '''tokenize text'''
    tokens = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", rawtxt)
    return tokens

def cleanUpTxt(tokens):
    '''make all words lowercase and remove stop words'''
    lower_tokens = []
    for w in tokens:
        word = w.lower()
        lower_tokens.append(word)
    stop_words = stopwords.words('english')
    stop_pun = [',', ';', '.', '?', '!', ':', '"']
    filtered_tokens=[]
    for w in lower_tokens:
        if w not in stop_words:
            if w not in stop_pun:
                filtered_tokens.append(w)
    return filtered_tokens

'''create idenitfiers of each theme based on phrases or words'''
def theme():
    union = ['uniformity', 'uniform']
    aoc = ['articles of confederation', 'regulations of states', 'article of confederation', 'improved on model', 'defects of confederation', 'new constitution', 'improvement on articles of confederation', 'defect of power', 'limitations articles of confederation', 'sovereignty union', 'existing congress', 'confusion', 'improper power', 'serious embarrassments', 'improvement on clause relating to this subject articles of confederation']
    sfg =['powers lodged general government', 'general government', 'federal government', 'harmony and proper intercourse among states', 'superintending authority', 'general laws', 'public care']    
    rp = []
    cab = []
    sop = []
    lb = ['Congress']
    eb = ['president', 'executive branch', 'executive']
    jb = ['judicial branch']
    ncr = ['natural rights', 'civil rights', 'right']
    hn = ['common knowledge of human affairs']
    fed = ['foreign','federal administration', 'restriction on general government', 'restraints imposed on authority of states', 'legislative right of any state']
    igt = []
    nat = ['one nation', 'the people', 'great body', 'great bulk of the citizens', 'majority of America']
    themes = {'union' : union, 'aoc' : aoc, 'sfg' : sfg, 'rp' : rp, 'cab': cab, 'sop': sop, 'lb': lb, 'eb':eb, 'jb':jb, 'ncr':ncr, 'hn':hn, 'fed':fed, 'igt':igt, 'nat':nat}
    #print(list(themes)[0])
    return (themes)

'''match tokens with themes'''
def identify(tokens, themes):
    tempThemeTokens = []
    for x in range(len(tokens)):
        themeToken = tokenCheck(tokens[x], themes)
        tempThemeTokens.append(themeToken)
    themeTokens = []
    for x in range(len(tempThemeTokens)):
        if tempThemeTokens[x][0] != 0:
            themeTokens.append(tempThemeTokens[x])
    return themeTokens
                  
'''see if token was in phrase; okay idea but would lead to over counting'''
def tokenCheck(word, themes):
    for x in range(len(list(themes))):
        for y in range(len(list(themes.values())[x])):
            if word in list(themes.values())[x][y] or word == list(themes.values())[x][y]:
                themeToken = (word, list(themes)[x])
                return themeToken
    return (0, 0) #for null tuple

def visual(themeTokens):
    '''makes bar graph'''
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    thc = {'union': 0, 'aoc':0,'sfg':0, 'rp':0, 'cab':0, 'sop':0,'lb':0, 'eb':0, 'jb':0, 'ncr':0, 'hn':0, 'fed':0, 'igt':0, 'nat':0}
    th = list(thc.keys())
    for x in range(len(themeTokens)):
        for y in range(len(th)):
            if themeTokens[x][1] == th[y]:
                thc[th[y]] = thc[th[y]] + 1
    thy = list(thc.values())
    ax.bar(th,thy)
    plt.show()
            
    
main()
    
