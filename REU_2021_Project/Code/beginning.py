'''last date modifed: 6/22/2021
author: Savannah Scott
Objectives: Tokenize the text for a single federalist paper
            Look at statistical analysis of essay to uncover themes
edit 8/6/2021:
    very first efforts into the project
    this is file is more exploratory data analysis.
    looking into how to tokenize raw text after light preprocessing '''
import nltk, re, pprint
from nltk.probability import FreqDist
from nltk.corpus import stopwords
#nltk.download('webtext')
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

def main():
    '''open file and store raw data into variable'''
    file = open('/mnt/linuxlab/home/r21sscott/REU_Project/FEDERALIST No. 42.txt', 'r') #filepath depends on user
    rawtxt = file.read()
    file.close()
    tokens = tokenize(rawtxt)
    filtered_tokens =cleanUpTxt(tokens)
    print(filtered_tokens)
    numWords = freqWords(filtered_tokens)
    #ngrams()
    
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

def freqWords(tokens):
    '''find the frequency distribution of the tokens passed in'''
    fdist1 = FreqDist(tokens)
    words = fdist1.most_common(50)
    return(words)

def ngrams():
    '''find collocations/ bigrams from text'''
    words = [w.lower() for w in webtext.words('/mnt/linuxlab/home/r21sscott/REU_Project/FEDERALIST No. 42.txt')]
    bigram_collocation = BigramCollocationFinder.from_words(words)
    stopset = set(stopwords.words('english'))
    filter_stops = lambda w: len(w) < 3 or w in stopset
    bigram_collocation.apply_word_filter(filter_stops)
    print(bigram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 25))
    '''work with trigrams'''
    trigram_collocation = TrigramCollocationFinder.from_words(words)
    trigram_collocation.apply_word_filter(filter_stops)
    trigram_collocation.apply_freq_filter(3)
    print(trigram_collocation.nbest(TrigramAssocMeasures.likelihood_ratio, 10))
    
    
    
main()
