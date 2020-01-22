#!/bin/python
import numpy
C = numpy.linspace(20, 30, 10)
max_iter = 10000
num_jobs = 6
gridsearch_verbosity = 3
training_verbosity = 0

solver = ['liblinear', 'sag', 'lbfgs']
scaling = False

#word2vec params
min_length = 1
importance = .001

#tfidfvectorizer params
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t).lower().strip('.,;:') for t in word_tokenize(doc) if len(t) >= min_length]

tfidf = True
analyzer = 'word' #char_wb'
use_idf = True
ngram_min = 1
ngram_max = 1
stop_vocab = None #['the', 'a', 'an', 'and', 'of', 'that', 'to']
sublinear_tf = True
binary = False
tokenizer = LemmaTokenizer() #None

use_grid_search = False
gridSearchParams = {'C': C,
                    'solver': solver,
                    'max_iter': [max_iter]}

nonGridSearchParams = {'C': 21
                       , 'solver': 'liblinear'
                       , 'max_iter': max_iter}
