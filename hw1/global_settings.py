#!/bin/python
import numpy
C = numpy.logspace(-2, 3, 5)
max_iter = 10000
num_jobs = 6
gridsearch_verbosity = 3
training_verbosity = 3

solver = ['sag', 'saga']#'liblinear', 'lbfgs']
scaling = True
tfidf = True
analyzer = 'char_wb'
use_idf = True
ngram_min = 3
ngram_max = 10
stop_vocab = None
sublinear_tf = True

gridSearchParams = {'C': C,
                    'solver': solver,
                    'max_iter': [max_iter]}
    #,
    #               'fit_intercept': [True, False]}