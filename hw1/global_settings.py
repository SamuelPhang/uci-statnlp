#!/bin/python
import numpy
C = numpy.linspace(0.5, 10, 20)
max_iter = 1000

solver = ['sag', 'liblinear', 'lbfgs']
scaling = False
tfidf = True
use_idf = True
ngram_max = 1
stop_vocab = None

gridSearchParams = {'C': C,
                    'solver': solver,
                    'max_iter': [max_iter],
                    'fit_intercept': [True, False]}