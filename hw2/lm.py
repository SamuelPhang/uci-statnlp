#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))


    def unicode(*args, **kwargs):
        return str(*args, **kwargs)


class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        vocab_set = set(self.vocab())
        words_set = set([w for s in corpus for w in s])
        numOOV = len(words_set - vocab_set)
        return pow(2.0, self.entropy(corpus, numOOV))

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1  # for EOS
            sum_logprob += self.logprob_sentence(s, numOOV)
        return -(1.0 / num_words) * (sum_logprob)

    def logprob_sentence(self, sentence, numOOV):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i], numOOV)
        p += self.cond_logprob('END_OF_SENTENCE', sentence, numOOV)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence):
        pass

    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self):
        pass

    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV):
        pass

    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        pass


class Unigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')  # because spaces are stripped off?

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
            return self.model[word]
        else:
            return self.lunk_prob - log(numOOV, 2)

    def vocab(self):
        return self.model.keys()

    class Ngram(LangModel):
        def __init__(self, unk_prob=0.0001, n=3):
            self.model = dict()
            self.lunk_prob = log(unk_prob, 2)
            if n > 0:
                self.n = n
            else:
                print('N must be 1 or greater')
                raise ValueError()

        def inc_word(self, context):
            """Given a list containing a word and its previous N-1 words, add these to the model and update the count.
            Example: ['Sam', 'I', 'am'] where the word is 'am', using a trigram model"""

            assert (len(context) == self.n)  # first check that the list is the correct length

            curr_level = self.model
            for i in range(self.n):
                w = context[i]
                if i < self.n - 1:  # if not at actual word (still at context word)
                    if w not in curr_level:
                        # add entry if it didn't exist before
                        curr_level[w] = dict()
                    curr_level = curr_level[w]
                else:
                    if w in curr_level:
                        curr_level[w] += 1.0
                    else:
                        curr_level[w] = 1.0

        def fit_sentence(self, sentence):
            """Add start of sentence markers and place words and their contexts into model"""
            sos = 'START_OF_SENTENCE'
            context = []
            for w_index in range(len(sentence)):
                context = []
                for sos_char in range(self.n - 1 - w_index):
                    context.append(sos)  # add appropriate number of start of sentence tokens
                for offset in range(-min(w_index, self.n - 1), 1, 1):
                    context.append(sentence[w_index + offset])  # add the remaining context to the list
                self.inc_word(context)

            context.append('END_OF_SENTENCE')
            context.pop(0)
            self.inc_word(context)

        def norm(self):
            """Normalize and convert to log2-probs."""
            tot = 0.0
            for word in self.model:
                tot += self.model[word]
            ltot = log(tot, 2)
            for word in self.model:
                self.model[word] = log(self.model[word], 2) - ltot

        def cond_logprob(self, word, previous, numOOV):
            if word in self.model:
                return self.model[word]
            else:
                return self.lunk_prob - log(numOOV, 2)

        def vocab(self):
            return self.model.keys()
