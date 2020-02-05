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
    def __init__(self, smooth_lambda=1, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)
        self.smooth = smooth_lambda

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')  # because periods are stripped off?

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
            from collections import defaultdict
            self.totals = defaultdict(int)  # total count for every word
            self.lunk_prob = log(unk_prob, 2)
            # self.contexts = defaultdict(int)  # total count for every context (not including word)
            self.context_sum = 'CONTEXT_SUM'
            if n > 0:
                self.n = n
            else:
                print('N must be 1 or greater')
                raise ValueError()

        def fit_corpus(self, corpus):
            """Learn the language model for the whole corpus.

            The corpus consists of a list of sentences."""
            self.__get_totals(corpus)
            for s in corpus:
                self.fit_sentence(s)
            # self.norm()

        def inc_word(self, context):
            """Given a list containing a word and its previous N-1 words, add these to the model and update the count.
            Example: ['Sam', 'I', 'am'] where the word is 'am', using a trigram model"""

            assert (len(context) == self.n)  # first check that the list is the correct length
            context_sum = self.context_sum

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

            # increment sum of all words given this context
            if context_sum in curr_level:
                curr_level[context_sum] += 1.0
            else:
                curr_level[context_sum] = 1.0

            # self.contexts[context[:-1]] += 1

        def __get_totals(self, corpus):
            for s in corpus:
                for w in s:
                    self.totals[w] += 1
                self.totals['END_OF_SENTENCE'] += 1

        def fit_sentence(self, sentence):
            """Add start of sentence markers and place words and their contexts into model. self.totals must be
            computed already """
            unks = self.__unk_parse()  # get set of unknowns first
            sos = 'START_OF_SENTENCE'
            unk = 'UNK'
            context = []
            for w_index in range(len(sentence)):
                context = []
                for sos_char in range(self.n - 1 - w_index):
                    context.append(sos)  # add appropriate number of start of sentence tokens
                curr_word = ""
                for offset in range(-min(w_index, self.n - 1), 1, 1):
                    curr_word = sentence[w_index + offset]
                    if curr_word in unks:
                        curr_word = unk
                    context.append(curr_word)  # add the remaining context to the list
                if curr_word in unks:  # if the word (not context) is unk, increase its count
                    self.totals[unk] += 1
                self.inc_word(context)

            context.append('END_OF_SENTENCE')
            context.pop(0)
            self.inc_word(context)

        def __unk_parse(self):
            """Returns a list of unknown words"""
            unknowns = set()
            ltotal = log(sum(self.totals.values()), 2)  # total num of all words
            for word, count in self.totals.items():  # get list of unknown words
                if log(count, 2) - ltotal < self.lunk_prob:
                    unknowns.add(word)

            return unknowns

        def norm(self):
            """Normalize and convert to log2-probs, including smoothing."""
            for word in self.model:
                ldenom = log(self.totals[word] + self.smooth * len(self.model), 2)
                self.__norm_recurse(self.model[word], ldenom)

        def __norm_recurse(self, d, ltotal):
            for k, v in d.iteritems():
                if isinstance(v, d):
                    self.__norm_recurse(v, ltotal)
                else:
                    d[k] = log(d[k] + self.smooth, 2) - ltotal

        def cond_logprob(self, word, previous, numOOV):
            """Returns the conditional log probability given a word and its N-gram context.
            The context is a list of the N-1 words before the given word."""
            assert len(previous) == self.n - 1
            V = len(self.vocab() - 1)
            lunk_factor = 0  # set to numOOV if word is an unknown
            if word not in self.model:
                word = 'UNK'
                lunk_factor = log(numOOV, 2)

            num_in_context = self.__context_checker(previous)
            if num_in_context == -1:  # if context not found, return log(1/V)
                return -log(len(V-1), 2) - lunk_factor

            #  go to dictionary containing word (n-1th level)
            curr_level = self.model
            for i in range(0, len(previous)):
                curr_level = curr_level[previous[i]]
            #  check if word hasn't appeared in this context
            if word not in curr_level:
                return log(self.smooth, 2) - log(self.smooth*V + num_in_context) - lunk_factor
            else:
                cond_count = curr_level[word]
                return log(self.smooth + cond_count, 2) - log(self.smooth*V + num_in_context) - lunk_factor

        def __context_checker(self, previous):
            """Return -1 if context doesn't exist.
            Else return the total number of words in this context"""
            curr_level = self.model
            for x in previous:
                if x in curr_level:
                    curr_level = curr_level[x]
                else:
                    return -1
            return curr_level[self.context_sum]

        def vocab(self):
            return self.model.keys()

        def perplexity(self, corpus):
            """Computes the perplexity of the corpus by the model.

            Assumes the model uses an EOS symbol at the end of each sentence.
            """
            vocab_set = set(self.vocab())
            words_set = set([w for s in corpus for w in s])
            numOOV = len(words_set - (vocab_set - 1))  # -1 b/c end of sentence is in model
            return pow(2.0, self.entropy(corpus, numOOV))