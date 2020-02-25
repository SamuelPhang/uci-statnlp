import numpy as np


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array - score[token][label]
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    y = [] # score table
    sequence = [] # final sequence
    # init table, N rows and L columns
    for i in range(N):
        y.append([0]*L)

    for i in range(N):
        for y_i in range(L):
            max_score = float("-inf")
            for y_prev in range(L):
                score = 0
                if i == 0: # if first token, use start transition
                    # note, previous score is zero
                    score = emission_scores[i][y_i] + start_scores[y_i]
                else: # else lookup transition score in table
                    score = emission_scores[i][y_i] + trans_scores[y_prev][y_i] + y[i-1][y_prev]
                
                max_score = max(score, max_score)
            y[i][y_i] = max_score 

    final_score = float("-inf")
    for y_end in range(L): # consider end transition scores, assume zero emission for eos
        # y_end is the label of the last token
        score = end_scores[y_end] + y[N-1][y_end]
        final_score = max(score, final_score)

    # TODO: get sequence
    sequence = [0]*N

    # score set to 0
    return (final_score, sequence)