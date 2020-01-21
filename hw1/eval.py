from . import speech
import numpy as np
import pandas as pd

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in a document and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_mean_feats(Xtr, features, grp_ids, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    # if grp_ids:
    #     D = Xtr[grp_ids].toarray()
    # else:
    #     D = Xtr.toarray()
    D = Xtr[grp_ids].toarray()
    D[D < min_tfidf] = 0

    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

if __name__ == "__main__":
    speech.main()

#get top tfidf scores averaged
    rand_indices = np.random.randint(0, (speech.trainX.shape[0]) - 1, 50)
    df = top_mean_feats(speech.trainX, speech.count_vect.get_feature_names(), rand_indices, top_n=100)
    print(df[-10:])

#get highest weighted features
    weights = abs(cls.coef_).mean(0)
    sorted_weights = [(x[0], x[1]) for x in sorted(zip(weights, speech.count_vect.get_feature_names()))]
    df = pd.DataFrame(sorted_weights)
    df.columns = ['feature', 'average of abs(coefficients)']
    print(df[-10:])