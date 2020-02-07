import hw1.global_settings as g
from hw1.speech import *
from gensim.models import Word2Vec
import numpy as np
import sys

def read_unlabeled(tarfname, speech):
    """Reads the unlabeled data.

	The returned object contains three fields that represent the unlabeled data.

	data: documents, represented as sequence of words
	fnames: list of filenames, one for each document
	X: bag of word vector for each document, using the speech.vectorizer
	"""
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")

    class Data:
        pass

    unlabeled = Data()
    unlabeled.data = []
    unlabeled.fnames = []
    for m in tar.getmembers():
        if "unlabeled" in m.name and ".txt" in m.name:
            unlabeled.fnames.append(m.name)
            unlabeled.data.append(read_instance(tar, m.name))
    unlabeled.X = speech.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled


def dev_evaluate(X, yt, cls, dev_corpus, vectorizer):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics, preprocessing

    X = adjust_features_wv(X, dev_corpus, vectorizer)
    if g.scaling:
        X = preprocessing.scale(X, with_mean=False)
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("  Accuracy", acc)
    return yp


def adjust_features_wv(X, corpus, vectorizer):
    import nltk
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer

    alpha = g.importance
    nltk.download('punkt')
    nltk.download('wordnet')
    lmtzr = WordNetLemmatizer()
    lemmatized = [[lmtzr.lemmatize(word).lower() for word in word_tokenize(doc.decode("utf-8")) if len(word) >= g.min_length] for doc in
                  corpus]

    wv = Word2Vec.load('word2vec_old.model')
    X = X.tolil()

    for doc_index in range(len(corpus)):
        if (doc_index%100 == 0):
            print("Adjusting doc number; " + str(doc_index))
        for word in lemmatized[doc_index]:
            if word not in vectorizer.vocabulary_:
                # print(word + " not in vectorizer vocab")
                i = 0  # feature index
                for element in vectorizer.get_feature_names():
                    try:
                        X[doc_index, i] += alpha * wv.wv.similarity(word, element)
                        if X[doc_index, i] < 0:
                            X[doc_index, i] = 0
                        # print('X[{}, {}] changed by {}'.format(doc_index, i, alpha*wv.wv.similarity(word, element)))
                    except Exception as e:
                        # print(e, file=sys.stderr)
                        # print(lemmatized[doc_index])
                        # print("unknown word: " + word)
                        # print("compared feature word: " +
                        pass
                    i += 1

    return X.tocsr()


#def main():
if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/speech.tar.gz"
    speech = read_files(tarfname)
    print(type(speech.devX))
    print("Training classifier")
    import hw1.classify as classify

    cls = classify.train_classifier(speech.trainX, speech.trainy)
    print("Evaluating")
    classify.evaluate(speech.trainX, speech.trainy, cls)
    predictions = dev_evaluate(speech.devX, speech.devy, cls, speech.dev_data, speech.count_vect)

    # print("Reading unlabeled data")
    # unlabeled = read_unlabeled(tarfname, speech)
    # print("Writing pred file")
    #
    # unlabeled.X = adjust_features_wv(unlabeled.X, unlabeled.data, speech.count_vect)
    # write_pred_kaggle_file(unlabeled, cls, "data/speech-pred.csv", speech)

# if __name__ == "__main__":
#     main()


