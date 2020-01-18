#!/bin/python
import global_settings as g


def train_classifier(X, y):
    """Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    if g.scaling:
        from sklearn import preprocessing
        X = preprocessing.scale(X, with_mean=False)
    clf = GridSearchCV(
        LogisticRegression(), g.gridSearchParams, n_jobs=g.num_jobs, verbose=g.gridsearch_verbosity
    )
    clf.fit(X, y)
    print(clf.best_params_)
    cls = LogisticRegression(**clf.best_params_, verbose=g.training_verbosity)
    cls.fit(X, y)
    return cls


def evaluate(X, yt, cls):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics, preprocessing

    if g.scaling:
        X = preprocessing.scale(X, with_mean=False)
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("  Accuracy", acc)
