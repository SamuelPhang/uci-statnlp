#!/bin/python
import global_settings as g
def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression

	if g.scaling:
		from sklearn import preprocessing
		X = preprocessing.scale(X, with_mean=False)
	cls = LogisticRegression(C = g.C, max_iter=g.max_iter)
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
