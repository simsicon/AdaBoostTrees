from __future__ import division

import random
import time
import numpy as np
from collections import Counter

import tree

class Boosting:
    def __init__(self, X, y, n_estimators=10, n_samples=1024):
        self.X = X
        self.y = y

        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.N = self.y.shape[0]
        self.weights = [1 / self.N for _ in range(self.N)]
        self.alphas = []
        self.estimators = None
        self.count = 0

    def init_estimator(self):
        indices = [i for i in np.random.choice(X.shape[0], self.n_samples, p=self.weights)]
        X_tree = np.array([X[i, :] for i in indices])
        y_tree = np.array([y[i] for i in indices])

        print "%s / %s" % (self.count, self.n_estimators)

        while True:
            t1 = time.time()
            tree = Tree(X_tree, y_tree)
            t2 = time.time()

            print "tree generation time: %s" % (t2 - t1)

            predictions = tree.predict(self.X)
            accuracy = accuracy_score(self.y, predictions)
            print "accuracy: %s" % accuracy
            if accuracy != 0.50:
                self.estimators.append(tree)
                break

        return tree, predictions

    def train(self):
        self.count = 0
        self.estimators = []
        t1 = time.time()
        for _ in range(self.n_estimators):
            self.count += 1

            estimator, y_pred = self.init_estimator()

            errors = np.array([ y_i != y_p for y_i, y_p in zip(y, y_pred)])
            agreements = [-1 if e else 1 for e in errors]
            epsilon = sum(errors * self.weights)

            print "epsilon: %s" % epsilon
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            z = 2 * np.sqrt(epsilon * ( 1 - epsilon))
            self.weights = np.array([(weight / z) * np.exp(-1 * alpha * agreement)
                                     for weight, agreement in zip(self.weights, agreements)])
            print "weights sum: %s" % sum(self.weights)
            self.alphas.append(alpha)
        t2 = time.time()
        print "train took %s s" % (t2 - t1)

    def predict(self, X):
        predicts = np.array([estimator.predict(X) for estimator in self.estimators])
        weighted_prdicts = [[(p_i, alpha) for p_i in p] for alpha, p in zip(self.alphas, predicts)]

        H = []
        for i in range(X.shape[0]):
            bucket = []
            for j in range(len(self.alphas)):
                bucket.append(weighted_prdicts[j][i])
            H.append(bucket)

        return [self.weighted_majority_vote(h) for h in H]

    def weighted_majority_vote(self, h):
        weighted_vote = {}
        for label, weight in h:
            if label in weighted_vote:
                weighted_vote[label] = weighted_vote[label] + weight
            else:
                weighted_vote[label] = weight

        max_weight = 0
        max_vote = 0
        for vote, weight in weighted_vote.iteritems():
            if max_weight < weight:
                max_weight = weight
                max_vote = vote

        return max_vote
