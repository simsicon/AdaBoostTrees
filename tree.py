from __future__ import division

import random
import time
import numpy as np
from collections import Counter


class Node:
    STOP_ESITIMATOR_NUM = 10

    def __init__(self, X, y, verbose=False, verboseverbose=False):
        self.X = X
        self.y = y
        self.left_child, self.right_child = None, None
        self.is_leaf = False
        self.best_attr_index = None
        self.threshold_func = self.random_uniform_threshold_split
        self.before_split_entropy = self.entropy(self.y)
        self.verbose = verbose
        self.verboseverbose = verboseverbose

    def walk(self, x, indent=0):
        if self.is_leaf:
            _v = self.vote()
            if self.verboseverbose:
                print indent * "  " + "leaf: %s" % _v
            return _v

        if self.verboseverbose:
            print indent * "  " + "branch: %s, %s" % (self.best_attr_index, self.best_threshold)

        if x[self.best_attr_index] <= self.best_threshold:
            return self.left_child.walk(x, indent=indent+1)
        elif x[self.best_attr_index] > self.best_threshold:
            return self.right_child.walk(x, indent=indent+1)

    def vote(self):
        if self.is_leaf:
            return Counter(self.y).most_common(1)[0][0]
        else:
            return None

    def choose_best_attr(self):
        if self.X.shape[0] < self.STOP_ESITIMATOR_NUM:
            if self.verboseverbose:
                print "time to stop with sample %s, %s" % self.X.shape

            self.is_leaf = True
            return

        max_info_gain = -1
        _best_attr_index = None
        _best_threshold = None
        _best_X_left, _best_y_left, _best_X_right, _best_y_right = None, None, None, None

        for i in range(self.X_attrs_num()):
            X_left, y_left, X_right, y_right, threshold, conditional_entropy = \
                 self.split_with_attr(i)

            info_gain = self.before_split_entropy - conditional_entropy

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                _best_attr_index = i
                _best_threshold = threshold
                _best_X_left, _best_y_left, _best_X_right, _best_y_right = X_left, y_left, X_right, y_right

            if self.verboseverbose:
                print "attr %s with info gain %s, current max info gain is %s" % (i, info_gain, max_info_gain)

        if _best_attr_index is not None:
            self.best_attr_index = _best_attr_index
            self.best_threshold = _best_threshold
            self.X_left = _best_X_left
            self.y_left = _best_y_left
            self.X_right = _best_X_right
            self.y_right = _best_y_right

    def split_with_attr(self, attr_index):
        if self.threshold_func is None:
            self.threshold_func = self.random_uniform_threshold_split

        return self.threshold_func(attr_index)

    def random_uniform_threshold_split(self, attr_index):
        X_sorted = sorted(self.X, key=lambda x: x[attr_index])
        _min, _max = X_sorted[0][attr_index], X_sorted[-1][attr_index]
        threshold = np.random.uniform(_min, _max, 1)[0]
        _conditional_entropy, _X_left, _y_left, _X_right, _y_right = self.conditional_entropy(attr_index, threshold)
        return _X_left, _y_left, _X_right, _y_right, threshold, _conditional_entropy

    def exhaustive_approx_threshold_split(self, attr_index, approx_precision=1):
        total_count = len(self.y)

        X_sorted = sorted(self.X, key=lambda x: x[attr_index])
        thresholds = [(X_sorted[i][attr_index] + X_sorted[i+1][attr_index]) / 2
                      for i in range(total_count) if i < total_count - 1]
        approx_thresholds = set([round(threshold, approx_precision) for threshold in thresholds])

        _best_threshold_of_attr = None
        _max_info_gain_of_attr = -1
        _least_conditional_entropy = None

        if self.verboseverbose:
            print "    %s thresholds to approx: %s" % (len(approx_thresholds), approx_thresholds)

        for threshold in approx_thresholds:
            _conditional_entropy, _X_left, _y_left, _X_right, _y_right = self.conditional_entropy(attr_index, threshold)
            info_gain = self.before_split_entropy - _conditional_entropy

            if info_gain > _max_info_gain_of_attr:
                _max_info_gain_of_attr = info_gain
                _best_threshold_of_attr = threshold
                _least_conditional_entropy = _conditional_entropy
                X_left, y_left, X_right, y_right = _X_left, _y_left, _X_right, _y_right

            if self.verboseverbose:
                print "        approx threshold %s with info gain %s" % (threshold, info_gain)

        return X_left, y_left, X_right, y_right, _best_threshold_of_attr, _least_conditional_entropy


    def X_len(self):
        return self.X.shape[0]

    def X_attrs_num(self):
        return self.X.shape[1]

    def entropy(self, labels):
        labels_counter = Counter(labels)
        total_count = len(labels)
        label_probabilities = [label_count / total_count for label_count in labels_counter.values()]
        return sum([ - p * np.log2(p) for p in label_probabilities if p])

    def conditional_entropy(self, attr_index, threshold):
        total_count = len(self.y)
        _X_left, _y_left, _X_right, _y_right = self.partitions(attr_index, threshold)

        entropy_left = self.entropy(_y_left)
        entropy_right = self.entropy(_y_right)

        _conditional_entropy = ((len(_y_left) / total_count) * entropy_left +
                                (len(_y_right) / total_count) * entropy_right)

        return _conditional_entropy, _X_left, _y_left, _X_right, _y_right

    def partitions(self, attr_index, threshold):
        indices_left = [i for i, x_i in enumerate(self.X) if x_i[attr_index] <= threshold]
        indices_right = [i for i, x_i in enumerate(self.X) if x_i[attr_index] > threshold]

        X_left = np.array([self.X[i] for i in indices_left])
        y_left = np.array([self.y[i] for i in indices_left])
        X_right = np.array([self.X[i] for i in indices_right])
        y_right = np.array([self.y[i] for i in indices_right])

        return X_left, y_left, X_right, y_right

    def X_left_len(self):
        return self.X_left.shape[0]

    def X_right_len(self):
        return self.X_right.shape[0]

class Tree:
    def __init__(self, X, y, verbose=False, verboseverbose=False):
        self.X = X
        self.y = y
        self.verbose = verbose
        self.verboseverbose = verboseverbose
        self.root = self.build_tree(self.X, self.y)

    def build_tree(self, X, y, indent=0):
        """
        Three concerns:

        1. Node has no enough samples to choose the best attr and split,
           then return the node itself.
        2. Either left or right child has no enough samples to continue,
           then attach the child and contiue the other.
           If the other child is classified, return the node.
        3. Neither left nor right child has enough samples to continue,
           then attach the both children and return the node itself.
        """

        if self.verbose:
            print indent * "  " + str(X.shape[0])

        if X.shape[0] == 0:
            return None

        node = Node(X, y, verbose=self.verbose, verboseverbose=self.verboseverbose)

        if len(set(y)) == 1 or node.X_len() < node.STOP_ESITIMATOR_NUM:
            node.is_leaf = True
            return node

        node.choose_best_attr()

        if not node.is_leaf:
            node.left_child = self.build_tree(node.X_left, node.y_left, indent=indent + 1)
            node.right_child = self.build_tree(node.X_right, node.y_right, indent=indent + 1)

        return node

    def predict(self, X):
        return [self.root.walk(x_i) for x_i in X]
