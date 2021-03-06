import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine

from copy import copy

from common import SoftTreeNodeClassifier


class TreeNodeClassifier(DecisionTreeClassifier):

    def __init__(
        self, criterion='gini', splitter='best', min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
        random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, class_weight=None, presort='deprecated'):
        super().__init__(
            criterion, splitter, 1, min_samples_split, min_samples_leaf,
            min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
            min_impurity_decrease, min_impurity_split, class_weight, presort)
        self.child_left = None
        self.child_right = None
        self.feature = None
        self.threshold = None
        self.value = None
        self.impurity = None
        self.n_node_samples = None
        self.classes = None


    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        super().fit(X, y, sample_weight, check_input,
            X_idx_sorted)
        self.child_left = self.tree_.children_left[0]
        self.child_right = self.tree_.children_right[0]
        self.feature = self.tree_.feature[0]
        self.threshold = self.tree_.threshold[0]
        self.value = self.tree_.value
        self.impurity = self.tree_.impurity
        self.n_node_samples = self.tree_.n_node_samples
        return self


class SoftDecisionTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self, tree_node_classifier=None, soft_tree_node_classifier=None, feature_selection=True):
        self.tree_node_classifier = tree_node_classifier if tree_node_classifier else TreeNodeClassifier()
        self.soft_tree_node_classifier = soft_tree_node_classifier if soft_tree_node_classifier else SoftTreeNodeClassifier()
        self.tree_node_classifiers = []
        self.soft_tree_node_classifiers = []
        self.classes = None
        self.feature_selection = feature_selection

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        self.classes = np.unique(y)
        self._fit(X, y, sample_weight, check_input, X_idx_sorted, is_root=True)
        #print([clf.child_left for clf in self.soft_tree_node_classifiers])
        return self
        
    def _fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None, is_root=False):
        unique_y = np.unique(y)
        print(unique_y)
        
        tree_node_classifier = copy(self.tree_node_classifier)
        tree_node_classifier.classes = unique_y
        tree_node_classifier.fit(X, y, sample_weight, check_input, X_idx_sorted)
        self.tree_node_classifiers.append(tree_node_classifier)
        feature_idx = tree_node_classifier.feature
        leaves = tree_node_classifier.apply(X) - 1

        soft_tree_node_classifier = copy(self.soft_tree_node_classifier)
        soft_tree_node_classifier.classes = unique_y
        soft_tree_node_classifier.feature = feature_idx
        self.soft_tree_node_classifiers.append(soft_tree_node_classifier)
        #print(tree_node_classifier.value, tree_node_classifier.impurity)

        impurity = tree_node_classifier.impurity
        # left node
        child_left = tree_node_classifier.child_left
        child_presence = False 
        if impurity[0] > impurity[child_left] and impurity[child_left] > 0.0:
            child_presence = True
            #print('left')
            is_left = leaves == child_left - 1
            left_sample_weight = sample_weight[is_left] if sample_weight else None
            soft_tree_node_classifier.child_left = len(self.soft_tree_node_classifiers)
            self._fit(X[is_left], y[is_left], left_sample_weight)
        # right node
        child_right = tree_node_classifier.child_right
        if impurity[0] > impurity[child_right] and impurity[child_right] > 0.0: 
            child_presence = True
            #print('right')
            is_right = leaves == child_right - 1
            right_sample_weight = sample_weight[is_right] if sample_weight else None
            soft_tree_node_classifier.child_right = len(self.soft_tree_node_classifiers)
            self._fit(X[is_right], y[is_right], right_sample_weight)
        if self.feature_selection:
            feature = X[:, feature_idx:feature_idx+1]
        else:
            feature = X
        if child_presence:
            soft_tree_node_classifier.fit(feature, leaves, sample_weight) 
        else:
            soft_tree_node_classifier.fit(feature, y, sample_weight)
            soft_tree_node_classifier.is_leaf = True
        return self

    def predict_proba(self, X):
        return self._predict_proba(0, X, True)

    def _predict_proba(self, tree_node_idx, X, is_root=False):
        tree_node_classifier = self.tree_node_classifiers[tree_node_idx]
        soft_tree_node_classifier = self.soft_tree_node_classifiers[tree_node_idx]
        feature_idx = soft_tree_node_classifier.feature
        if self.feature_selection:
            feature = X[:, feature_idx:feature_idx+1]
        else:
            feature = X
        soft_tree_node_pred_proba = soft_tree_node_classifier.predict_proba(feature)
        if soft_tree_node_classifier.is_leaf:
            leaf_soft_tree_node_pred_proba = np.zeros((len(soft_tree_node_pred_proba), len(self.classes)))
            for i, c in enumerate(soft_tree_node_classifier.classes):
                leaf_soft_tree_node_pred_proba[:, c] = soft_tree_node_pred_proba[:, i]
            return leaf_soft_tree_node_pred_proba
        left_leaves = soft_tree_node_pred_proba[:, 0]
        child_left = soft_tree_node_classifier.child_left
        #print('left', child_left)
        if child_left is not None:
            child_left_pred_proba = self._predict_proba(child_left, X)
        else:
            raw_child_left_pred_proba = tree_node_classifier.predict_proba(X)
            child_left_pred_proba = np.zeros((len(raw_child_left_pred_proba), len(self.classes)))
            for i, c in enumerate(tree_node_classifier.classes):
                child_left_pred_proba[:, c] = raw_child_left_pred_proba[:, i]
        right_leaves = soft_tree_node_pred_proba[:, 1]
        child_right = soft_tree_node_classifier.child_right
        #print('right', child_right)
        if child_right is not None:
            child_right_pred_proba = self._predict_proba(child_right, X)
        else:
            raw_child_right_pred_proba = tree_node_classifier.predict_proba(X)
            child_right_pred_proba = np.zeros((len(raw_child_right_pred_proba), len(self.classes)))
            for i, c in enumerate(tree_node_classifier.classes):
                child_right_pred_proba[:, c] = raw_child_right_pred_proba[:, i]
        #print(soft_tree_node_pred_proba.shape, child_left_pred_proba.shape, child_right_pred_proba.shape)
        return np.array([
            left_leaves * v for v in child_left_pred_proba.T]).T + np.array([
                right_leaves * v for v in child_right_pred_proba.T]).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y, sample_weight=None):   
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class LogisticRegressionDecisionTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self, tree_node_classifier=None, soft_tree_node_classifier=None, feature_selection=True, weight_fitting=True):
        self.tree_node_classifier = tree_node_classifier if tree_node_classifier else TreeNodeClassifier()
        self.soft_tree_node_classifier = soft_tree_node_classifier if soft_tree_node_classifier else SoftTreeNodeClassifier()
        self.tree_node_classifiers = []
        self.soft_tree_node_classifiers = []
        self.classes = None
        self.feature_selection = feature_selection
        self.weight_fitting = weight_fitting

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        self.classes = np.unique(y)
        self._fit(X, y, sample_weight, check_input, X_idx_sorted, is_root=True)
        #print([clf.child_left for clf in self.soft_tree_node_classifiers])
        return self
        
    def _fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None, is_root=False):
        unique_y = np.unique(y)
        print(unique_y)
        
        tree_node_classifier = copy(self.tree_node_classifier)
        tree_node_classifier.classes = unique_y
        tree_node_classifier.fit(X, y, sample_weight, check_input, X_idx_sorted)
        self.tree_node_classifiers.append(tree_node_classifier)
        feature_idx = tree_node_classifier.feature

        soft_tree_node_classifier = copy(self.soft_tree_node_classifier)
        soft_tree_node_classifier.classes = unique_y
        if self.feature_selection:
            feature = X[:, feature_idx:feature_idx+1]
        else:
            feature = X
        soft_tree_node_classifier.fit(feature, y, sample_weight) 
        soft_tree_node_classifier.feature = feature_idx
        self.soft_tree_node_classifiers.append(soft_tree_node_classifier)
        #print(tree_node_classifier.value, tree_node_classifier.impurity)
        y_pred = soft_tree_node_classifier.predict(feature)
        if self.weight_fitting:
            soft_tree_node_classifier.value = accuracy_score(
                y, y_pred, sample_weight=sample_weight)
        else:
            soft_tree_node_classifier.value = 0.5
        print(soft_tree_node_classifier.value)

        leaves = tree_node_classifier.apply(X)
        impurity = tree_node_classifier.impurity
        # left node
        child_left = tree_node_classifier.child_left
        if impurity[0] > impurity[child_left] and impurity[child_left] > 0.0:
            #print('left')
            is_left = leaves == child_left
            left_sample_weight = sample_weight[is_left] if sample_weight else None
            soft_tree_node_classifier.child_left = len(self.soft_tree_node_classifiers)
            self._fit(X[is_left], y[is_left], left_sample_weight)
        # right node
        child_right = tree_node_classifier.child_right
        if impurity[0] > impurity[child_right] and impurity[child_right] > 0.0: 
            #print('right')
            is_right = leaves == child_right
            right_sample_weight = sample_weight[is_right] if sample_weight else None
            soft_tree_node_classifier.child_right = len(self.soft_tree_node_classifiers)
            self._fit(X[is_right], y[is_right], right_sample_weight)
        return self

    def predict_proba(self, X, dinamic_weight=False):
        return self._predict_proba(0, X, True, dinamic_weight)

    def _predict_proba(self, tree_node_idx, X, is_root=False, dinamic_weight=False):
        soft_tree_node_classifier = self.soft_tree_node_classifiers[tree_node_idx]
        tree_node_classifier = self.tree_node_classifiers[tree_node_idx]
        feature_idx = soft_tree_node_classifier.feature
        if self.feature_selection:
            feature = X[:, feature_idx:feature_idx+1]
        else:
            feature = X
        raw_soft_tree_node_pred_proba = soft_tree_node_classifier.predict_proba(feature)
        soft_tree_node_pred_proba = np.zeros((len(raw_soft_tree_node_pred_proba), len(self.classes)))
        for i, c in enumerate(soft_tree_node_classifier.classes):
            soft_tree_node_pred_proba[:, c] = raw_soft_tree_node_pred_proba[:, i]
        leaves = tree_node_classifier.apply(X)
        left_leaves = np.array(leaves == 1).astype(np.int)
        child_left = soft_tree_node_classifier.child_left
        #print('left', child_left)
        if child_left is not None:
            raw_child_left_pred_proba = self._predict_proba(child_left, X)
            child_left_pred_proba = np.zeros((len(soft_tree_node_pred_proba), len(self.classes)))
            for i, c in enumerate(self.soft_tree_node_classifiers[child_left].classes):
                child_left_pred_proba[:, c] = raw_child_left_pred_proba[:, i]
        else:
            child_left_pred_proba = soft_tree_node_pred_proba
        right_leaves = np.array(leaves == 2).astype(np.int)
        child_right = soft_tree_node_classifier.child_right
        #print('right', child_right)
        if child_right is not None:
            raw_child_right_pred_proba = self._predict_proba(child_right, X)
            child_right_pred_proba = np.zeros((len(soft_tree_node_pred_proba), len(self.classes)))
            for i, c in enumerate(self.soft_tree_node_classifiers[child_right].classes):
                child_right_pred_proba[:, c] = raw_child_right_pred_proba[:, i]
        else:
            child_right_pred_proba = soft_tree_node_pred_proba
        #print(soft_tree_node_pred_proba.shape, child_left_pred_proba.shape, child_right_pred_proba.shape)
        if dinamic_weight:
            weight = np.max(soft_tree_node_pred_proba, axis=1)
            child_pred_proba = np.array([
                left_leaves * v for v in child_left_pred_proba.T]).T + np.array([
                    right_leaves * v for v in child_right_pred_proba.T]).T
            return np.array([
                weight * v for v in soft_tree_node_pred_proba.T]).T + np.array([
                    (1.0 - weight) * v for v in child_pred_proba.T]).T
        else:
            weight = soft_tree_node_classifier.value
            return weight * soft_tree_node_pred_proba + (1.0 - weight) * np.array([
                left_leaves * v for v in child_left_pred_proba.T]).T + np.array([
                    right_leaves * v for v in child_right_pred_proba.T]).T

    def predict(self, X, dinamic_weight=False):
        return np.argmax(self.predict_proba(X, dinamic_weight), axis=1)

    def score(self, X, y, sample_weight=None, dinamic_weight=False):   
        return accuracy_score(y, self.predict(X, dinamic_weight), sample_weight=sample_weight)