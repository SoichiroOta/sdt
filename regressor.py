import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import cosine

from copy import copy

from common import SoftTreeNodeClassifier


class LeafNodeRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self, base_estimator=None):
        self.estimator = base_estimator if base_estimator else linear_model.ElasticNet()
        self.feature = None
        self.value = None
        self.is_leaf = True

    def fit(self, X, y):
        self.estimator.fit(np.asfortranarray(X), np.asfortranarray(y))
        return self

    def predict(self, X):
        return self.estimator.predict(X)


class TreeNodeRegressor(DecisionTreeRegressor):

    def __init__(
        self, criterion='mse', splitter='best', min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
        random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, presort='deprecated', ccp_alpha=0.0):
        super().__init__(
            criterion, splitter, 1, min_samples_split, min_samples_leaf,
            min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
            min_impurity_decrease, min_impurity_split, presort, ccp_alpha)
        self.child_left = None
        self.child_right = None
        self.feature = None
        self.threshold = None
        self.value = None
        self.impurity = None
        self.n_node_samples = None
        self.is_leaf = False


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


class SoftDecisionTreeRegreesor(BaseEstimator, RegressorMixin):

    def __init__(
        self, leaf_node_regressor=None, tree_node_regressor=None, soft_tree_node_classifier=None, feature_selection=True):
        self.leaf_node_regressor = leaf_node_regressor if leaf_node_regressor else LeafNodeRegressor()
        self.tree_node_regressor = tree_node_regressor if tree_node_regressor else TreeNodeRegressor()
        self.soft_tree_node_classifier = soft_tree_node_classifier if soft_tree_node_classifier else SoftTreeNodeClassifier()
        self.tree_node_regressors = []
        self.soft_tree_node_classifiers = []
        self.feature_selection = feature_selection

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        self._fit(X, y, sample_weight, check_input, X_idx_sorted, is_root=True)
        #print([clf.child_left for clf in self.soft_tree_node_classifiers])
        return self
        
    def _fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None, is_root=False):
        mean_y = np.average(y, weights=sample_weight)
        print(y, mean_y)
        
        tree_node_regressor = copy(self.tree_node_regressor)
        tree_node_regressor.value = y
        tree_node_regressor.fit(X, y, sample_weight, check_input, X_idx_sorted)
        feature_idx = tree_node_regressor.feature
        leaves = tree_node_regressor.apply(X) - 1

        soft_tree_node_classifier = copy(self.soft_tree_node_classifier)
        soft_tree_node_classifier.value = y
        soft_tree_node_classifier.feature = feature_idx
        if self.feature_selection:
            feature = X[:, feature_idx:feature_idx+1]
        else:
            feature = X
        soft_tree_node_classifier.fit(feature, leaves, sample_weight) 
        self.soft_tree_node_classifiers.append(soft_tree_node_classifier)
        #print(tree_node_classifier.value, tree_node_classifier.impurity)

        impurity = tree_node_regressor.impurity
        n_node_samples = tree_node_regressor.n_node_samples
        # left node
        child_left = tree_node_regressor.child_left 
        is_left = leaves == child_left - 1
        left_sample_weight = sample_weight[is_left] if sample_weight else None
        soft_tree_node_classifier.child_left = len(self.soft_tree_node_classifiers)
        if impurity[0] > impurity[child_left] and impurity[child_left] > 0.0 and n_node_samples[child_left] > 0:
            #print('left')
            self._fit(X[is_left], y[is_left], left_sample_weight)
        else:
            tree_node_regressor.is_leaf = True
            leaf_node_regressor = copy(self.leaf_node_regressor)
            leaf_node_regressor.feature = feature_idx
            leaf_node_regressor.fit(feature, y)
            self.soft_tree_node_classifiers.append(leaf_node_regressor)
        # right node
        child_right = tree_node_regressor.child_right
        is_right = leaves == child_right - 1
        right_sample_weight = sample_weight[is_right] if sample_weight else None
        soft_tree_node_classifier.child_right = len(self.soft_tree_node_classifiers)
        if impurity[0] > impurity[child_right] and impurity[child_right] > 0.0 and n_node_samples[child_right] > 0:
            #print('right')
            self._fit(X[is_right], y[is_right], right_sample_weight)
        else:
            leaf_node_regressor = copy(self.leaf_node_regressor)
            leaf_node_regressor.feature = feature_idx
            leaf_node_regressor.fit(feature, y)
            self.soft_tree_node_classifiers.append(leaf_node_regressor)
        return self

    def predict(self, X):
        return self._predict(0, X, True)

    def _predict(self, tree_node_idx, X, is_root=False):
        soft_tree_node_classifier = self.soft_tree_node_classifiers[tree_node_idx]
        feature_idx = soft_tree_node_classifier.feature
        if self.feature_selection:
            feature = X[:, feature_idx:feature_idx+1]
        else:
            feature = X
        if tree_node_idx < len(self.soft_tree_node_classifiers):
            return soft_tree_node_classifier.predict(feature)
        else:
            soft_tree_node_pred_proba = soft_tree_node_classifier.predict_proba(feature)
            tree_node_regressor = self.tree_node_regressors[tree_node_idx]
        left_leaves = soft_tree_node_pred_proba[:, 0]
        child_left = soft_tree_node_classifier.child_left
        #print('left', child_left)
        if child_left is not None:
            child_left_pred = self._predict(child_left, X)
        else:
            child_left_pred = tree_node_regressor.predict(X)
        right_leaves = soft_tree_node_pred_proba[:, 1]
        child_right = soft_tree_node_classifier.child_right
        #print('right', child_right)
        if child_right is not None:
            child_right_pred = self._predict(child_right, X)
        else:
            child_right_pred = tree_node_regressor.predict(X)
        #print(soft_tree_node_pred_proba.shape, child_left_pred_proba.shape, child_right_pred_proba.shape)
        return np.array([
            left_leaves * v for v in child_left_pred.T]).T + np.array([
                right_leaves * v for v in child_right_pred.T]).T

    def score(self, X, y, sample_weight=None):   
        return r2_score(y, self.predict(X), sample_weight=sample_weight)