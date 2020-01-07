from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class SoftTreeNodeRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self, base_estimator=None):
        self.estimator = base_estimator if base_estimator else linear_model.ElasticNet()
        self.child_left = None
        self.child_right = None
        self.feature = None
        self.value = None
        self.classes = None
        self.is_leaf = False

    def fit(self, X, y):
        self.estimator.fit(np.asfortranarray(X), np.asfortranarray(y))
        return self

    def predict(self, X):
        return np.clip(self.estimator.predict(X), 0.0, a_max=None)                                                              


class SoftTreeNodeClassifier(linear_model.LogisticRegression):

    def __init__(
        self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
        intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
        max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None):
        super().__init__(
            penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight,
            random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs)
        self.child_left = None
        self.child_right = None
        self.feature = None
        self.value = None
        self.classes = None
        self.is_leaf = False
