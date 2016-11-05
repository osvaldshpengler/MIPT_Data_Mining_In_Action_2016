#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import ClassifierMixin, BaseEstimator
from scipy.special import expit


class BinaryBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators, lr=0.1, max_depth=3):
        self.base_regressor = DecisionTreeRegressor(criterion='friedman_mse',
                                                    splitter='best',
                                                    max_depth=max_depth)
        self.lr = lr
        self.n_estimators = n_estimators
        self.feature_importances_ = None
        self.estimators_ = []

    def loss_grad(self, original_y, pred_y):
        grad = original_y * expit(-original_y * pred_y)

        return grad

    def fit(self, X, original_y):
        base_est = BaseEstimator()
        base_est.predict = lambda X: np.zeros(X.shape[0], dtype=float)
        self.estimators_ = [base_est]

        for i in range(self.n_estimators):
            grad = self.loss_grad(original_y, self._predict(X))
            estimator = deepcopy(self.base_regressor)
            estimator.fit(X, grad)

            self.estimators_.append(estimator)

        self.out_ = self._outliers(grad)
        self.feature_importances_ = self._calc_feature_imps()

        return self

    def _predict(self, X):
        y_pred = np.sum(map(lambda est: self.lr*est.predict(X), self.estimators_), axis=0)

        return y_pred

    def predict(self, X):
        y_pred = np.sign(self._predict(X))

        return np.array(y_pred)

    def _outliers(self, grad):
        sorted_idx = grad.argsort()
        _outliers = sorted_idx[0:10].tolist() + sorted_idx[-10:].tolist()

        return _outliers

    def _calc_feature_imps(self):
        f_imps = self.estimators_[1].feature_importances_
        for est in self.estimators_[2:]:
            f_imps += est.feature_importances_

        return f_imps / len(self.estimators_)
