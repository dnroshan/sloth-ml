# linear_regression.py
#
# Copyright 2022 Dilnavas Roshan <dilnavasroshan@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, iterations=1000, epsilon=0.005):
        self.lr = lr
        self.losses = []
        self.prev_loss = None
        self.epsilon = epsilon
        self.iterations = iterations

    def fit(self, X, y):
        self.ninsta = X.shape[0]
        self.nfeat = X.shape[1]
        self.params = np.random.rand(self.nfeat + 1)  # shape: nfeat x 1
        ones = np.ones(self.ninsta)  # shape:
        X_ = np.c_[ones, X]
        for _ in range(self.iterations):
            y_hat = self.predict(X)
            loss = self.loss(y, y_hat)
            self.losses.append(loss)
            if self.prev_loss and (loss - self.prev_loss) < self.epsilon:
                break
            delta_y = y_hat - y
            self.params = self.params - self.lr * X_.T.dot(delta_y)

    def predict(self, X):
        ones = np.ones(self.ninsta)  # shape:
        X_ = np.c_[ones, X]
        return X_.dot(self.params)

    def loss(self, y, y_hat):
        return np.sum((y - y_hat) ** 2) / self.ninsta
