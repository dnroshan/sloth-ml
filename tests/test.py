from linear_regression import LinearRegression
import numpy as np

lg = LinearRegression()
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + 0.6 * X[:, 1] - 1.2 * X[:, 2] + 0.45

print(y)
lg.fit(X, y)
y_hat = lg.predict(X)
print(lg.params)
print(lg.losses)
print(y_hat)
