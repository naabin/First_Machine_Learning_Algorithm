import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LinearRegression
import sklearn.metrics as sk_metrics


class CustomLinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, X, y):
        a = np.array(np.ones(len(X)))
        new_x = X
        if self.fit_intercept:
            new_x = np.column_stack((a, new_x))
        a_inv = np.linalg.inv(new_x.T @ new_x)
        b = new_x.T @ y
        self.coefficient = a_inv @ b
        if self.fit_intercept:
            self.intercept = self.coefficient[0]
            self.coefficient = self.coefficient[1:]

    def predict(self, X):
        if self.fit_intercept:
            return (X @ self.coefficient) + self.intercept
        return X @ self.coefficient

    def r2_score(self, y, y_hat):
        y_mean = y.mean()
        nom = 0
        denom = 0
        for z in range(len(y)):
            nom += math.pow((y[z] - y_hat[z]), 2)
            denom += math.pow((y[z] - y_mean), 2)
        return 1 - (nom / denom)

    def rmse(self, y, y_hat):
        res = 0
        for m in range(len(y)):
            res += math.pow((y[m] - y_hat[m]), 2)
        res = res / len(y)
        return math.sqrt(res)


if __name__ == '__main__':
    l_r = CustomLinearRegression(fit_intercept=True)
    data = {
        'f1': [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87],
        'f2': [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 66.6, 96.1, 100, 85.9, 94.3],
        'f3': [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2, 15.2],
        'y': [24, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15]
    }
    sk_lr = LinearRegression(fit_intercept=True)
    df = pd.DataFrame(data)
    independent_v = df[['f1', 'f2', 'f3']]
    target_v = df['y']
    l_r.fit(independent_v, target_v)
    sk_lr.fit(independent_v, target_v)
    y_hat = l_r.predict(independent_v)
    sk_y_hat = sk_lr.predict(independent_v)
    r2_score = l_r.r2_score(target_v, y_hat)
    rmse = l_r.rmse(target_v, y_hat)
    sk_r2_score = sk_metrics.r2_score(target_v, sk_y_hat)
    sk_mse = math.sqrt(sk_metrics.mean_squared_error(target_v, sk_y_hat))
    result = {
        'Intercept': abs(l_r.intercept - sk_lr.intercept_),
        'Coefficient': abs(l_r.coefficient - sk_lr.coef_),
        'R2': abs(r2_score - sk_r2_score),
        'RMSE': abs(rmse - sk_mse)
    }

    print(result)
