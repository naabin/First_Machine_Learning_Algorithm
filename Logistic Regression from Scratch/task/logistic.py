# write your code here
import math
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = [0 for _ in range(len(x_train.columns))]
        self.mse_error_first = []
        self.log_loss_error_first = []
        self.mse_error_last = []
        self.log_loss_error_last = []
        if self.fit_intercept:
            self.coef_.append(0)

    def sigmoid(self, t):
        return 1/(1 + math.pow(math.e, -t))

    def predict_proba(self, row, coef_):
        t = 0
        if self.fit_intercept:
            t += coef_[0]
            t += np.dot(row, coef_[1:])
            return self.sigmoid(t)
        else:
            t = np.dot(row, coef_)
            return self.sigmoid(t)

    def fit_mse(self, x_train, y_train):
        N = len(x_train)
        for e in range(self.n_epoch):
            for i, r in x_train.iterrows():
                y_hat = self.predict_proba(r.values, self.coef_)
                if e == 0:
                    self.mse_error_first.append(((y_hat - y_train[i])**2)/N)
                if e == self.n_epoch - 1:
                    self.mse_error_last.append(((y_hat - y_train[i])**2)/N)
                derivative = self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                index = 0
                if self.fit_intercept:
                    self.coef_[index] -= derivative
                    index += 1
                for j in range(len(r.values)):
                    self.coef_[index] -= derivative * r.values[j]
                    index += 1
        return self

    def predict(self, x_test, cut_off=0.5):
        predicted_probabilities = []
        for i, row in x_test.iterrows():
            y_hat = self.predict_proba(row.values, self.coef_)
            if y_hat < cut_off:
                predicted_probabilities.append(0)
            else:
                predicted_probabilities.append(1)
        return np.array(predicted_probabilities)

    def fit_log_loss(self, x_train, y_train):
        N = len(x_train)
        for e in range(self.n_epoch):
            for i, r in x_train.iterrows():
                y_hat = self.predict_proba(r.values, self.coef_)
                err = -((y_train[i]*np.log(y_hat)) + (1 - y_train[i]) * (np.log(1 - y_hat)))/N
                if e == 0:
                    self.log_loss_error_first.append(err)
                if e == self.n_epoch - 1:
                    self.log_loss_error_last.append(err)
                derivative = self.l_rate * (y_hat - y_train[i]) / N
                index = 0
                if self.fit_intercept:
                    self.coef_[index] -= derivative
                    index += 1
                for j in range(len(r.values)):
                    self.coef_[index] -= derivative * r.values[j]
                    index += 1
        return self


def standardize_features(x):
    m = np.mean(x)
    st_dev = np.std(x)
    for i in range(len(x)):
        x[i] = (x[i] - m)/st_dev
    return x


df, y = load_breast_cancer(return_X_y=True, as_frame=True)
features = ['worst concave points', 'worst perimeter', 'worst radius']
target = y
df = df[features]
df[features[0]] = standardize_features(df[features[0]])
df[features[1]] = standardize_features(df[features[1]])
df[features[2]] = standardize_features(df[features[2]])
x_train, x_test, y_train, y_test = train_test_split(df, target, train_size=0.8, random_state=43)
m1 = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
m2 = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
sklearn_regression_model = LogisticRegression(fit_intercept=True)
sklearn_regression_model.fit(x_train, y_train)
m1 = m1.fit_log_loss(x_train, y_train)
m2 = m2.fit_mse(x_train, y_train)
m1_pred = m1.predict(x_test)
m2_pred = m2.predict(x_test)
sk_pred = sklearn_regression_model.predict(x_test)

m1_log_loss_accuracy = accuracy_score(y_test, m1_pred)
m2_mse_accuracy = accuracy_score(y_test, m2_pred)
sk_accuracy = accuracy_score(y_test, sk_pred)
result = {
    'mse_accuracy': m2_mse_accuracy,
    'logloss_accuracy': m1_log_loss_accuracy,
    'sklearn_accuracy': sk_accuracy,
    'mse_error_first': m2.mse_error_first,
    'mse_error_last': m2.mse_error_last,
    'logloss_error_first': m1.log_loss_error_first,
    'logloss_error_last': m1.log_loss_error_last
}
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
ax1.plot(m2.mse_error_first)
ax1.set_title('MSE: First Epoch Errors')
ax2.plot(m2.mse_error_last)
ax2.set_title('MSE: Last Epoch Errors')
ax3.plot(m1.log_loss_error_first)
ax3.set_title('Log-Loss: First Epoch Errors')
ax4.plot(m1.log_loss_error_last)
ax4.set_title('Log-Loss: Last Epoch Errors')
plt.show()
print(result)
print('Answers to the questions:')
print(f'1) {format(min(m2.mse_error_first),".5f")}')
print(f'2) {format(min(m2.mse_error_last),".5f")}')
print(f'3) {format(max(m1.log_loss_error_first),".5f")}')
print(f'4) {format(max(m1.log_loss_error_last),".5f")}')
print(f'5) expanded')
print(f'6) expanded')













