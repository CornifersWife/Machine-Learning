import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from Plot import plot_decision_regions


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            # errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def main():
    iris = datasets.load_iris()
    print(iris.data)
    X = iris.data[:, [2, 3]]
    print(X)
    y = iris.target
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)] y_train_01_subset = y_train[(y_train == 0) | (
    # y_train == 1)]
    # w perceptronie wyjście jest albo 1 albo -1
    # y_train_01_subset[(y_train_01_subset == 0)] = -1
    y_train[y_train != 2] = -1
    y_train[y_train == 2] = 1
    print(y_train)

    ppn = Perceptron(eta=0.1, n_iter=1000)
    ppn.fit(X_train, y_train)

    plot_decision_regions(X=X_train, y=y_train, classifier=ppn)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()