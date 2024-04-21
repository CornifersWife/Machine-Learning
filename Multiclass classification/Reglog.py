import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn import datasets
from sklearn.model_selection import train_test_split

from Plot import plot_decision_regions


def activation(z):
    return softmax(z, axis=1)


class LogisticRegressionGradientDescent:
    def __init__(self, eta=0.05, n_iter=100, random_state=1, num_classes=3):
        self.w_ = None
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.num_classes = num_classes

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(X.shape[1] + 1, self.num_classes))
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum(axis=0)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # noinspection PyTypeChecker
        return np.argmax(activation(self.net_input(X)), axis=1)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    class_labels = iris.target_names
    num_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    Y_train_onehot = np.eye(num_classes)[y_train]

    logisticRegressionGradientDescent = LogisticRegressionGradientDescent(eta=0.05, n_iter=500, random_state=1, num_classes=num_classes)
    logisticRegressionGradientDescent.fit(X_train, Y_train_onehot)

    plot_decision_regions(X=X_train, y=y_train, classifier=logisticRegressionGradientDescent, class_labels=class_labels)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.title('Logistic Regression - Multiclass Classification')
    plt.show()


if __name__ == '__main__':
    main()
