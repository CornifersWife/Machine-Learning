import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax #TODO ask if i can just do it like this
from sklearn import datasets
from sklearn.model_selection import train_test_split

from Plot import plot_decision_regions


def activation(z):
    return softmax(z, axis=1)


class LogisticRegressionGradientDescent:
    def __init__(self, eta=0.05, number_of_iterations=100, random_state=1, number_of_classes=3):
        self.weights = None
        self.eta = eta
        self.number_of_iterations = number_of_iterations
        self.random_state = random_state
        self.number_of_classes = number_of_classes

    def fit(self, X, y):
        random_generation = np.random.RandomState(self.random_state)
        self.weights = random_generation.normal(scale=0.01, size=(X.shape[1] + 1, self.number_of_classes))
        for i in range(self.number_of_iterations):
            net_input = self.net_input(X)
            output = activation(net_input)
            errors = y - output
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum(axis=0)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        # noinspection PyTypeChecker
        return np.argmax(activation(self.net_input(X)), axis=1)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    class_labels = iris.target_names
    number_of_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    Y_train_onehot = np.eye(number_of_classes)[y_train]

    logisticRegressionGradientDescent = LogisticRegressionGradientDescent(eta=0.05, number_of_iterations=500,
                                                                          number_of_classes=number_of_classes)
    logisticRegressionGradientDescent.fit(X_train, Y_train_onehot)

    print(logisticRegressionGradientDescent.weights)
    plot_decision_regions(X=X_test, y=y_test, classifier=logisticRegressionGradientDescent, class_labels=class_labels)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.title('Logistic Regression - Multiclass Classification')
    plt.show()


if __name__ == '__main__':
    main()
