import random

from perceptron import Perceptron


class Trainer:
    def __init__(self, learning_rate, target_error, train_set, test_set, ):
        self.perceptron = None
        self.learning_rate = learning_rate
        self.train_set = train_set
        self.test_set = test_set
        self.target_error = target_error

    def initialize_perceptron(self, desired, activation_function):
        self.perceptron = Perceptron(len(self.train_set[0]), desired, activation_function)

    def teach(self, max_epoch=1000):
        epoch = 0
        error = self.target_error
        while error >= self.target_error and epoch < max_epoch:
            random.shuffle(self.train_set)
            weight_updates = [0] * self.perceptron.dimension
            for values in self.train_set:
                updates = self.perceptron.learn(values[:-1], values[-1], self.learning_rate)
                weight_updates = [weight_updates[i] + updates[i] for i in range(self.perceptron.dimension)]

            self.perceptron.update_weights(weight_updates)

            error = self.__calc_error()
            epoch += 1
        print(self.perceptron.weights)
        print(f'{epoch}: {error * 100}%')

    def __calc_error(self):
        error = 0

        for values in self.test_set:
            d = self.perceptron.desired_to_numb(values[-1])
            y = self.perceptron.compute_out(values[:-1])
            error += (d - y) ** 2
        return error / len(self.test_set)
