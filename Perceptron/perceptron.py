import random


class Perceptron:

    def __init__(self, dimension, desired, activation_function):
        self.activation_function = activation_function
        self.dimension = dimension
        self.weights = [random.uniform(0, 0.1) for _ in range(dimension)]
        self.desired = desired

    def compute_out(self, inputs):
        inputs.append(-1)
        net = 0
        for i in range(self.dimension):
            net += inputs[i] * self.weights[i]
        return self.activation_function(net)

    def learn(self, inputs, desired_out, learning_rate):
        d = self.desired_to_numb(desired_out)
        y = self.compute_out(inputs)
        inputs.append(-1)
        weight_updates = [learning_rate * (d - y) * inputs[i] for i in range(self.dimension)]
        return weight_updates

    def update_weights(self, weight_updates):
        for i in range(self.dimension):
            self.weights[i] += weight_updates[i]

    def desired_to_numb(self, desired):
        return 1 if desired == self.desired else 0
