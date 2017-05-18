import numpy as np

class SimpleNN(object):

    def __init__(self, sizes):
        if isinstance(sizes, list):
            self.num_layers = len(sizes)
            self.num_neurons = sizes
            self.biases = [np.random.randn(y, 1) for y in self.num_neurons[1:]]
            self.weights = [np.random.randn(y, x) for x, y, in zip(self.num_neurons[:-1], self.num_neurons[1:])]

    def get(self, a):
        """
        :param a:  input that will be fed through the network
                   the input must be a list of the same length as the number of inputs in the network.
        :return:   the value produced by feeding a through the network. In other words, the output of the network.
        """
        return self.feed_forward(a)
    def feed_forward(self, a):
        """
        :param a:  input that will be fed through the network
                   the input must be a list of the same length as the number of inputs in the network.
        :return:   the value produced by feeding a through the network. In other words, the output of the network.
        """
        if (len(a) == self.sizes[0]):
            if not isinstance(a[0], list):
                a = [[i] for i in a]
            for b, w in zip(self.biases[:-1], self.weights[:-1]):
                a = self.activation(np.dot(w, a) + b)
            a = self.output_activation(np.dot(self.weights[-1], a) + self.biases[-1])
            return a
        return "Invalid Input Error"

    def activation(self, z, w = 1, b = 0):
        """
        The activation function used is a Leaky ReLU function.

        :param z:  the input for the function.
        :param w:  an optional weight that will be multiplied to z
        :param b:  an optional bias that will be added to z
        :return:  the value produced by applying the activation function to z.
        """
        if isinstance(w, list):
            z = np.dot(w, z)
        if isinstance(b, list):
            z = np.array(z)+b

        def is_negative(num):
            if num < 0:
                return 1
            return 0
        def ReLUmax(i):
            return np.max(0, i)+0.015625*is_negative(i)

        return map(ReLUmax, z)

    def output_activation(self, z, w = 1, b = 0):
        """
        The activation function for the final layer of the network will be the softmax activation function

        :param z:  the input for the function.
        :param w:  an optional weight that will be multiplied to z
        :param b:  an optional bias that will be added to z
        :return:  the value produced by applying the activation function to z.
        """
        sum = np.sum(np.exp(z));
        def cross_entropy(i):
            return -np.log(np.exp(i) / sum)
        return map(cross_entropy(), z)

    def train(self, training_input, expected_output, epochs, mini_batch_size, eta):
        """
        Uses Mini-batch gradient descent

        :param training_input:  the training data that will be fed forward through the network
        :param expected_ouput:  the output that the input training data is expected to produce
        :param epochs:          the number of times the training sequence is run
        :param mini_batch_size: the size of the mini-batch
        :param eta:             the learning rate
        """
