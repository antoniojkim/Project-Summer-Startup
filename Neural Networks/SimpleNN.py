import numpy as np

class SimpleNN(object):

    def __init__(self, sizes):
        if isinstance(sizes, list):
            self.num_layers = len(sizes)
            self.num_neurons = sizes
            self.reg = 0.01
            self.biases = [np.zeros((1, y)) for y in self.num_neurons[1:]]
            self.weights = [np.random.randn(x, y)/np.sqrt(x) for x, y, in zip(self.num_neurons[:-1], self.num_neurons[1:])]

            # self.activation = self.Leaky_ReLU
            # self.activation_prime = self.Leaky_ReLU_Prime
            # self.output_activation = self.Softmax
            # self.output_activation_prime = self.Softmax_prime
            self.activation = self.tanh
            self.activation_prime = self.tanh_prime
            self.output_activation = self.tanh
            self.output_activation_prime = self.tanh_prime
            # self.loss = self.cross_entropy_loss
            # self.loss_prime = self.cross_entropy_loss_prime
            self.cost = self.mean_squared_loss
            self.cost_prime = self.mean_squared_loss_prime

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
        if (len(a[0]) == self.num_neurons[0]):
            if not isinstance(a[0], list):
                a = [[i] for i in a]
            for b, w in zip(self.biases[:-1], self.weights[:-1]):
                a = self.activation(np.dot(a, w) + b)
            a = self.output_activation(np.dot(a, self.weights[-1]) + self.biases[-1])
            return a
        return "Invalid Input Error"

    # Different Activation Functions
    def sigmoid(self, z):
        """
        The sigmoid activation function

        :param z:   The input of the function
        :return:    Produces the value produced by applying the sigmoid function on z
        """
        return 1.0/(1.0+np.exp(-z))
    def sigmoid_prime(self, z):
        """
        The derivative of the sigmoid activation function

        :param z:   The input of the function
        :return:    Produces the value produced by applying the derivative of the sigmoid function on z
        """
        sigmoid = self.sigmoid(z)
        return sigmoid*(1-sigmoid)
    def tanh(self, z):
        """
        The tanh activation function

        :param z:   The input of the function
        :return:    Produces the value produced by applying the tanh function on z
        """
        return np.tanh(z)
    def tanh_prime(self, z):
        """
        The derivative of the tanh activation function

        :param z:   The input of the function
        :return:    Produces the value produced by applying the derivative of the tanh function on z
        """
        return 1.0/(np.cosh(z)**2)
    def Leaky_ReLU(self, z):
        """
        The Leaky ReLU activation function.

        :param z:  the input for the function.
        :return:  the value produced by applying the Leaky ReLU function on z.
        """
        return np.maximum(z, 0.05*z)
    def Leaky_ReLU_Prime(self, z):
        """
        The derivative of the Leaky ReLU activation function.

        :param z:  the input for the function.
        :return:  the value produced by applying the derivative of the Leaky ReLU function on z.
        """
        def prime(i):
            if isinstance(i, list):
                return self.activation_prime(i)
            return (1 if i > 0 else 0)
        return list(map(prime, z))
    def Softmax(self, z):
        """
        The Softmax activation function.

        :param z:  the input for the function.
        :return:  the value produced by applying the Softmax activation function on z.
        """
        exp = np.exp(z - z.max())
        return exp/np.sum(exp)
        # return z
    def Softmax_prime(self, z):
        """
        The derivative of the Softmax activation function.

        :param z:  the input for the function.
        :return:  the value produced by applying the derivative of the Softmax activation function on z.
        """
        z = z-z.max()
        exp = np.exp(z - z.max())
        sum = np.sum(exp)
        sumsqr = sum**2
        return [(sum-i)/sumsqr for i in exp]

    def train(self, training_input, expected_output, epochs, interval=0, eta=0.5):
        """
        Uses Gradient descent

        :param training_input:  the training data that will be fed forward through the network
        :param expected_ouput:  the output that the input training data is expected to produce
        :param epochs:          the number of times the training sequence is run
        :param interval:        how often the loss should be printed
        :param eta:             the learning rate
        """
        if interval == 0:
            for i in range(epochs):
                self.gradient_descent(training_input, expected_output, eta)
        else:
            for i in range(epochs+1):
                loss = self.gradient_descent(training_input, expected_output, eta)
                if i%interval == 0:
                    print("Epoch", i, ":   Loss: ", loss)
    def gradient_descent(self, training_input, expected_output, eta=0.5):
        """
        Backpropogation using Gradient Descent

        :param training_input:  The set of training data
        :param expected_output: The expected output of given training data
        :param eta:  The learning Rate
        :return:  The Loss of the function
        """
        training_input = np.array(training_input)
        expected_output = np.array(expected_output)

        integrations = []
        activations = [training_input]
        activation_primes = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            integrations.append(np.dot(activations[-1], w) + b)
            activations.append(self.activation(integrations[-1]))
            activation_primes.append(self.activation_prime(integrations[-1]))
        integrations.append(np.dot(activations[-1], self.weights[-1]) + self.biases[-1])
        activations.append(self.output_activation(integrations[-1]))
        activation_primes.append(self.output_activation_prime(integrations[-1]))

        output = activations[-1]

        loss = self.cost(expected_output, output)

        delta = np.multiply(self.cost_prime(expected_output, output), activation_primes[-1])
        # dBiases = [delta]
        dWeights = [np.dot(activations[-2].T, delta)]
        for i in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-i+1].T) * activation_primes[-i]
            # dBiases.append(delta)
            dWeights.append(np.dot(activations[-i-1].T, delta))

        # dBiases.reverse()
        dWeights.reverse()

        self.weights = [w - eta * dw for w, dw in zip(self.weights, dWeights)]
        # self.biases = [b - eta * db for b, db in zip(self.biases, dBiases)]

        return loss

    # Different Cost Functions
    def mean_squared_loss(self, expected, output):
        """
        The mean squared cost function

        :param expected:    The expected output of the network
        :param output:      The actual output produced by the network
        :return:            The mean squared cost
        """
        return 0.5*np.sum((output-expected)**2)
    def mean_squared_loss_prime(self, expected, output):
        """
        The derivative of the mean squared cost function

        :param expected:    The expected output of the network
        :param output:      The actual output produced by the network
        :return:            The derivative of the mean squared cost
        """
        return (output-expected)
    def cross_entropy_loss(self, expected, output):
        """
        The Cross Entropy cost function

        :param expected:    The expected output of the network
        :param output:      The actual output produced by the network
        :return:            The Cross Entropy cost
        """
    def cross_entropy_loss_prime(self, expected, output):
        """
        The derivative of the Cross Entropy cost function

        :param expected:    The expected output of the network
        :param output:      The actual output produced by the network
        :return:            The derivative of the Cross Entropy cost
        """



# Simple Test for SimpleNN

training_input = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
expected_output = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

nn = SimpleNN([3, 5, 7, 5, 1])

print(nn.get(training_input))

nn.train(training_input, expected_output, 2000, 200, 0.5)

print(nn.get(training_input))