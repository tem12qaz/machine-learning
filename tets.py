import numpy as np
from numpy import e


class Activation:
    @staticmethod
    def activation(x) -> int:
        pass

    @classmethod
    def prime(cls, x) -> int:
        pass

    def __repr__(self):
        return 'sigma'


class Sigmoid(Activation):
    @staticmethod
    def activation(x):
        return 1 / (1 + e**-x)

    @classmethod
    def prime(cls, x):
        return cls.activation(x) * (1 - cls.activation(x))

    def __repr__(self):
        return 'sigma'


class Max(Activation):
    @staticmethod
    def activation(x):
        if x > 0:
            return x
        else:
            return 0

    @classmethod
    def prime(cls, x):
        if x > 0:
            return 1
        else:
            return 0

    def __repr__(self):
        return 'max'


class Neuron:
    def __init__(self, weights: list, bias, activation: Activation = Sigmoid):
        self.bias = bias
        self.weights = weights
        self.activation = activation.activation
        self.prime = activation.prime


class Network:
    def __init__(self, neurons: list[list[Neuron]], inputs: int):
        self.layers = []
        self.funcs = []
        self.primes = []
        self.biases = []
        self.weights = []
        self.activations = []
        self.summaries = []
        for layer in neurons:
            self.layers.append(layer)
            self.biases.append(np.array([[neuron.bias] for neuron in layer]))
            self.funcs.append(np.array([[neuron.activation] for neuron in layer]))
            self.primes.append(np.array([[neuron.prime] for neuron in layer]))
            weights = []
            for neuron in layer:
                weights.append(neuron.weights)
            self.weights.append(np.array(weights))

    @staticmethod
    def activate(func, data):
        return np.array([[func[i, :][0](data[i, :][0])] for i in range(func.shape[0])])

    def forward_one_layer(self, data: np.ndarray, layer: int):
        summary = (self.weights[layer] @ data) + self.biases[layer]
        self.summaries.append(summary)
        return self.activate(self.funcs[layer], summary)

    def forward_propagation(self, data: np.ndarray):
        self.activations = []
        for i in range(len(self.layers)):
            activation = self.forward_one_layer(data, i)
            self.activations.append(activation)
            data = activation
        # print(data)

    def result_error(self, result):
        error = (self.activations[-1] - result) * self.activate(self.primes[-1], self.summaries[-1])
        return error

    def back_one_layer(self, layer, err_l1):
        error = (self.weights[layer + 1].T * err_l1) * self.activate(self.primes[layer], self.summaries[layer])
        return error

    def back_propagation(self, result):
        result_error = self.result_error(result)
        errs = [result_error]
        for layer in range(len(self.layers)-2, -1, -1):
            errors = self.back_one_layer(layer, errs[0])
            errs.insert(0, errors)
        return errs

    def one_step(self, inputs, result):
        self.forward_propagation(inputs)
        errs = self.back_propagation(result)
        print(errs)

    def __repr__(self):
        print('weights')
        print(self.weights)
        print('biases')
        print(self.biases)
        print('activations')
        print(self.funcs)


if __name__ == '__main__':
    n = Neuron
    nn = Network(
        [
            [n([0.7, 0.2, 0.7], 0, Max), n([0.8, 0.3, 0.6], 0)],
            [n([0.2, 0.4], 0)]
        ],
        inputs=3
    )
    nn.one_step(np.array([[0], [1], [1]]), np.array([[1]]))






