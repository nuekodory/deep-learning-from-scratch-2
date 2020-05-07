import numpy as np
from common.layers import SoftmaxWithLoss
from my_ch01.forward_net import Affine, Sigmoid


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        input_size, hidden_size, output_size = input_size, hidden_size, output_size

        w1 = 0.01 * np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        w2 = 0.01 * np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)

        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2)
        ]

        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
