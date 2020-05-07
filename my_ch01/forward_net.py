from typing import Dict, Tuple, Iterable, Sequence, Optional
import numpy as np


class Sigmoid:
    def __init__(self):
        self.params, self.grads = (), ()
        self.out: Optional[np.ndarray] = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, w, b):
        self.params: Tuple[np.ndarray, ...] = (w, b)
        self.grads: Tuple[np.ndarray, ...] = (np.zeros_like(w), np.zeros_like(b))
        self.x: Optional[np.ndarray] = None

    def forward(self, x):
        w, b = self.params
        out = np.dot(x, w) + b
        self.x = x
        return out

    def backward(self, dout):
        w, b = self.params
        dx = np.dot(dout, w.T)
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dw
        self.grads[1][...] = db

        return dx


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        input_size, hidden_size, output_size = input_size, hidden_size, output_size

        w1 = np.random.randn(input_size, hidden_size)
        b1 = np.random.randn(hidden_size)
        w2 = np.random.randn(hidden_size, output_size)
        b2 = np.random.randn(output_size)

        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2),
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == '__main__':
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
