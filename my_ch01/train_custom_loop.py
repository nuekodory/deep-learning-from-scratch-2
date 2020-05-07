import numpy as np
from my_ch01.optimizer import StochasticGradientDescent
from dataset import spiral
import matplotlib.pyplot as plt
from my_ch01.two_layer_net import TwoLayerNet


MAX_EPOCH = 300
BATCH_SIZE = 30
HIDDEN_SIZE = 10
LEARNING_LATE = 1.0


x, t = spiral.load_data(57)
model = TwoLayerNet(input_size=2, hidden_size=HIDDEN_SIZE, output_size=3)
optimizer = StochasticGradientDescent(lr=LEARNING_LATE)

data_size = len(x)
max_iters = data_size // BATCH_SIZE
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(0, MAX_EPOCH):
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(0, max_iters):
        batch_x = x[iters*BATCH_SIZE:(iters+1)*BATCH_SIZE]
        batch_t = t[iters*BATCH_SIZE:(iters+1)*BATCH_SIZE]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(f"| epoch {epoch+1} | iter {iters+1} / {max_iters} | loss {avg_loss}")
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

# 学習結果のプロット
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

# 境界領域のプロット
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# データ点のプロット
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()
