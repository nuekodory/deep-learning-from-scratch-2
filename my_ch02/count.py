import numpy as np
import matplotlib.pyplot as plt
import common.util


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = common.util.preprocess(text)
vocab_size = len(id_to_word)
c = common.util.create_co_matrix(corpus, vocab_size, window_size=1)
w = common.util.ppmi(c)

u, s, v = np.linalg.svd(w)

print(u[0])

for word, word_id in word_to_id.items():
    plt.annotate(word, (u[word_id, 0], u[word_id, 1]))

plt.scatter(u[:, 0], u[:, 1], alpha=0.5)
plt.show()
