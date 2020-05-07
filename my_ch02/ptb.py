import sys
import numpy as np
from sklearn.utils.extmath import randomized_svd

sys.path.append("..")
from dataset import ptb
from common.util import most_similar, create_co_matrix, ppmi


corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size, 2)
W = ppmi(C, verbose=True)

U, S, V = randomized_svd(W, n_components=100, n_iter=5, random_state=None)
word_vecs = U[:, :100]

print(f"corpus size: {len(corpus)}")
print(f"corpus[:30]: {corpus[:30]}")
print(f"")
queries = ["you", "year", "car", "toyota", "amazing"]

for q in queries:
    most_similar(q, word_to_id, id_to_word, word_vecs)


