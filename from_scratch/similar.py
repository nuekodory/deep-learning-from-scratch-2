import sys
from pathlib import Path
from typing import List, Dict
import numpy as np


def main():
    word_vecs_size = 100
    args = sys.argv
    input_path = Path(args[1])
    query_path = Path(args[2])

    print("reading words...")
    words = read_all(input_path)

    print("creating corpus...")
    corpus, word2id, id2word = create_corpus(words)

    print("calculating PPMI...")
    co_matrix = create_co_matrix(corpus, len(word2id), window_size=2)
    ppmi_matrix = ppmi(co_matrix)

    print("calculating svd...")
    matrix_u, matrix_s, matrix_v = svd(ppmi_matrix, n_components=word_vecs_size)

    word_vecs = matrix_u[:, :word_vecs_size]

    with query_path.open() as f:
        for line in f:
            query = line.strip()
            print(f"\n{query} :")
            print_most_similar(word_vecs=word_vecs, word2id=word2id, id2word=id2word,
                               query=query, n_best=3)


def read_all(path: Path) -> List[str]:
    words = []
    with path.open() as f:
        for line in f:
            line.strip()
            w = line.split(' ')
            words.extend(w)

    return words


def create_corpus(words: List[str]) -> (np.ndarray, Dict[str, int], Dict[int, str]):
    word2id = {}
    id2word = {}
    corpus = []

    for word in words:
        if word in word2id.keys():
            corpus.append(word2id[word])
            continue

        new_id = len(word2id)
        id2word[new_id] = word
        word2id[word] = new_id
        corpus.append(new_id)

    corpus = np.array(corpus)
    return corpus, word2id, id2word


def create_co_matrix(corpus: np.ndarray, vocab_size, window_size=1) -> np.ndarray:
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            former_idx = idx - i
            latter_idx = idx + i
            if former_idx >= 0:
                former_word_id = corpus[former_idx]
                co_matrix[word_id, former_word_id] += 1
            if latter_idx < corpus_size:
                latter_word_id = corpus[latter_idx]
                co_matrix[word_id, latter_word_id] += 1
    return co_matrix


def cos_similarity(x: np.ndarray, y: np.ndarray):
    nx = x / np.sqrt(np.sum(x ** 2))
    ny = y / np.sqrt(np.sum(y ** 2))
    return np.dot(nx, ny)


def ppmi(co_matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix = np.zeros_like(co_matrix, dtype=np.float32)
    count_all = np.sum(co_matrix)
    count_each = np.sum(co_matrix, axis=0)

    for i in range(0, co_matrix.shape[0]):
        for j in range(0, co_matrix.shape[1]):
            if co_matrix[i, j] == 0:
                pmi = 0.0
            else:
                pmi = np.log2((co_matrix[i, j] * count_all) / (count_each[i] * count_each[j] + eps))

            matrix[i, j] = max(0, pmi)

    return matrix


def svd(ppmi_matrix: np.ndarray, n_components: int):
    try:
        from sklearn.utils.extmath import randomized_svd
        return randomized_svd(ppmi_matrix, n_components=n_components, n_iter=5)
    except ImportError:
        print("sklearn not found, using numpy.linalg.svd")
        return np.linalg.svd(ppmi_matrix)


def print_most_similar(word_vecs: np.ndarray,
                       word2id: Dict[str, int],
                       id2word: Dict[int, str],
                       query: str,
                       n_best: int):

    if query not in word2id.keys():
        print(f"{query} not in words")
        return
    query_word_id = word2id[query]

    vocab_size = len(word2id)
    sims = np.zeros(vocab_size)

    for i in range(0, vocab_size):
        sims[i] = cos_similarity(word_vecs[query_word_id], word_vecs[i])

    count = 0
    for i in (-1 * sims).argsort():
        if i == query_word_id:
            continue
        print(f"{id2word[i]}, {sims[i]}")
        count += 1
        if count > n_best:
            break


if __name__ == '__main__':
    main()
