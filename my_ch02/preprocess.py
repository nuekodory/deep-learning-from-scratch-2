import math
from typing import List, Dict
import numpy as np


def preprocess(text: str) -> (List[int], Dict, Dict):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def co_words(corpus, id_to_word) -> List:
    co_arr = []
    for word_id in id_to_word.keys():
        arr = [0] * len(id_to_word)
        idl = [i for i, wid in enumerate(corpus) if wid == word_id]
        for idx in idl:
            start = max(idx-window, 0)
            end = min(idx+window+1, len(corpus))
            ids = list(corpus[start:idx])
            ids.extend(corpus[idx+1:end])
            for wid in ids:
                arr[wid] += 1
        co_arr.append(arr)
    return co_arr


def cos_similarity(x: List, y: List, eps=1e-8) -> float:
    xy = sum(_x * _y for _x, _y in zip(x, y))
    xq = math.sqrt(sum(_x ** 2 for _x in x))
    yq = math.sqrt(sum(_y ** 2 for _y in y))
    return xy / ((xq + eps) * (yq + eps))


def most_similar(query: str, word_to_id: dict, id_to_word: dict, word_matrix: List, top: int = 5):
    if query not in word_to_id.keys():
        print(f"{query} not found in words")
        return

    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.asarray([cos_similarity(word_matrix[i], query_vec) for i in range(0, vocab_size)])

    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f"{id_to_word[i]} {similarity[i]}")
        count += 1

        if count >= top:
            return


if __name__ == '__main__':
    window = 1
    txt = "you say goodbye and I say hello."
    cop, w2i, i2w = preprocess(txt)

    cow = co_words(cop, i2w)
    print(cow)
    print(cos_similarity(cow[w2i["i"]], cow[w2i["you"]]))
    most_similar("goodbye", w2i, i2w, cow)
