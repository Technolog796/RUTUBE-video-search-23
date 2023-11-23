# from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np

from implicit.nearest_neighbours import bm25_weight
from scipy import sparse

import pymorphy2
import re
from string import punctuation

from tqdm.auto import tqdm


def clear_text(text: str) -> str:
    text = re.sub(f"[{punctuation}]", '', text)
    return text


class Ranker:
    def __init__(self, users: Union[np.ndarray, List[int]], items: List[List[str]]) -> None:
        self._item_dict = {}
        enum_users, enum_items = [], []

        for i in range(len(users)):
            for item in items[i]:
                enum_users.append(i)
                if item not in self._item_dict:
                    self._item_dict[item] = len(self._item_dict)
                enum_items.append(self._item_dict[item])

        self._sparse_matrix = sparse.csr_matrix(
            (np.ones(len(enum_users)), (enum_users, enum_items)),
            shape=(len(users), len(self._item_dict))
        )
        self._sparse_matrix = bm25_weight(self._sparse_matrix, K1=1.5, B=0.75)

    def recommend(self, items: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        enum_items = []
        for item in items:
            if item in self._item_dict:
                enum_items.append(self._item_dict[item])
        enum_users = [0] * len(enum_items)

        cur_matrix = sparse.csr_matrix(
            (np.ones(len(enum_users)), (enum_users, enum_items)),
            shape=(1, len(self._item_dict))
        )
        scores = cur_matrix.dot(self._sparse_matrix.T)
        ids = scores.data.argsort()[-top_k:][::-1]
        return scores.indices[ids], scores.data[ids]


class LemmaRecommender:
    def __init__(self) -> None:
        self._tokenized_corpus = None
        self._mapper = {}

        self._morph = pymorphy2.MorphAnalyzer()
        self._ranker = None

    def get_nf(self, word: str) -> str:
        if word not in self._mapper:
            self._mapper[word] = self._morph.parse(word)[0].normal_form  # кэшируем для ускорения обучения
        return self._mapper[word]

    # @lru_cache(maxsize=1_000_000)
    # def get_nf(self, word: str) -> str:
    #     return self.morph.parse(word)[0].normal_form

    def lemmatize(self, text: str) -> List[str]:
        words = text.split()  # разбиваем текст на слова
        res = list()
        used = set()
        for word in words:
            nf = self.get_nf(word)
            if nf not in used:
                res.append(nf)
                used.add(nf)
        return res

    def fit(self, corpus: List[str]) -> "LemmaRecommender":
        self._tokenized_corpus = [self.lemmatize(clear_text(sent)) for sent in tqdm(corpus)]
        self._ranker = Ranker(np.arange(len(corpus)), self._tokenized_corpus)
        return self

    def predict(self, query: str, top_k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return top_k relevant indexes and their scores"""

        if self._ranker is None:
            raise Exception("Fit model at first!")

        tokenized_query = self.lemmatize(clear_text(query))

        return self._ranker.recommend(tokenized_query, top_k)
