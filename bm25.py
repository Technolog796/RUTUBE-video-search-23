# from functools import lru_cache
from typing import List, Tuple, Any

import numpy as np
import pymorphy2
from rank_bm25 import BM25Okapi

import re
from string import punctuation

from tqdm.auto import tqdm


def clear_text(text: str) -> str:
    text = re.sub(f"[{punctuation}]", '', text)
    return text


class BM25Model:
    def __init__(self, max_k: int) -> None:
        self.max_k = max_k
        self.tokenized_corpus = None
        self.mapper = {}

        self.morph = pymorphy2.MorphAnalyzer()
        self.bm25 = None

    def get_nf(self, word: str) -> str:
        if word not in self.mapper:
            self.mapper[word] = self.morph.parse(word)[0].normal_form  # кэшируем для ускорения обучения
        return self.mapper[word]

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
            res.append(nf)
        return res

    def fit(self, corpus: List[str]) -> "BM25Model":
        self.tokenized_corpus = [self.lemmatize(clear_text(sent)) for sent in tqdm(corpus)]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        return self

    def predict(self, query: str, top_k: int = None) -> Tuple[Any, list]:
        """Return top_k relevant indexes and their scores"""

        if self.bm25 is None:
            raise Exception("Fit model at first!")

        tokenized_query = self.lemmatize(clear_text(query))
        doc_scores = self.bm25.get_scores(tokenized_query)

        if top_k is None:
            n_positive = np.sum(np.array(doc_scores) > 0)
            top_k = np.min([n_positive, self.max_k])

        inds = np.argsort(doc_scores)[::-1]
        scores = sorted(doc_scores, reverse=True)
        return inds[:top_k], scores[:top_k]
