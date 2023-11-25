# from functools import lru_cache
from typing import List, Tuple

import numpy as np
import polars as pl

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
    def __init__(self, items: List[List[str]]) -> None:
        self._item_dict = {}
        enum_users, enum_items = [], []

        for i in range(len(items)):
            for item in items[i]:
                enum_users.append(i)
                if item not in self._item_dict:
                    self._item_dict[item] = len(self._item_dict)
                enum_items.append(self._item_dict[item])

        self._sparse_matrix = sparse.csr_matrix(
            (np.ones(len(enum_users)), (enum_users, enum_items)),
            shape=(len(items), len(self._item_dict))
        )
        self._sparse_matrix = bm25_weight(self._sparse_matrix, K1=1.5, B=0.75)

    def recommend(self, items: List[List[str]], top_k: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_recs, all_scores = [], []
        batch_size = 100_000
        for i in tqdm(range(0, len(items), batch_size)):
            batch = items[i: i + batch_size]
            enum_users, enum_items = [], []
            for j in range(len(batch)):
                for item in batch[j]:
                    if item in self._item_dict:
                        enum_users.append(j)
                        enum_items.append(self._item_dict[item])

            cur_matrix = sparse.csr_matrix(
                (np.ones(len(enum_users)), (enum_users, enum_items)),
                shape=(len(batch), len(self._item_dict))
            )
            scores = cur_matrix.dot(self._sparse_matrix.T)
            for j in range(len(batch)):
                cur_row = scores.getrow(j)
                ids = cur_row.data.argsort()[-top_k:][::-1]
                all_recs.append(cur_row.indices[ids])
                all_scores.append(cur_row.data[ids])
        return all_recs, all_scores

    def recommend_one(self, items: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        enum_items = []
        for item in items:
            if item in self._item_dict:
                enum_items.append(self._item_dict[item])

        cur_matrix = sparse.csr_matrix(
            (np.ones(len(enum_items)), (np.zeros(len(enum_items)), enum_items)),
            shape=(1, len(self._item_dict))
        )
        scores = cur_matrix.dot(self._sparse_matrix.T)
        ids = scores.data.argsort()[-top_k:][::-1]
        return scores.indices[ids], scores.data[ids]


class LemmaRecommender:
    def __init__(self) -> None:
        self._tokenized_corpus = None
        self._mapper = {}
        self._video_dict = {}

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

    def fit(self, videos: List[str], corpus: List[str]) -> "LemmaRecommender":
        for i, video in enumerate(videos):
            self._video_dict[i] = video
        self._tokenized_corpus = [self.lemmatize(clear_text(sent)) for sent in tqdm(corpus)]
        self._ranker = Ranker(self._tokenized_corpus)
        return self

    def predict(self, queries: List[str], top_k: int) -> pl.DataFrame:
        """Return top_k relevant indexes and their scores"""

        if self._ranker is None:
            raise Exception("Fit model at first!")

        tokenized_queries = [self.lemmatize(clear_text(query)) for query in queries]

        all_recs, all_scores = self._ranker.recommend(tokenized_queries, top_k)
        return pl.DataFrame([
            pl.Series('query_id', np.repeat(np.arange(len(queries)), [len(r) for r in all_recs]), pl.Int32),
            pl.Series('video_id', [self._video_dict[item] for rec in all_recs for item in rec]),
            pl.Series('lex_score', [item for rec in all_scores for item in rec], pl.Float32),
            pl.Series('lex_rank', [rnk + 1 for rec in all_recs for rnk, _ in enumerate(rec)], pl.Int8),
        ])

    def predict_one(self, query: str, top_k: int) -> pl.DataFrame:
        """Return top_k relevant indexes and their scores"""

        if self._ranker is None:
            raise Exception("Fit model at first!")

        tokenized_query = self.lemmatize(clear_text(query))

        recs, scores = self._ranker.recommend_one(tokenized_query, top_k)
        return pl.DataFrame([
            pl.Series('video_id', [self._video_dict[item] for item in recs]),
            pl.Series('lex_score', scores, pl.Float32),
            pl.Series('lex_rank', [rnk + 1 for rnk, _ in enumerate(recs)], pl.Int8),
        ])
