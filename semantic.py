from functools import lru_cache

import numpy as np
import polars as pl

import faiss

from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SemanticRecommender:
    def __init__(
            self,
            video_ids: np.ndarray, index: faiss.IndexFlatIP,
            tokenizer: AutoTokenizer, model: AutoModel,
    ) -> None:
        self._video_ids = video_ids
        self._index = index
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
        self._tokenizer = tokenizer
        self._model = model

    def predict(self, all_recs: np.ndarray, all_scores: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame([
            pl.Series('query_id', np.repeat(np.arange(len(all_recs)), [len(r) for r in all_recs]), pl.Int32),
            pl.Series('video_id', [self._video_ids[item] for rec in all_recs for item in rec]),
            pl.Series('sem_score', [item for rec in all_scores for item in rec], pl.Float32),
            pl.Series('sem_rank', [rnk + 1 for rec in all_recs for rnk, _ in enumerate(rec)], pl.Int8),
        ])

    @lru_cache
    def predict_one(self, query: str, top_k: int) -> pl.DataFrame:
        encoded_input = self._tokenizer(
            [query], padding=True, truncation=True, max_length=128, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            model_output = self._model(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings).detach().cpu()
        scores, recs = self._index.search(embeddings.numpy(), top_k)
        return pl.DataFrame([
            pl.Series('video_id', [self._video_ids[item] for item in recs[0]]),
            pl.Series('sem_score', scores[0], pl.Float32),
            pl.Series('sem_rank', [rnk + 1 for rnk, _ in enumerate(recs[0])], pl.Int8),
        ])
