"""https://github.com/Inspirateur/Fast-BM25/blob/main/fast_bm25.py"""
import os
import json
from typing import Callable, Literal, Optional, Union
import hashlib
from collections import defaultdict, Counter
import heapq
import math
import pickle
import sys
import numpy as np
from tqdm import tqdm
from functools import reduce, partial
from joblib import Parallel, delayed
from .base import BaseRecPredictor, BaseIRPredictor
from .query_reduce import QueryReduce

PARAM_K1 = 1.5
PARAM_B = 0.75
IDF_CUTOFF = 4


class FastBM25:
    """Fast Implementation of Best Matching 25 ranking function.

	Attributes
	----------
	t2d : <token: <doc, freq>>
		Dictionary with terms frequencies for each document in `corpus`.
	idf: <token, idf score>
		Pre computed IDF score for every term.
	doc_len : list of int
		List of document lengths.
	avgdl : float
		Average length of document in `corpus`.
	"""

    def __init__(self, corpus: list[list[str]], k1=None, b=None, alpha=None):
        """
		Parameters
		----------
		corpus : list of list of str
			Given corpus.
		k1 : float
			Constant used for influencing the term frequency saturation. After saturation is reached, additional
			presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
			that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
			the type of documents or queries.
		b : float
			Constant used for influencing the effects of different document lengths relative to average document length.
			When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
			[1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
			depends on factors such as the type of documents or queries.
		alpha: float
			IDF cutoff, terms with a lower idf score than alpha will be dropped. A higher alpha will lower the accuracy
			of BM25 but increase performance
		"""
        self.k1 = k1 if k1 is not None else PARAM_K1
        self.b = b if b is not None else PARAM_B
        self.alpha = alpha if alpha is not None else IDF_CUTOFF

        self.avgdl = 0
        self.t2d = {}
        self.idf = {}
        self.doc_len = []
        self.token_scores_cache = {}
        if corpus:
            self._initialize(corpus)

    @property
    def corpus_size(self):
        return len(self.doc_len)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        for i, document in enumerate(corpus):
            self.doc_len.append(len(document))

            for word in document:
                if word not in self.t2d:
                    self.t2d[word] = {}
                if i not in self.t2d[word]:
                    self.t2d[word][i] = 0
                self.t2d[word][i] += 1

        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        to_delete = []
        for word, docs in self.t2d.items():
            idf = math.log(self.corpus_size - len(docs) +
                           0.5) - math.log(len(docs) + 0.5)
            # only store the idf score if it's above the threshold
            if idf > self.alpha:
                self.idf[word] = idf
            else:
                # print(idf, word)
                to_delete.append(word)
        # print(f"Dropping {len(to_delete)} terms")
        for word in to_delete:
            del self.t2d[word]

        self.average_idf = sum(self.idf.values()) / (len(self.idf) + 1e-6)

        if self.average_idf < 0:
            print(
                f'Average inverse document frequency is less than zero. Your corpus of {self.corpus_size} documents'
                ' is either too small or it does not originate from natural text. BM25 may produce'
                ' unintuitive results.', file=sys.stderr
            )

        # Pre-compute all token scores
        for token in self.t2d:
            self._get_token_scores(token)

    def _get_token_scores(self, token: str):
        if token in self.token_scores_cache:
            return self.token_scores_cache[token]

        scores: dict[int, float] = defaultdict(float)
        for index, freq in self.t2d[token].items():
            denom_cst = self.k1 * (
                1 - self.b + self.b * self.doc_len[index] / self.avgdl
            )
            scores[index] += self.idf[token] * freq * (self.k1 +
                                                       1) / (freq + denom_cst)
        self.token_scores_cache[token] = scores
        return scores

    def get_scores(self, query: list[str], documents: list[str]):
        """
		Retrieve the top n documents for the query.

		Parameters
		----------
		query: list of str
			The tokenized query
		documents: list
			The documents to return from
		n: int
			The number of documents to return

		Returns
		-------
		list
			The top n documents
		"""
        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"
        scores = defaultdict(float)
        if not isinstance(query, dict):
            query = Counter(query)
        for token, weight in query.items():
            if token in self.t2d:
                for i, score in self._get_token_scores(token).items():
                    scores[i] += weight * score
                # for index, freq in self.t2d[token].items():
                #     denom_cst = self.k1 * (
                #         1 - self.b + self.b * self.doc_len[index] / self.avgdl
                #     )
                #     scores[index] += weight * self.idf[token] * freq * (
                #         self.k1 + 1
                #     ) / (freq + denom_cst)

        return scores

    def get_top_n(self, query, documents, n=5):
        """
		Retrieve the top n documents for the query.

		Parameters
		----------
		query: list of str
			The tokenized query
		documents: list
			The documents to return from
		n: int
			The number of documents to return

		Returns
		-------
		list
			The top n documents
		"""
        scores = self.get_scores(query, documents)

        return [
            documents[i]
            for i in heapq.nlargest(n, scores.keys(), key=scores.__getitem__)
        ]

    def save(self, filename):
        with open(f"{filename}.pkl", "wb") as fsave:
            pickle.dump(self, fsave, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(f"{filename}.pkl", "rb") as fsave:
            return pickle.load(fsave)


class VeryFastBM25(FastBM25):
    """Fast Implementation of Best Matching 25 ranking function.

	Attributes
	----------
	t2d : <token: <doc, freq>>
		Dictionary with terms frequencies for each document in `corpus`.
	idf: <token, idf score>
		Pre computed IDF score for every term.
	doc_len : list of int
		List of document lengths.
	avgdl : float
		Average length of document in `corpus`.
	"""

    def _initialize(self, corpus):
        super()._initialize(corpus)
        # Pre-compute all token scores
        self.scores = np.zeros((len(self.t2d), len(corpus)))
        self.token_to_id = {}
        for token_id, token in enumerate(self.t2d):
            self.token_to_id[token] = token_id
            score_dict = self._get_token_scores(token)
            for document_id, score in score_dict.items():
                self.scores[token_id, document_id] = score

    def get_scores(self, query: np.ndarray, documents: list[str]):
        """
		Retrieve the top n documents for the query.

		Parameters
		----------
		query: list of str
			The tokenized query
		documents: list
			The documents to return from
		n: int
			The number of documents to return

		Returns
		-------
		list
			The top n documents
		"""
        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"

        assert len(query.shape) == 1
        assert query.shape[0] == self.scores.shape[0]
        return query @ self.scores


class BM25IRPredictor(BaseIRPredictor):

    def __init__(
        self,
        tag_corpus: list[list[str]],
        document_corpus: Optional[list[list[str]]] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        alpha: Optional[float] = None,
        use_gpu: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            product_tags (dict[int, list[str]]): Mapping that map product ids to their tags.
        """

        kwargs.update(k1=k1, b=b, alpha=alpha)
        self._raw_tag_corpus = tag_corpus
        self.document_corpus = document_corpus or tag_corpus
        self.bm25 = VeryFastBM25(self.document_corpus, **kwargs)
        self.use_gpu = (
            (len(tag_corpus) > 1000) if use_gpu is None else use_gpu
        )
        self.tag_corpus = self._process_tag_corpus(tag_corpus)
        self.verbose = verbose
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                print("Warning: cupy not available, falling back to CPU (numpy)")
                self.use_gpu = False
                self.xp = np
        else:
            self.xp = np

        cache_path = cache_dir and os.path.join(cache_dir, f'{self._hash}.npy')
        if (
            force_reload is False and cache_dir is not None
            and os.path.isfile(cache_path)
        ):
            self._precomputed_scores = self.xp.load(cache_path)
            if verbose:
                print(f'Cache for the bm25 model found. loaded from cache.')
        else:
            self._precomputed_scores = self._precompute()
            if cache_path is not None:
                os.makedirs(cache_dir, exist_ok=True)
                self.xp.save(cache_path, self._precomputed_scores)
                if verbose:
                    print(f'The bm25 model has been saved in {cache_path}.')
        return

    def _precompute(self):
        xp = self.xp
        return xp.asarray(self.tag_corpus) @ xp.asarray(self.bm25.scores)

    def _process_tag_corpus(self, tag_corpus: list[list[str]]):
        new_tag_corpus = np.zeros(
            (len(tag_corpus), len(self.bm25.token_to_id))
        )
        for corpus_id, tags_counts in enumerate(map(Counter, tag_corpus)):
            for tag, count in tags_counts.items():
                if tag in self.bm25.token_to_id:
                    new_tag_corpus[corpus_id,
                                   self.bm25.token_to_id[tag]] = count
        return new_tag_corpus

    def _predict(self, x: list[int]):
        assert len(x) == 1
        r = self._precomputed_scores[x[0]]
        return r

    def predict_scores(self, x: list[int]):
        """Calculate recommendation(ranking) scores of x (sequence of sequence of product ids)
        """
        if self.use_gpu:
            # no need to use gpu in ir case (only indexing)
            self._precomputed_scores = self.xp.asnumpy(
                self._precomputed_scores
            )
        return list(map(self._predict, x))
        # return Parallel(n_jobs=self.n_jobs
        #                 )(delayed(self._predict)(i) for i in x)

    @property
    def _hash(self):
        s = json.dumps(
            dict(
                name=self.__class__.__name__,
                tag_corpus=self._raw_tag_corpus,
                document_corpus=self.document_corpus,
                k1=self.bm25.k1,
                b=self.bm25.b,
                alpha=self.bm25.alpha,
            )
        )
        return hashlib.sha1(s.encode()).hexdigest()


class BM25RecPredictor(BM25IRPredictor, BaseRecPredictor):

    def __init__(
        self,
        tag_corpus: list[list[str]],
        document_corpus: Optional[list[list[str]]] = None,
        query_fn: Union[Literal['union', 'intersection', 'sum', 'last'], Callable] = 'union',
        seq_weights: Optional[Literal['linear', 'log2', 'exp']] = None,
        max_seq_len: Optional[int] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        alpha: Optional[float] = None,
        use_gpu: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            product_tags (dict[int, list[str]]): Mapping that map product ids to their tags.
            query_fn (Union[Literal['union', 'sum'], Callable[[list[list[str]]]], list[str]], optional): 
                Function to reduce the tags of loaded products (list[list[str]]) to be query (list[str]).
                Defaults to 'sum'.
        """

        super().__init__(
            tag_corpus=tag_corpus,
            document_corpus=document_corpus,
            k1=k1,
            b=b,
            alpha=alpha,
            use_gpu=use_gpu,
            cache_dir=cache_dir,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.query_fn = QueryReduce(
            n_items=len(self.document_corpus),
            query_fn=query_fn,
            seq_weights=seq_weights,
            max_seq_len=max_seq_len,
            # use_gpu=self.use_gpu,
            use_gpu=False,
        )
        return

    def _predict(self, seq: list[int]):
        seq = self.query_fn.reduce(seq)  # |I|
        seq = self.xp.asarray(seq)
        r = (seq @ self._precomputed_scores)
        if not self.use_gpu:
            return {iid: r[iid] for iid in np.nonzero(r)[0]}
        r = r.get()
        return r

    def _predict_batch(self, seqs: list[list[int]]):
        seqs = self.query_fn.xp.stack(
            [self.query_fn.reduce(seq) for seq in seqs],
            axis=0,
        )
        seqs = self.xp.asarray(seqs)
        # B x |I|
        r = (seqs @ self._precomputed_scores)
        if not self.use_gpu:
            return {iid: r[iid] for iid in np.nonzero(r)[0]}
        r = r.get()
        return r

    def predict_scores(self, x: list[list[int]]):
        """Calculate recommendation(ranking) scores of x (sequence of sequence of product ids)
        """
        if self.query_fn.max_seq_len is None:
            self.query_fn.init_seq_weights(max(map(len, x)))
        # XXX: should be an argument
        BATCH_SIZE = 16384

        def batch_predict():
            with tqdm(
                range(0, len(x), BATCH_SIZE), leave=False, total=len(x)
            ) as pbar:
                for i in pbar:
                    scores = self._predict_batch(x[i:(i + BATCH_SIZE)])
                    yield scores
                    pbar.update(BATCH_SIZE)

        if self.use_gpu:
            results = np.concatenate(list(batch_predict()))
            # Avoid output too sparse
            wherezero = np.where(results == 0)
            results[wherezero] -= np.random.rand(len(wherezero[0]))
        else:
            results = list(map(self._predict, tqdm(x, leave=False)))

        return results
