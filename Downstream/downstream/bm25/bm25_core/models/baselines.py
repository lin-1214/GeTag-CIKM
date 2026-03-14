from typing import Optional, Union
from collections import Counter
from typing_extensions import deprecated
import numpy as np
from tqdm import tqdm
from .base import BaseRecPredictor, BaseIRPredictor
from .query_reduce import QueryReduce


@deprecated(
    'Browsed items exclude while calculating ranking metrics, '
    'making this baseline equivalent to random. '
    'Use RandomPredictor Instead.'
)
class BrowsedPredictor(BaseRecPredictor):
    """Score products using their indices in load sequences.

    E.g. Given loads(list of pids): [3, 10, 5, 3], then score(p3)==3, score(p10)==2, score(p5)==1.
    """

    def __init__(self, tag_corpous: dict[int, list[str]] = None):
        assert not tag_corpous
        return

    def predict_scores(self, x: Union[list[np.ndarray], list[int]]):
        """Score products using their indices in load sequences.

        E.g. Given loads(list of pids): [3, 10, 5, 3], then score(p3)==3, score(p10)==2, score(p5)==1.

        Args:
            x (list[list[int]]): list of product ids
        """

        def cal_score(loaded_pids: list[int]):
            return {pid: (i + 1) for i, pid in enumerate(loaded_pids)}

        return [cal_score(s) for s in x]


class RandomPredictor(BaseRecPredictor, BaseIRPredictor):

    def __init__(self, tag_corpous: dict[int, list[str]] = None):
        assert not tag_corpous
        return

    def predict_scores(self, x: Union[list[np.ndarray], list[int]]):
        return [{} for s in x]


class PopularItemPredictor(BaseRecPredictor, BaseIRPredictor):

    def __init__(
        self,
        tag_corpous: dict[int, list[str]] = None,
        user_behavior_seqs: Union[list[list[int], Optional[np.ndarray]]] = None,
    ):
        assert not tag_corpous
        assert user_behavior_seqs is not None

        if isinstance(user_behavior_seqs[0], np.ndarray):
            counter = Counter(np.concatenate(user_behavior_seqs))
        else:
            counter = Counter(sum(user_behavior_seqs, []))
        self.scores = dict(counter)
        return

    def predict_scores(self, x: Union[list[np.ndarray], list[int]]):
        return [self.scores for s in x]


class ItemDistributionPredictor(BaseRecPredictor, BaseIRPredictor):

    def __init__(
        self,
        tag_corpous: dict[int, list[str]] = None,
        user_behavior_seqs: Union[list[list[int], Optional[np.ndarray]]] = None,
        n_items: Optional[int] = None,
        query_type=None,
        seq_weights=None,
        max_seq_len=None,
    ):
        assert not tag_corpous
        assert user_behavior_seqs is not None
        assert n_items is not None

        counter = np.zeros((n_items, n_items))
        for seq in user_behavior_seqs:
            for prev, nxt in zip(seq, seq[1:]):
                counter[prev, nxt] += 1
        counter_sum = counter.sum(axis=1) + 1e-10
        counter /= counter_sum.reshape(-1, 1)
        self.query_fn = QueryReduce(
            n_items,
            query_fn=query_type,
            seq_weights=seq_weights,
            max_seq_len=max_seq_len,
        )
        self.w = counter.T
        if n_items > 1000:
            import cupy as cp
            self.xp = cp
            self.use_gpu = True
            self.w = cp.asarray(self.w)
        else:
            self.xp = np
            self.use_gpu = False
        return

    def _predict(self, x: list[int]):
        q = self.query_fn.reduce(x)
        scores = self.w @ self.xp.asarray(q)
        if self.use_gpu:
            scores = scores.get()
        return scores

    def _predict_batch(self, x: list[list[int]]):
        d0 = self.query_fn.xp.stack(
            [self.query_fn.reduce(x_) for x_ in x], axis=-1
        )
        d0 = self.xp.asarray(d0)
        scores = (self.w @ d0).T
        if self.use_gpu:
            scores = scores.get()
        return scores

    def predict_scores(self, x: list[list[int]]):
        """Calculate recommendation(ranking) scores of x (sequence of sequence of product ids)
        """
        if self.query_fn.max_seq_len is None:
            self.query_fn.init_seq_weights(max(map(len, x)))

        # # XXX: should be an argument
        BATCH_SIZE = 8192

        def batch_predict():
            with tqdm(
                range(0, len(x), BATCH_SIZE), leave=False, total=len(x)
            ) as pbar:
                for i in pbar:
                    scores = self._predict_batch(x[i:(i + BATCH_SIZE)])
                    yield scores
                    pbar.update(BATCH_SIZE)

        return np.concatenate(list(batch_predict()))
        # r2 = np.array(list(map(self._predict, tqdm(x, leave=False))))
        # return r2
