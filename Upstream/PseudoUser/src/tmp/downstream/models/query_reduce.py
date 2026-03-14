from functools import partial
from typing import Literal, Callable
from collections import Counter
import numpy as np


def _query_last(loaded_tags: list[Counter]):
    return loaded_tags[-1]


class QueryReduce:
    """Reduce a sequence of items into query"""

    def __init__(
        self,
        n_items: int,
        query_fn: Literal['union', 'intersection', 'sum', 'last']
        | Callable = 'union',
        seq_weights: Literal['linear', 'log2', 'exp'] | None = None,
        max_seq_len: int | None = None,
        use_gpu: bool = False,
    ):
        self._seq_weights = None
        self.seq_weights_name = seq_weights
        self.max_seq_len = max_seq_len
        assert use_gpu is False, 'use_gpu is not faster, will be remove.'
        if use_gpu:
            import cupy as cp
            self.xp = cp
            _QUERY_FNS = {
                'union': cp.max,
                'intersection': cp.min,
                'last': _query_last,
                'sum': cp.sum,
            }
            self.query_fn = _QUERY_FNS.get(query_fn, query_fn)
        else:
            self.xp = np
            _QUERY_FNS = {
                'union': np.max,
                'intersection': np.min,
                'last': _query_last,
                'sum': np.sum,
            }
            self.query_fn = _QUERY_FNS.get(query_fn, query_fn)
        self._item_encode = self.xp.eye(n_items)

        if max_seq_len is not None:
            self.init_seq_weights(max_seq_len)
        return

    def init_seq_weights(self, max_seq_len: int):
        if self.seq_weights_name is None:
            return
        if self.seq_weights_name == 'linear':
            self._seq_weights = (self.xp.arange(max_seq_len) +
                                 1).reshape(-1, 1)
            return
        if self.seq_weights_name == 'exp':
            self._seq_weights = (self.xp.logspace(0, 10,
                                                  max_seq_len)).reshape(-1, 1)
            return

        assert self.seq_weights_name == 'log2'
        weights = 1 / self.xp.log2(2 + self.xp.arange(max_seq_len))
        self._seq_weights = weights[::-1].reshape(-1, 1)
        return

    def reduce(self, seq: list[int]) -> np.ndarray:
        if self.max_seq_len is not None:
            seq = seq[-self.max_seq_len:]

        seq = self._item_encode[seq]  # N x I
        if self.seq_weights_name is not None:
            assert self._seq_weights is not None, 'plz init_seq_weights first.'
            if self.seq_weights_name == 'linear':
                seq *= self._seq_weights[:len(seq)]
            elif self.seq_weights_name == 'exp':
                seq *= self._seq_weights[:len(seq)]
            elif self.seq_weights_name == 'log2':
                seq *= self._seq_weights[-len(seq):]
        return self.query_fn(seq, axis=0)  # I

    # def batch_reduce(self, seqs: list[list[int]]) -> np.ndarray:
    #     if self.max_seq_len is not None:
    #         seqs = [seq[-self.max_seq_len:] for seq in seqs]
    #         max_n = self.max_seq_len
    #     else:
    #         max_n = max(len(seq) for seq in seqs)

    #     if self.seq_weights_name in ('linear', 'exp'):

    #         def pad_seq(seq):
    #             return self.xp.pad(seq, [(0, max_n - len(seq)), (0, 0)])
    #     else:
    #         assert self.seq_weights_name == 'log2'

    #         def pad_seq(seq):
    #             return self.xp.pad(seq, [(max_n - len(seq), 0), (0, 0)])

    #     seqs = self.xp.stack([pad_seq(self._item_encode[seq]) for seq in seqs])

    #     if self.seq_weights_name is not None:
    #         assert self._seq_weights is not None, 'plz init_seq_weights first.'
    #         if self.seq_weights_name == 'linear':
    #             seqs *= self._seq_weights[:max_n]
    #         elif self.seq_weights_name == 'exp':
    #             seqs *= self._seq_weights[:max_n]
    #         elif self.seq_weights_name == 'log2':
    #             seqs *= self._seq_weights[-max_n:]
    #     return self.query_fn(seqs, axis=1)  # B x I
