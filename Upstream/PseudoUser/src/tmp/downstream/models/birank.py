import os
import json
from typing import Literal, Callable
from scipy import sparse
import numpy as np
from joblib import Parallel, delayed
import hashlib
from tqdm import tqdm
from .base import BaseRecPredictor, BaseIRPredictor
from .query_reduce import QueryReduce
from .. import corpus_utils

# class FastBirank:

#     def __init__(
#         self,
#         item_user_graph: sparse.coo_matrix,
#         alpha: float,
#         beta: float,
#         query_type: Literal['item', 'user'],
#         tol: float = 1e-12,
#         max_iters: int = 1000,
#         cache_dir: str | None = None,
#         force_reload: bool = False,
#         verbose: bool = False,
#     ):
#         self.item_user_graph = item_user_graph
#         self.alpha = alpha
#         self.beta = beta
#         self.max_iters = max_iters
#         self.tol = tol
#         assert query_type in {'item', 'user'}
#         assert alpha == beta, 'There is no mathematical meaning setting alpha!=beta'
#         # XXX: alpha and beta should be one argument
#         self.query_type = query_type
#         assert query_type == 'item'  # TODO

#         cache_path = cache_dir and os.path.join(
#             cache_dir, f'birank_{self._hash}.npy'
#         )
#         if (
#             force_reload is False and cache_dir is not None
#             and os.path.isfile(cache_path)
#         ):
#             import cupy as cp
#             self._accumulated_transition = cp.load(cache_path)
#             if verbose:
#                 print('Cache for birank found. Loaded from cache.')
#         else:
#             self._accumulated_transition = self._precompute()
#             if cache_path is not None:
#                 os.makedirs(cache_dir, exist_ok=True)
#                 np.save(cache_path, self._accumulated_transition)
#                 if verbose:
#                     print(
#                         f'The birank model has been computed and saved in {cache_path}.'
#                     )
#         return
# def get_scores(self, query: np.ndarray):
#     import cupy as cp
#     scores = self._accumulated_transition @ cp.asarray(query)
#     if self.query_type == 'item':
#         scores *= (1 - self.alpha)
#     return scores.get()


def fast_birank(
    item_user_graph: sparse.coo_matrix,
    alpha: float,
    beta: float,
    transition_type: Literal['item->item', 'user->item'],
    tol: float = 1e-12,
    max_iters: int = 1000,
    verbose: bool = False,
):
    import cupy as cp
    g = item_user_graph
    deg_u = np.array(g.sum(axis=1)).flatten()
    deg_p = np.array(g.sum(axis=0)).flatten()
    # avoid divided by zero issue
    deg_u[np.where(deg_u == 0)] += 1
    deg_p[np.where(deg_p == 0)] += 1
    deg_u = sparse.diags(1 / np.sqrt(deg_u))
    deg_p = sparse.diags(1 / np.sqrt(deg_p))
    # Sp = Kp_bi.dot(WT).dot(Kd_bi)
    s = deg_u.dot(g).dot(deg_p)

    base_transistion = alpha * beta * (
        cp.asarray(s.toarray()) @ cp.asarray(s.T.toarray())
    )
    cur_transistion = cp.eye(base_transistion.shape[0])
    accumulated_transition = cur_transistion.copy()

    with tqdm(
        range(max_iters),
        leave=False,
        desc='precomputing birank...',
        disable=(not verbose),
    ) as pbar:
        for _ in pbar:
            cur_transistion = base_transistion @ cur_transistion
            accumulated_transition += cur_transistion
            step_size = cp.linalg.norm(cur_transistion)
            pbar.set_description(
                f'precomputing, step_size = {step_size:e}/{tol:e}'
            )
            if step_size < tol:
                break
    del cur_transistion
    del base_transistion
    if transition_type == 'item->item':
        # scores = (1- alpha) * (accumulated_transition @ item_query)
        return (1 - alpha) * accumulated_transition
    assert transition_type == 'user->item'
    # scores = alpha * (1- beta) * (accumulated_transition @ s @ user_query)
    return (
        alpha * (1 - beta) *
        (accumulated_transition @ cp.asarray(s.toarray()))
    )


def pagerank(
    adj, x0=None, alpha=0.85, max_iter=200, tol=1.0e-4, verbose=False
):
    """
    Return the PageRank of the nodes in a graph using power iteration.
    This funciton takes the sparse matrix as input directly, avoiding the overheads
    of converting the network to a networkx Graph object and back.

    Input:
        adj::scipy.sparsematrix:Adjacency matrix of the graph
        d::float:Dumping factor
        max_iter::int:Maximum iteration times
        tol::float:Error tolerance to check convergence
        verbose::boolean:If print iteration information

    Output:
        ::numpy.ndarray:The PageRank values
    """
    adj = adj.astype('float', copy=False)
    n_node = adj.shape[0]
    S = np.array(adj.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = sparse.spdiags(S.T, 0, *adj.shape, format='csr')
    M = Q * adj

    if x0 is None:
        x0 = np.repeat(1.0 / n_node, n_node)
    x = x0.copy()

    for i in range(max_iter):
        xlast = x
        x = alpha * (x * M) + (1 - alpha) * x0
        err = np.absolute(x - xlast).sum()
        if verbose:
            print(i, err)
        if err < tol:
            break

    return x


def birank(
    W,
    normalizer: Literal['HITS', 'CoHITS', 'BGRM', 'BiRank'],
    alpha=0.85,
    beta=0.85,
    max_iter=200,
    tol=1.0e-4,
    d0=None,
    p0=None,
    verbose=False,
):
    """
    Calculate the PageRank of bipartite networks directly.
    See paper https://ieeexplore.ieee.org/abstract/document/7572089/
    for details.
    Different normalizer yields very different results.
    More studies are needed for deciding the right one.

    Input:
        W::scipy's sparse matrix:Adjacency matrix of the bipartite network D*P
        normalizer::string:Choose which normalizer to use, see the paper for details
        alpha, beta::float:Damping factors for the rows and columns
        max_iter::int:Maximum iteration times
        tol::float:Error tolerance to check convergence
        verbose::boolean:If print iteration information

    Output:
         d, p::numpy.ndarray:The BiRank for rows and columns
    """

    W = W.astype('float', copy=False)
    WT = W.T

    Kd = np.array(W.sum(axis=1)).flatten()
    Kp = np.array(W.sum(axis=0)).flatten()
    # Kd = scipy.array(W.sum(axis=1)).flatten()
    # Kp = scipy.array(W.sum(axis=0)).flatten()
    # avoid divided by zero issue
    Kd[np.where(Kd == 0)] += 1
    Kp[np.where(Kp == 0)] += 1

    Kd_ = sparse.diags(1 / Kd, 0)
    Kp_ = sparse.diags(1 / Kp, 0)

    assert normalizer in ['HITS', 'CoHITS', 'BGRM', 'BiRank']
    if normalizer == 'HITS':
        Sp = WT
        Sd = W
    elif normalizer == 'CoHITS':
        Sp = WT.dot(Kd_)
        Sd = W.dot(Kp_)
    elif normalizer == 'BGRM':
        Sp = Kp_.dot(WT).dot(Kd_)
        Sd = Sp.T
    elif normalizer == 'BiRank':
        Kd_bi = sparse.diags(1 / np.sqrt(Kd))
        Kp_bi = sparse.diags(1 / np.sqrt(Kp))
        Sp = Kp_bi.dot(WT).dot(Kd_bi)
        Sd = Sp.T

    if d0 is None:
        d0 = np.repeat(1 / Kd_.shape[0], Kd_.shape[0])
    if p0 is None:
        p0 = np.repeat(1 / Kp_.shape[0], Kp_.shape[0])
    assert d0.shape[0] == Kd_.shape[0]
    assert p0.shape[0] == Kp_.shape[0]
    d_last = d0.copy()
    p_last = p0.copy()

    for i in range(max_iter):
        p = alpha * (Sp.dot(d_last)) + (1 - alpha) * p0
        d = beta * (Sd.dot(p_last)) + (1 - beta) * d0

        if normalizer == 'HITS':
            p = p / p.sum()
            d = d / d.sum()

        err_p = np.absolute(p - p_last).sum()
        err_d = np.absolute(d - d_last).sum()
        if verbose:
            print(
                "Iteration : {}; top error: {}; bottom error: {}".format(
                    i, err_d, err_p
                )
            )
        if err_p < tol and err_d < tol:
            break
        d_last = d
        p_last = p

    return d, p


class BiRankRecPredictor(BaseRecPredictor):

    def __init__(
        self,
        tag_corpus: list[list[str]],
        query_type: Literal['union', 'intersection', 'sum', 'last']
        | Callable = 'union',
        seq_weights: Literal['linear', 'log2', 'exp'] | None = None,
        max_seq_len: int | None = None,
        alpha: float = 0.85,
        beta: float = 0.85,
        cache_dir: str | None = None,
        use_gpu: bool = True,
        force_reload: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        self.use_gpu = use_gpu
        self.tag_corpus = tag_corpus
        self.graph = corpus_utils.to_graph(tag_corpus)
        self.query_fn = QueryReduce(
            n_items=len(tag_corpus),
            query_fn=query_type,
            seq_weights=seq_weights,
            max_seq_len=max_seq_len,
            use_gpu=False,
        )
        kwargs.update(alpha=alpha, beta=beta)
        self.kwargs = kwargs
        if use_gpu:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        cache_path = cache_dir and os.path.join(
            cache_dir, f'birankrec_{self._hash}.npy'
        )
        if (
            force_reload is False and cache_dir is not None
            and os.path.isfile(cache_path)
        ):
            self.birank_transition = self.xp.load(cache_path)
            if verbose:
                print('Cache for birank found. Loaded from cache.')
        else:
            self.birank_transition = fast_birank(
                self.graph,
                transition_type='item->item',
                verbose=verbose,
                **kwargs,
            )
            if cache_path is not None:
                os.makedirs(cache_dir, exist_ok=True)
                self.xp.save(cache_path, self.birank_transition)
                if verbose:
                    print(
                        f'The birank transition has been computed and saved in {cache_path}.'
                    )
        return

    def _predict(self, x: list[int]):
        d0 = self.query_fn.reduce(x)
        scores = self.birank_transition @ self.xp.asarray(d0)
        if self.use_gpu:
            scores = scores.get()
        return scores

    def _predict_batch(self, x: list[list[int]]):
        d0 = self.query_fn.xp.stack(
            [self.query_fn.reduce(x_) for x_ in x], axis=-1
        )
        d0 = self.xp.asarray(d0)
        scores = (self.birank_transition @ d0).T
        if self.use_gpu:
            scores = scores.get()
        return scores

    def predict_scores(self, x: list[list[int]]):
        """Calculate recommendation(ranking) scores of x (sequence of sequence of product ids)
        """
        if self.query_fn.max_seq_len is None:
            self.query_fn.init_seq_weights(max(map(len, x)))

        # XXX: should be an argument
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

    @property
    def _hash(self):
        s = json.dumps(
            dict(
                name=self.__class__.__name__,
                graph=self.tag_corpus,
                alpha=self.kwargs['alpha'],
                beta=self.kwargs['beta'],
                tol=self.kwargs.get('tol', ''),
                max_iters=self.kwargs.get('max_iters', ''),
            )
        )
        return hashlib.sha1(s.encode()).hexdigest()


class BiRankIRPredictor(BaseIRPredictor):

    def __init__(
        self,
        tag_corpus: list[list[str]],
        alpha: float = 0.85,
        beta: float = 0.85,
        cache_dir: str | None = None,
        use_gpu: bool = True,
        force_reload: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        self.tag_corpus = tag_corpus
        self.graph = corpus_utils.to_graph(tag_corpus)
        self._to_onehot = np.eye(self.graph.shape[0])
        self.use_gpu = use_gpu
        kwargs.update(alpha=alpha, beta=beta)
        self.kwargs = kwargs
        if use_gpu:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        cache_path = cache_dir and os.path.join(
            cache_dir, f'birankir_{self._hash}.npy'
        )
        if (
            force_reload is False and cache_dir is not None
            and os.path.isfile(cache_path)
        ):
            self.birank_transition = self.xp.load(cache_path)
            if verbose:
                print('Cache for birank found. Loaded from cache.')
        else:
            _w = fast_birank(
                self.graph,
                transition_type='user->item',
                verbose=verbose,
                **kwargs,
            )
            self.birank_transition = _w @ self.xp.asarray(
                self.graph.T.toarray()
            )  # make the transition item->item
            if cache_path is not None:
                os.makedirs(cache_dir, exist_ok=True)
                self.xp.save(cache_path, self.birank_transition)
                if verbose:
                    print(
                        f'The birank transition has been computed and saved in {cache_path}.'
                    )
        if self.use_gpu:
            self.birank_transition = self.birank_transition.get()
            # No need to use gpu during prediction
        return

    def _predict(self, x: list[int]):
        # i0 = np.zeros(self.graph.shape[0])
        # t0 = self.graph.tocsr()[x].toarray().flatten()
        # i, _t = birank(
        #     self.graph, d0=i0, p0=t0, normalizer='BiRank', **self.kwargs
        # )
        assert len(x) == 1
        return self.birank_transition[:, x[0]]

    def predict_scores(self, x: list[int]):
        """Calculate recommendation(ranking) scores of x (sequence of sequence of product ids)
        """
        return np.array(list(map(self._predict, x)))

    @property
    def _hash(self):
        s = json.dumps(
            dict(
                name=self.__class__.__name__,
                graph=self.tag_corpus,
                alpha=self.kwargs['alpha'],
                beta=self.kwargs['beta'],
                tol=self.kwargs.get('tol', ''),
                max_iters=self.kwargs.get('max_iters', ''),
            )
        )
        return hashlib.sha1(s.encode()).hexdigest()


def bipartite_to_homogeneous(badj: sparse.coo_matrix):
    u, v = badj.row, badj.col
    n = sum(badj.shape)
    adj = sparse.coo_matrix((badj.data, (u, v + badj.shape[0])), shape=(n, n))
    adj += adj.T
    return adj


class PageRankRecPredictor(BaseRecPredictor):

    def __init__(
        self,
        tag_corpous: list[list[str]] | sparse.spmatrix,
        query_type: Literal['union', 'sum'],
        n_jobs: int = 0,
        alpha: float = 0.85,
        **kwargs,
    ):
        self.tag_corpus = tag_corpous
        if not isinstance(tag_corpous, sparse.spmatrix):
            self.graph = corpus_utils.to_graph(tag_corpous)  # item->tag
        else:
            self.graph = tag_corpous
        self.homo_graph = bipartite_to_homogeneous(
            self.graph
        )  # ([item,tag] -> [item,tag])
        self.query_type = query_type
        self._to_onehot = np.eye(self.graph.shape[0])
        kwargs.update(alpha=alpha)
        self.kwargs = kwargs
        self.n_jobs = n_jobs
        return

    def _seq_to_onehot(self, seq: list[int]):
        onehot = self._to_onehot[seq]
        if self.query_type == 'union':
            return onehot.sum(axis=0).astype(bool).astype(float)
        return onehot.sum(axis=0)

    def _predict(self, x: list[int]):
        d0 = self._seq_to_onehot(x)
        p0 = np.zeros(self.graph.shape[1])
        r0 = np.concatenate([d0, p0])
        r = pagerank(self.homo_graph, r0, **self.kwargs)
        return r[:len(d0)]

    def predict_scores(self, x: list[list[int]]):
        """Calculate recommendation(ranking) scores of x (sequence of sequence of product ids)
        """
        if self.n_jobs == 0:
            return np.array(list(map(self._predict, x)))
        return np.array(
            Parallel(n_jobs=self.n_jobs
                     )(delayed(self._predict)(seq) for seq in x)
        )


class PageRankIRPredictor(BaseIRPredictor):

    def __init__(
        self,
        tag_corpous: list[list[str]] | sparse.spmatrix,
        n_jobs: int = 0,
        alpha: float = 0.85,
        **kwargs,
    ):
        self.tag_corpus = tag_corpous
        if not isinstance(tag_corpous, sparse.spmatrix):
            self.graph = corpus_utils.to_graph(tag_corpous)  # item->tag
        else:
            self.graph = tag_corpous
        self.homo_graph = bipartite_to_homogeneous(
            self.graph
        )  # ([item,tag] -> [item,tag])
        kwargs.update(alpha=alpha)
        self.kwargs = kwargs
        self.n_jobs = n_jobs
        return

    def _predict(self, x: int):
        d0 = np.zeros(self.graph.shape[0])
        p0 = self.graph.tocsr()[x].toarray().flatten()
        r0 = np.concatenate([d0, p0])
        r = pagerank(self.homo_graph, r0, **self.kwargs)
        return r[:len(d0)]

    def predict_scores(self, x: list[int]):
        """Calculate recommendation(ranking) scores of x (sequence of sequence of product ids)
        """
        if self.n_jobs == 0:
            return np.array(list(map(self._predict, x)))
        return np.array(
            Parallel(n_jobs=self.n_jobs)(delayed(self._predict)(i) for i in x)
        )
