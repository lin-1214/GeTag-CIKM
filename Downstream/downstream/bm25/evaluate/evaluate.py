from typing import Literal, Optional, Union
import pandas as pd
import numpy as np


def _score_dict_to_arr(score_dict: dict[int, float], n_items: int):
    scores = np.random.rand(n_items) - 11.
    for pid, score in score_dict.items():
        scores[pid] = score
    return scores


def ranking_metrics(
    y_score: Union[np.ndarray, pd.Series],
    y_true: Union[np.ndarray, pd.Series],
    prediction_domain: Optional[np.ndarray] = None,
    ks: Union[list[int], int] = None,
    sample_weight: Optional[np.ndarray] = None,
):
    """Calculate the ranking metrics, including hr, mrr, ndcg

    Args:
        y_score (np.ndarray): predicted scores in shape (#n_samples, #n_products)
        y_true (np.ndarray): ground truth in shape (#n_samples, #n_products)
        ks (Union[list[int], int]): list of k to use. Default to [1, 5, 10, 20]

    Returns:
        dict
    """

    if isinstance(y_score, pd.Series):
        y_score = np.array(y_score.to_list())
    elif not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)
    if isinstance(y_true, pd.Series):
        y_true = np.array(y_true.to_list())
    if len(y_score.shape) == 1:
        assert isinstance(y_score[0], dict)
        y_score = np.array(
            [
                _score_dict_to_arr(score_dict, y_true.shape[-1])
                for score_dict in y_score
            ]
        )

    assert y_score.shape == y_true.shape
    if ks is None:
        ks = [5, 10, 20]
    elif isinstance(ks, int):
        ks = [ks]

    if prediction_domain is not None:
        assert len(prediction_domain.shape) in [1, 2]
        assert prediction_domain.dtype == bool
        y_score = y_score.copy()
        y_true = y_true.copy()
        y_score[..., ~prediction_domain] = -np.inf
        y_true[..., ~prediction_domain] = 0
    y_rank = np.argsort(-y_score, axis=1)
    mrr_weights = 1 / (np.arange(max(ks)) + 1)
    dcg_weights = 1 / np.log2(2 + np.arange(max(ks)))
    idcg_cumulative_weights = dcg_weights.cumsum()

    if sample_weight is not None:
        sample_weight /= sample_weight.sum()
        reduce_fn = lambda x: (x * sample_weight).sum()
    else:
        reduce_fn = lambda x: x.mean()

    def _cal_metrics():
        for k in ks:
            y_rank_at_k = y_rank[:, :k]
            y_rank_true_at_k = np.take_along_axis(y_true, y_rank_at_k, axis=1)
            n_true = y_true.sum(axis=1).clip(max=k)
            dcg_ = (dcg_weights[:k] * y_rank_true_at_k).sum(axis=1)
            idcg_ = idcg_cumulative_weights[n_true - 1]

            ndcg = reduce_fn(dcg_ / idcg_)
            mrr = reduce_fn((mrr_weights[:k] * y_rank_true_at_k).sum(axis=1))
            hr = reduce_fn(y_rank_true_at_k.sum(axis=1) / n_true)
            yield f'hr@{k}', hr
            yield f'mrr@{k}', mrr
            yield f'ndcg@{k}', ndcg

    order = [f'{met}@{k}' for met in ['hr', 'mrr', 'ndcg'] for k in ks]
    res = dict(_cal_metrics())
    return {key: res[key] for key in order}


def ranking_metrics_with_pseudo_negative_sampling(
    y_score: Union[np.ndarray, pd.Series],
    y_true: Union[np.ndarray, pd.Series],
    prediction_domain: Optional[np.ndarray] = None,
    ks: Union[list[int], Optional[int]] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_negative_samples: Optional[list[int]] = None,
):
    """Calculate the ranking metrics, including hr, mrr, ndcg

    Args:
        y_score (np.ndarray): predicted scores in shape (#n_samples, #n_products)
        y_true (np.ndarray): ground truth in shape (#n_samples, #n_products)
        ks (Union[list[int], int]): list of k to use. Default to [1, 5, 10, 20]

    Returns:
        dict
    """

    if n_negative_samples is None:
        n_negative_samples = [-1]
    assert len(set(n_negative_samples)) == len(n_negative_samples)
    if isinstance(y_score, pd.Series):
        y_score = np.array(y_score.to_list())
    elif not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)
    if isinstance(y_true, pd.Series):
        y_true = np.array(y_true.to_list())
    if len(y_score.shape) == 1:
        assert isinstance(y_score[0], dict)
        y_score = np.array(
            [
                _score_dict_to_arr(score_dict, y_true.shape[-1])
                for score_dict in y_score
            ]
        )

    metrics = {}
    for n_negatives in n_negative_samples:
        if n_negatives > 0:
            assert len(prediction_domain.shape) == 2
            tem = prediction_domain.copy()
            tem[y_true] = False
            negative_domain_indices = np.stack(
                [
                    np.random.choice(
                        np.nonzero(a)[0],
                        size=n_negatives,
                        replace=False,
                    ) for a in tem
                ]
            )
            negative_domain = np.zeros_like(prediction_domain)
            np.put_along_axis(
                negative_domain, negative_domain_indices, True, axis=1
            )
            metrics_suffix = f'/{n_negatives}'
        else:
            negative_domain = prediction_domain
            metrics_suffix = ''
        mets = ranking_metrics(
            y_score=y_score,
            y_true=y_true,
            prediction_domain=(y_true | negative_domain),
            ks=ks,
            sample_weight=sample_weight,
        )
        mets = {f'{key}{metrics_suffix}': val for key, val in mets.items()}
        metrics.update(mets)
    return metrics
