'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''

import numpy as np

def Recall_at_k(y_true, y_pred, k, agg="sum"):
    '''
        Average recall for top k recommended results.
        The training records should be set to -inf in y_pred
    '''
    batch_size = y_pred.shape[0]
    topk_idxes = np.argpartition(-y_pred, k, axis=1)[:, :k]
    y_pred_bin = np.zeros_like(y_pred, dtype=bool)
    y_pred_bin[np.arange(batch_size)[:, None], topk_idxes] = True
    y_true_bin = (y_true > 0)

    hits = np.sum(np.logical_and(y_true_bin, y_pred_bin), axis=-1).astype(np.float32)
    num_pos = np.sum(y_true_bin, axis=1)
    denom = np.minimum(k, num_pos)

    # Guard against rows with zero positives to avoid division by zero
    valid_mask = denom > 0
    recalls = np.zeros(batch_size, dtype=np.float32)
    recalls[valid_mask] = hits[valid_mask] / denom[valid_mask]

    if agg == "sum":
        recall = np.sum(recalls)
    elif agg == "mean":
        # Mean over valid users only to avoid biasing by zero-positives
        recall = np.mean(recalls[valid_mask]) if np.any(valid_mask) else 0.0
    else:
        raise NotImplementedError(f"aggregation method {agg} not defined!")
    return recall


def NDCG_at_k(y_true, y_pred, k, agg="sum"):
    '''
        Average NDCG for top k recommended results. 
        The training records should be set to -inf in y_pred
    '''

    batch_size = y_pred.shape[0]
    topk_idxes_unsort = np.argpartition(-y_pred, k, axis=1)[:, :k]
    topk_value_unsort = y_pred[np.arange(batch_size)[:, None],topk_idxes_unsort]
    topk_idxes_rel = np.argsort(-topk_value_unsort, axis=1)
    topk_idxes = topk_idxes_unsort[np.arange(batch_size)[:, None], topk_idxes_rel]
    y_true_topk = y_true[np.arange(batch_size)[:, None], topk_idxes]
    y_true_bin = (y_true > 0).astype(np.float32)
    weights = 1./np.log2(np.arange(2, k + 2))
    DCG = np.sum(y_true_topk*weights, axis=-1)

    num_pos = np.sum(y_true_bin, axis=-1)
    ideal_lengths = np.minimum(k, num_pos).astype(int)
    normalizer = np.array([np.sum(weights[:n]) for n in ideal_lengths])

    # Guard against division by zero for users with zero positives
    valid_mask = normalizer > 0
    ndcg = np.zeros(batch_size, dtype=np.float32)
    ndcg[valid_mask] = DCG[valid_mask] / normalizer[valid_mask]

    if agg == "sum":
        NDCG = np.sum(ndcg)
    elif agg == "mean":
        NDCG = np.mean(ndcg[valid_mask]) if np.any(valid_mask) else 0.0
    else:
        raise NotImplementedError(f"aggregation method {agg} not defined!")
    return NDCG
