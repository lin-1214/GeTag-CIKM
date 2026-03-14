from typing import Literal
import numpy as np
import pandas as pd

Split = Literal['train', 'val', 'test']


def seqs_to_onehot(
    seqs: list[list[int]], n_dim: int, ignore_duplicate: bool = True
):
    assert len(np.array(seqs).shape) == 2 or np.array(seqs).dtype == object
    to_onehot = np.eye(n_dim)

    onehot = np.array([to_onehot[seq].sum(axis=0) for seq in seqs])
    if ignore_duplicate:
        return onehot.astype(bool)
    return onehot


def pid_seqs_to_iid_seqs(
    seqs: list[list[int]] | list[int], all_pids: list[int]
):
    """chance product id into item id
    - Product id: original id used in the dataset
    - Item id: reindex id from 0~|I|. E.g. all_pids[i] is the product id of item i

    Args:
        seqs (list[list[int]] | list[int]): sequence of sequences
        all_pids (list[int] | None, optional): All product ids.
    """

    assert len(seqs)
    if isinstance(seqs[0], int):
        seqs = [seqs]
    pid_to_iid = {pid: iid for iid, pid in enumerate(all_pids)}

    return [np.array([pid_to_iid[pid] for pid in s]) for s in seqs]


def preparation_for_session_based_last_items_prediction(
    seqs: list[list[int]],
    n_last_items: int,
    min_in_seq_len: int,
    verbose: bool = False,
):
    """Data preparation for session-based last items prediction. This do the following:

    1. Split each sequence in seqs into (x, y)
    2. Drop the sequence if it cannot statisfy `n_last_item` and `min_in_seq_len`.

    Args:
        x (list[list[int]]): sessions. Each session is a sequence of product ids
        n_last_items (int): n items as prediction targets in y.
        min_in_seq_len (int): min length of input sequences. Input sequences x with
            length shorter than min_in_seq_len would be dropped.
    Returns:
        x, y
    """

    seqs_ = [
        np.array(s) for s in seqs if len(s) >= (min_in_seq_len + n_last_items)
    ]

    def get_seq_end_with_unseen_items(seq: list[int]):
        assert len(seq) > n_last_items
        for i in range(n_last_items, len(seq) + 1):
            x, y = seq[:-i], seq[-i:]
            y_set = set(y)
            new_y = y_set - set(x)
            if len(new_y) >= n_last_items:
                assert len(new_y) == n_last_items
                return x, new_y
        if verbose:
            print(f'Dropping invalid seq: {seq}=[{x}, {y}]')
        return [], None

    xs, ys = zip(
        *(
            (x, y) for x, y in map(get_seq_end_with_unseen_items, seqs_)
            if len(x) >= min_in_seq_len
        )
    )
    if verbose:
        print(
            f'{len(seqs) - len(xs)}/{len(seqs)} sequences have dropped '
            f'due to {min_in_seq_len = } and  {n_last_items = }'
        )

    return np.array(xs, dtype=object), np.array(ys, dtype=object)


def preparation_for_session_based_retrieval_task(
    seqs: list[list[int]],
    min_n_relvants: int,
    verbose: bool = False,
):

    def check_if_seq_valid(seq: list[int]):
        x = seq[0]
        ys = seq[1:]
        if len(set(ys) - {x}) < min_n_relvants:
            return None
        return x, ys

    xs, ys = zip(
        *(
            (x_ys[0], x_ys[1]) for x_ys in map(check_if_seq_valid, seqs)
            if x_ys is not None
        )
    )
    if verbose:
        print(
            f'{len(seqs) - len(xs)}/{len(seqs)} sequences have dropped '
            f'due to {min_n_relvants = }.'
        )

    return np.array(xs), np.array(ys, dtype=object)


def preparation_for_ui_based_last_item_prediction(
    inters: dict[Split, pd.Series],
    min_in_seq_len: int,
    verbose: bool = False,
):

    def process(inters: dict[Split, pd.Series]):

        def concat_inters(linters: pd.Series, rinters: pd.Series):

            def concat(s: pd.Series):
                s = s.fillna('')
                return (s['l'] or []) + (s['r'] or [])

            return pd.DataFrame({
                'l': linters,
                'r': rinters,
            }).apply(concat, axis=1)

        accumulated_inters = {'train': inters['train']}
        accumulated_inters['val'] = concat_inters(
            accumulated_inters['train'], inters['val']
        )
        accumulated_inters['test'] = concat_inters(
            accumulated_inters['val'], inters['test']
        )
        for uid, pids in inters['train'].items():
            user_pids = accumulated_inters['test'].loc[uid]
            yield 'train', pids[:-1], pids[-1:], user_pids

        past_inters = inters['train']
        for split in ['val', 'test']:
            for uid, pids in inters[split].items():
                past_pids = past_inters.get(uid, [])
                user_pids = accumulated_inters['test'].loc[uid]
                for i, pid in enumerate(pids):
                    yield split, past_pids + pids[:i], [pid], user_pids
            past_inters = accumulated_inters[split]

    all_df = pd.DataFrame(
        process(inters), columns=['split', 'x', 'y', 'y_pos']
    )
    original_len = len(all_df)
    all_df = all_df[all_df['x'].map(len) >= min_in_seq_len]
    if verbose:
        print(
            f'{original_len - len(all_df)}/{original_len} sequences have dropped '
            f'due to {min_in_seq_len = }'
        )
    dfs = {
        split: all_df[all_df['split'] == split]
        for split in ['train', 'val', 'test']
    }
    return {
        split: (
            np.array(df['x'].map(np.array), dtype=object),
            np.array(df['y'].map(np.array)),
            np.array(df['y_pos'].map(np.array), dtype=object),
        )
        for split, df in dfs.items()
    }
