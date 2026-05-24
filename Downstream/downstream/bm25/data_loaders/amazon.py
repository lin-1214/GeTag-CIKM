import json
import os
from typing import Literal, get_args, Optional
from functools import lru_cache, partial
import pandas as pd

_CACHE_DIR = 'data/preprocessed/games/bm25'

VariantT = Literal['scientific', 'leave_one_last_item']


def _read_json_as_df(path: str):
    with open(path, 'r', encoding='utf8') as fin:
        d = {int(i): v for i, v in json.load(fin).items()}

    return pd.DataFrame({
        'u': list(d),
        'loaded_pids': list(d.values())
    })


def load_df(cache_dir: str = _CACHE_DIR, auto_split: bool = True, variant: VariantT = None):
    """Load full sequences from sequences.json.

    Like food, we load full sequences (not pre-split).
    The leave-one-last-item split will be done at load time by inters_df().
    """
    assert variant in get_args(VariantT)

    # Map variant name to directory name
    if variant == 'leave_one_last_item':
        dir_name = '.'
    else:
        dir_name = variant.capitalize()

    # Load full sequences from sequences.json (not pre-split files)
    sequences_file = os.path.join(cache_dir, dir_name, 'sequences.json')
    with open(sequences_file, 'r', encoding='utf8') as f:
        sequences = json.load(f)

    # Convert to DataFrame with integer user IDs as index
    full_sequences = {int(user_id): pids for user_id, pids in sequences.items()}

    df = pd.DataFrame({
        'loaded_pids': list(full_sequences.values())
    }, index=list(full_sequences.keys()))

    return df


def inters_df(
    cache_dir: str = _CACHE_DIR,
    *,
    variant: VariantT = None,
    split: Optional[Literal['train', 'val', 'test', 'all']] = None,
    use_iid: bool = False,  # Default to False (item-based retrieval does conversion separately)
):
    assert variant in get_args(VariantT)

    # Load the full dataset (without pre-splitting)
    df = load_df(cache_dir, auto_split=False, variant=variant)
    # Filter to sequences with at least 3 items (required for train/val/test split)
    df = df[df['loaded_pids'].map(len) >= 3]

    if split is None:
        return {
            'train': inters_df(cache_dir, variant=variant, split='train', use_iid=use_iid),
            'val': inters_df(cache_dir, variant=variant, split='val', use_iid=use_iid),
            'test': inters_df(cache_dir, variant=variant, split='test', use_iid=use_iid),
        }

    if split == 'all':
        assert use_iid is False
        return df

    # Create result dataframe with 'u' column for user/session ID
    result_df = df.copy()
    result_df['u'] = result_df.index

    # Perform leave-one-last-item split at load time (like food)
    if split == 'train':
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[:-2])
    elif split == 'val':
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-2:-1])
    else:
        assert split == 'test'
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-1:])

    # Convert to IIDs if requested (AFTER splitting, like movielens)
    if use_iid:
        pid_to_iid = {pid: iid for iid, pid in enumerate(all_pids(cache_dir, variant=variant))}
        result_df['loaded_pids'] = result_df['loaded_pids'].map(
            lambda seq: [pid_to_iid[pid] for pid in seq]
        )

    return result_df


@lru_cache(maxsize=2)
def all_pids(cache_dir: str = _CACHE_DIR, *, variant: VariantT = None):
    """Returns all product IDs (ASINs in order of their integer index)"""
    assert variant in get_args(VariantT)
    # Map variant name to directory name
    if variant == 'leave_one_last_item':
        dir_name = '.'
    else:
        dir_name = variant.capitalize()

    with open(
        os.path.join(cache_dir, dir_name, 'smap.json')
    ) as fin:
        # Return ASINs in index order for creating pid_to_iid mapping
        smap = json.load(fin)
        # smap is {"0": "ASIN1", "1": "ASIN2", ...}, return values in key order
        return [smap[str(i)] for i in range(len(smap))]


def raw_pid_mapping(cache_dir: str = _CACHE_DIR, *, variant: VariantT = None):
    assert variant in get_args(VariantT)
    # Map variant name to directory name
    if variant == 'leave_one_last_item':
        dir_name = '.'
    else:
        dir_name = variant.capitalize()

    with open(
        os.path.join(cache_dir, dir_name, 'smap.json')
    ) as fin:
        return json.load(fin)


def base_tags(
    cache_dir: str = _CACHE_DIR,
    *,
    variant: VariantT = None,
    use_pid: bool = False,
):
    assert variant is not None
    pids = all_pids(variant=variant)
    # Map variant name to directory name
    if variant == 'leave_one_last_item':
        dir_name = '.'
    else:
        dir_name = variant.capitalize()

    with open(
        os.path.join(cache_dir, dir_name, 'base_tags.json'),
        encoding='utf8'
    ) as fin:
        indexed_tags = {int(idx): tags for idx, tags in json.load(fin).items()}

    if use_pid:
        # Convert to ASIN-indexed dict
        d = {pids[idx]: indexed_tags[idx] for idx in indexed_tags.keys()}
        return d

    # Return integer-indexed dict (default)
    return indexed_tags


def check_integrity(cache_dir: str = _CACHE_DIR):

    for variant in ['scientific']:
        base_tags(cache_dir, variant=variant)
        all_pids(cache_dir, variant=variant)
        inters_df(cache_dir, variant=variant)
    return None


class _VariantHelper:

    def __init__(self, variant: VariantT):
        self.variant = variant
        return

    def __getattr__(self, __name: str):
        fn = globals().get(__name, None)
        assert fn is not None, f'data: {__name} for recipe does not exist'
        return partial(fn, variant=self.variant)


scientific = _VariantHelper('scientific')
leave_one_last_item = _VariantHelper('leave_one_last_item')
loli = leave_one_last_item
