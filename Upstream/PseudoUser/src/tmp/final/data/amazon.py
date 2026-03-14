import json
import os
from typing import Literal, get_args
from functools import lru_cache, partial
import pandas as pd

_CACHE_DIR = 'dataset/amazon'

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

    Like i3fresh, we load full sequences (not pre-split).
    The leave-one-last-item split will be done at load time by inters_df().
    """
    assert variant in get_args(VariantT)

    # Map variant name to directory name
    if variant == 'leave_one_last_item':
        dir_name = 'Leave_one_last_item'
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
    split: Literal['train', 'val', 'test', 'all'] | None = None,
    use_iid: bool = True,  # interface alignment
):
    assert variant in get_args(VariantT)
    assert use_iid is True

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

    loaded_pids = df['loaded_pids']
    if use_iid:
        pid_to_iid = {pid: iid for iid, pid in enumerate(all_pids(cache_dir, variant=variant))}
        loaded_pids = loaded_pids.map(
            lambda seq: [pid_to_iid[pid] for pid in seq]
        )

    # Create result dataframe with 'u' column for user/session ID
    result_df = loaded_pids.to_frame()
    result_df['u'] = result_df.index

    # Perform leave-one-last-item split at load time (like i3fresh)
    if split == 'train':
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[:-2])
    elif split == 'val':
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-2:-1])
    else:
        assert split == 'test'
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-1:])

    return result_df


@lru_cache(maxsize=2)
def all_pids(cache_dir: str = _CACHE_DIR, *, variant: VariantT = None):
    assert variant in get_args(VariantT)
    # Map variant name to directory name
    if variant == 'leave_one_last_item':
        dir_name = 'Leave_one_last_item'
    else:
        dir_name = variant.capitalize()

    with open(
        os.path.join(cache_dir, dir_name, 'smap.json')
    ) as fin:
        return list(json.load(fin).values())


def raw_pid_mapping(cache_dir: str = _CACHE_DIR, *, variant: VariantT = None):
    assert variant in get_args(VariantT)
    # Map variant name to directory name
    if variant == 'leave_one_last_item':
        dir_name = 'Leave_one_last_item'
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
        dir_name = 'Leave_one_last_item'
    else:
        dir_name = variant.capitalize()

    with open(
        os.path.join(cache_dir, dir_name, 'base_tags.json'),
        encoding='utf8'
    ) as fin:
        d = {int(pid): tags for pid, tags in json.load(fin).items()}

    if use_pid:
        d = {pid: d[pid] for pid in pids}
        assert list(d) == pids
        return d

    return [d[pid] for pid in pids]


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
