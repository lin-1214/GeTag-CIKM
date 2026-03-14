import json
import os
from typing import Literal, get_args, Optional
from functools import lru_cache, partial
import pandas as pd

_CACHE_DIR = 'dataset/movielens/ml-1m'
_CACHE_DIR_V2 = 'dataset/movielens'

VariantT = Literal['global_temporal', 'leave_one_last_item']


def session_df(
    cache_dir: str = _CACHE_DIR,
    *,
    split: Optional[Literal['train', 'val', 'test', 'all']] = None,
):
    # assert variant is not None

    if split is None:
        return {
            'train': session_df(cache_dir, split='train'),
            'val': session_df(cache_dir, split='val'),
            'test': session_df(cache_dir, split='test'),
        }
    if split == 'all':
        df = pd.concat(
            [
                session_df(cache_dir, split=split)
                for split in ['train', 'val', 'test']
            ],
            axis=0,
        )
    else:
        path = os.path.join(cache_dir, f'session_{split}.csv')
        df = pd.read_csv(path)
        df.loc[:, 'loaded_pids'] = df['loaded_pids'].map(eval)
    return df


def inters_df(
    cache_dir: str = _CACHE_DIR,
    *,
    variant: VariantT = None,
    split: Optional[Literal['train', 'val', 'test', 'all']] = None,
    group_by_user: bool = True,
    use_iid: bool = False,
):
    assert variant in get_args(VariantT)

    if split is None:
        return {
            'train': inters_df(cache_dir, split='train', variant=variant),
            'val': inters_df(cache_dir, split='val', variant=variant),
            'test': inters_df(cache_dir, split='test', variant=variant),
        }
    name_suffix = '_loli' if variant == 'leave_one_last_item' else ''
    if split == 'all':
        assert use_iid is False
        inter_df = pd.concat(
            [
                pd.read_csv(
                    os.path.join(cache_dir, f'{split}{name_suffix}.csv')
                ) for split in ['train', 'val', 'test']
            ],
            axis=0,
        )
    else:
        path = os.path.join(cache_dir, f'{split}{name_suffix}.csv')
        inter_df = pd.read_csv(path)
    df = inter_df.sort_values(by=['userId', 'timestamp'])
    if use_iid:
        pid_to_iid = {pid: iid for iid, pid in enumerate(all_pids(cache_dir))}
        df.loc[:, 'movieId'] = df['movieId'].map(pid_to_iid)
    if group_by_user is True:
        df = df.groupby('userId')['movieId'].apply(list).reset_index()
        df.columns = ['u', 'loaded_pids']
    return df


@lru_cache(maxsize=1)
def all_pids(cache_dir: str = _CACHE_DIR):
    with open(os.path.join(cache_dir, 'all_items.json')) as fin:
        return json.load(fin)


def human_tags(
    cache_dir: str = _CACHE_DIR,
    use_pid: bool = False,
):
    # assert variant is not None
    pids = all_pids()
    with open(os.path.join(cache_dir, 'human_tags.json')) as fin:
        d = {int(pid): tags for pid, tags in json.load(fin).items()}

    if use_pid:
        d = {pid: d[pid] for pid in pids}
        assert list(d) == pids
        return d

    return [d[pid] for pid in pids]


def base_tags(
    cache_dir: str = _CACHE_DIR,
    *,
    use_pid: bool = False,
):
    pids = all_pids()
    with open(os.path.join(cache_dir, 'base_tags.json')) as fin:
        d = {int(pid): tags for pid, tags in json.load(fin).items()}

    if use_pid:
        d = {pid: d[pid] for pid in pids}
        assert list(d) == pids
        return d

    return [d[pid] for pid in pids]


def check_integrity(cache_dir: str = _CACHE_DIR):

    human_tags(cache_dir)
    all_pids(cache_dir)
    inters_df(cache_dir, variant='leave_one_last_item')
    base_tags(cache_dir)
    session_df(cache_dir)
    return None


class _VariantHelper:
    """Helper to load from variant-specific directories like Leave_one_last_item"""

    def __init__(self, variant_name: str):
        self.variant_name = variant_name

    def inters_df(
        self,
        cache_dir: str = _CACHE_DIR_V2,
        *,
        split: Optional[Literal['train', 'val', 'test', 'all']] = None,
        use_iid: bool = False,
    ):
        """Load interactions from variant directory (e.g., Leave_one_last_item)"""
        # Map variant name to directory name
        if self.variant_name == 'leave_one_last_item':
            dir_name = 'Leave_one_last_item'
        elif self.variant_name == 'global_temporal':
            dir_name = 'Global_temporal'
        else:
            raise ValueError(f'Unknown variant: {self.variant_name}')

        if split is None:
            return {
                'train': self.inters_df(cache_dir, split='train', use_iid=use_iid),
                'val': self.inters_df(cache_dir, split='val', use_iid=use_iid),
                'test': self.inters_df(cache_dir, split='test', use_iid=use_iid),
            }

        # Load full sequences from sequences.json
        sequences_file = os.path.join(cache_dir, dir_name, 'sequences.json')
        with open(sequences_file, 'r', encoding='utf8') as f:
            sequences = json.load(f)

        # Create dataframe from sequences
        # Convert string movie IDs to integers to match all_pids() and corpus
        result_df = pd.DataFrame([
            {'u': uid, 'loaded_pids': [int(pid) for pid in pids]}
            for uid, pids in sequences.items()
        ])

        # Perform leave-one-last-item split at load time
        if self.variant_name == 'leave_one_last_item':
            if split == 'train':
                # Everything except last 2 items
                result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[:-2])
            elif split == 'val':
                # Second to last item
                result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-2:-1])
            elif split == 'test':
                # Last item
                result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-1:])
            elif split == 'all':
                # Keep all items
                pass
        elif self.variant_name == 'global_temporal':
            # For global temporal, we might need different logic
            # For now, just use the full sequences
            pass

        # Convert to IIDs if requested
        if use_iid:
            pid_to_iid = {pid: iid for iid, pid in enumerate(self.all_pids(cache_dir))}
            result_df['loaded_pids'] = result_df['loaded_pids'].map(
                lambda pids: [pid_to_iid[pid] for pid in pids]
            )

        return result_df

    def all_pids(self, cache_dir: str = _CACHE_DIR_V2):
        """Load all movie IDs from smap.json"""
        # Map variant name to directory name
        if self.variant_name == 'leave_one_last_item':
            dir_name = 'Leave_one_last_item'
        elif self.variant_name == 'global_temporal':
            dir_name = 'Global_temporal'
        else:
            raise ValueError(f'Unknown variant: {self.variant_name}')

        smap_file = os.path.join(cache_dir, dir_name, 'smap.json')
        with open(smap_file, 'r', encoding='utf8') as fin:
            smap = json.load(fin)
            # smap is {index: movie_id}, we want list of movie_ids in index order
            # Convert to int to match corpus keys (CorpusConfig converts all PIDs to int)
            return [int(smap[str(i)]) for i in range(len(smap))]

    def base_tags(
        self,
        cache_dir: str = _CACHE_DIR_V2,
        *,
        use_pid: bool = False,
    ):
        """Load base tags from base_tags.json"""
        # Map variant name to directory name
        if self.variant_name == 'leave_one_last_item':
            dir_name = 'Leave_one_last_item'
        elif self.variant_name == 'global_temporal':
            dir_name = 'Global_temporal'
        else:
            raise ValueError(f'Unknown variant: {self.variant_name}')

        pids = self.all_pids(cache_dir)  # Now returns integers
        tags_file = os.path.join(cache_dir, dir_name, 'base_tags.json')
        with open(tags_file, 'r', encoding='utf8') as fin:
            # Convert to int to match all_pids and corpus keys
            d = {int(pid): tags for pid, tags in json.load(fin).items()}

        if use_pid:
            result = {pid: d.get(pid, []) for pid in pids}
            assert list(result) == pids
            return result

        return [d.get(pid, []) for pid in pids]

    def human_tags(
        self,
        cache_dir: str = _CACHE_DIR,
        *,
        use_pid: bool = False,
    ):
        """Delegate to the global human_tags function for backward compatibility"""
        return human_tags(cache_dir, use_pid=use_pid)

    def session_df(
        self,
        cache_dir: str = _CACHE_DIR,
        *,
        split: Optional[Literal['train', 'val', 'test', 'all']] = None,
    ):
        """Delegate to the global session_df function for backward compatibility"""
        return session_df(cache_dir, split=split)


leave_one_last_item = _VariantHelper('leave_one_last_item')
global_temporal = _VariantHelper('global_temporal')
loli = leave_one_last_item
