import json
import os
from typing import Literal
from functools import lru_cache, partial
import pandas as pd

_CACHE_DIR = 'dataset/Recipe-Dataset'

VariantT = Literal['small', 'large', 'mini', 'mini_leave_one_last_item']


def session_df(
    cache_dir: str = _CACHE_DIR,
    *,
    variant: VariantT = None,
    split: Literal['train', 'val', 'test', 'all'] | None = None,
):
    assert variant is not None

    if split is None:
        return {
            'train': session_df(cache_dir, split='train', variant=variant),
            'val': session_df(cache_dir, split='val', variant=variant),
            'test': session_df(cache_dir, split='test', variant=variant),
        }
    if split == 'all':
        df = pd.concat(
            [
                session_df(cache_dir, split=split, variant=variant)
                for split in ['train', 'val', 'test']
            ],
            axis=0,
        )
    else:
        path = os.path.join(cache_dir, variant, f'session_{split}.csv')
        df = pd.read_csv(path)
        df.loc[:, 'loaded_pids'] = df['loaded_pids'].map(eval)
    return df


def inters_df(
    cache_dir: str = _CACHE_DIR,
    *,
    variant: VariantT = None,
    split: Literal['train', 'val', 'test', 'all'] | None = None,
):
    assert variant is not None

    if split is None:
        return {
            'train': inters_df(cache_dir, variant=variant, split='train'),
            'val': inters_df(cache_dir, variant=variant, split='val'),
            'test': inters_df(cache_dir, variant=variant, split='test'),
        }

    def get_path(split: str):
        if variant == 'mini_leave_one_last_item':
            return os.path.join(cache_dir, 'mini', f'{split}_loli.csv')
        return os.path.join(cache_dir, variant, f'{split}.csv')

    if split == 'all':
        inter_df = pd.concat(
            [
                pd.read_csv(get_path(split))
                for split in ['train', 'val', 'test']
            ],
            axis=0,
        )

    else:
        inter_df = pd.read_csv(get_path(split))
    inter_df = inter_df.sort_values(by=['u', 'date'])
    df = inter_df.groupby('u')['recipe_id'].apply(list).reset_index()
    df.columns = ['u', 'loaded_pids']
    return df


@lru_cache(maxsize=2)
def all_pids(cache_dir: str = _CACHE_DIR, *, variant: VariantT = None):
    assert variant is not None
    with open(os.path.join(cache_dir, variant, 'all_items.json')) as fin:
        return json.load(fin)


def human_tags(
    cache_dir: str = _CACHE_DIR,
    *,
    variant: VariantT = None,
    use_pid: bool = False,
):
    assert variant is not None
    pids = all_pids(variant=variant)
    with open(os.path.join(cache_dir, variant, 'human_tags.json')) as fin:
        d = {int(pid): tags for pid, tags in json.load(fin).items()}

    if use_pid:
        assert list(d) == pids
        return d

    return [d[pid] for pid in pids]


def base_tags(
    cache_dir: str = _CACHE_DIR,
    *,
    variant: VariantT = None,
    use_pid: bool = False,
):
    assert variant is not None
    pids = all_pids(variant=variant)
    with open(
        os.path.join(cache_dir, variant, 'base_tags.json'), encoding='utf8'
    ) as fin:
        d = {int(pid): tags for pid, tags in json.load(fin).items()}

    if use_pid:
        d = {pid: d[pid] for pid in pids}
        assert list(d) == pids
        return d

    return [d[pid] for pid in pids]


def check_integrity(cache_dir: str = _CACHE_DIR):

    for variant in ['small', 'large', 'mini']:
        human_tags(cache_dir, variant=variant)
        all_pids(cache_dir, variant=variant)
        inters_df(cache_dir, variant=variant)
    inters_df(cache_dir, variant='mini_leave_one_last_item')
    base_tags(cache_dir, variant='small')
    base_tags(cache_dir, variant='mini')
    session_df(cache_dir, variant='mini')
    return None


class _VariantHelper:

    def __init__(self, variant: VariantT):
        self.variant = variant
        return

    def __getattr__(self, __name: str):
        fn = globals().get(__name, None)
        assert fn is not None, f'data: {__name} for recipe does not exist'
        if self.variant == 'mini_leave_one_last_item':
            if __name == 'inters_df':
                return partial(fn, variant='mini_leave_one_last_item')
            return partial(fn, variant='mini')
        return partial(fn, variant=self.variant)


small = _VariantHelper('small')
large = _VariantHelper('large')
mini = _VariantHelper('mini')
mini_leave_one_last_item = _VariantHelper('mini_leave_one_last_item')
mini_loli = mini_leave_one_last_item
