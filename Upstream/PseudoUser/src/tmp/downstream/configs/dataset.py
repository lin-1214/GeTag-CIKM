import abc
from typing import Annotated
import os
from functools import cached_property
import hashlib
from typing import Literal
import numpy as np
from pydantic import Field, model_validator
import final
from final.utils import Split
from .base import BaseConfig

_X_T = np.ndarray
_Y_T = np.ndarray
_PredictionDomain_T = np.ndarray
XY = tuple[_X_T, _Y_T] | tuple[_X_T, _Y_T, _PredictionDomain_T]


def _recursive_getattr(obj, name, default_value):
    for n in name.split('.'):
        res = getattr(obj, n, None)
        if res is None:
            return default_value
        obj = res
    return res


class BaseDatasetConfig(BaseConfig, abc.ABC):

    def init(
        self,
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ) -> dict[Split, XY]:

        if cache_dir is not None and not force_reload:
            datasets = self._load(cache_dir)
            if datasets is not None:
                if verbose:
                    print('cache for datasets found. loaded from cache.')
                return datasets

        datasets = self._init_dataset(global_config, verbose=verbose)
        if cache_dir is not None:
            self._save(datasets, cache_dir, verbose)
        return datasets

    @abc.abstractmethod
    def _init_dataset(self, global_config, verbose=False) -> dict[Split, XY]:
        ...

    @cached_property
    def hash(self):
        return hashlib.sha1(self.model_dump_json().encode()).hexdigest()

    def _save(self, dataset: dict[str, XY], cache_dir, verbose):
        if cache_dir is None:
            return

        outdir = {}
        for split, xy in dataset.items():
            x, y, prediction_domain, *_ = (*xy, None)
            outdir[f'{split}_x'] = x
            outdir[f'{split}_y'] = y
            if prediction_domain is not None:
                outdir[f'{split}_prediction_domain'] = prediction_domain

        os.makedirs(cache_dir, exist_ok=True)
        file = os.path.join(cache_dir, f'{self.hash}.npz')
        np.savez(file, **outdir)
        if verbose:
            print(f'datasets have been saved in {file}')
        return

    def _load(self, cache_dir):
        if cache_dir is None:
            return None
        file = os.path.join(cache_dir, f'{self.hash}.npz')
        if not os.path.exists(file):
            return None
        datasets = {}
        npzfile = np.load(file, allow_pickle=True)
        for split in ['train', 'val', 'test']:
            if f'{split}_prediction_domain' in npzfile:
                datasets[split] = (
                    npzfile[f'{split}_x'],
                    npzfile[f'{split}_y'],
                    npzfile[f'{split}_prediction_domain'],
                )
            else:
                datasets[split] = (
                    npzfile[f'{split}_x'], npzfile[f'{split}_y']
                )
        return datasets


class UnseenLastItemsPredictionDatasetConfig(BaseDatasetConfig):

    name: Literal['i3fresh.last'] = 'i3fresh.last'
    """name of the dataset. one of ['i3fresh']"""

    n_last_items: int
    """number last items used as the prediction target"""

    min_in_seq_len: int
    """minimum length of the input sequences (x). Sequences with x having length
    shorter than min_in_seq_len would be dropped.
    """

    def _init_dataset(self, global_config, verbose=False):
        datasets = {}

        assert 'i3fresh' in self.name
        all_pids = final.data.i3fresh.AVAILABLE_PRODUCT_IDS
        dfs = final.data.i3fresh.load_df()
        for split, df in dfs.items():
            x, y = final.utils.preparation_for_session_based_last_items_prediction(
                final.utils.pid_seqs_to_iid_seqs(
                    df['loaded_pids'].tolist(), all_pids=all_pids
                ),
                n_last_items=self.n_last_items,
                min_in_seq_len=self.min_in_seq_len,
                verbose=verbose,
            )
            y = np.array([list(y_) for y_ in y])
            y = final.utils.seqs_to_onehot(y, len(all_pids))
            datasets[split] = (x, y)
        return datasets


class RetrievalTaskDatasetConfig(BaseDatasetConfig):
    """Tags of first item as query, the rest items as relevant documents.
    """
    name: Literal['i3fresh.ir', 'recipe.small.ir', 'movielens.ir']

    min_n_relvants: int
    """minimun number of relvant documents"""

    max_n_relvants: int | None = None

    def _init_dataset(self, global_config, verbose=False):
        datasets = {}

        assert self.name.endswith('.ir')
        if 'i3fresh' in self.name:
            all_pids = final.data.i3fresh.AVAILABLE_PRODUCT_IDS
            dfs = final.data.i3fresh.load_df()
        else:
            getter = _recursive_getattr(final.data, self.name[:-3], None)
            assert getter is not None
            dfs = getter.inters_df()
            all_pids = getter.all_pids()

        for split, df in dfs.items():
            x, y = final.utils.preparation_for_session_based_retrieval_task(
                final.utils.pid_seqs_to_iid_seqs(
                    df['loaded_pids'].tolist(), all_pids=all_pids
                ),
                min_n_relvants=self.min_n_relvants,
                verbose=verbose,
            )
            y = final.utils.seqs_to_onehot(y, len(all_pids))
            datasets[split] = (x, y)
        return datasets


class UserItemLastItemPredictionDatasetConfig(BaseDatasetConfig):

    name: Literal[
        'recipe.mini_leave_one_last_item.last',
        'i3fresh.leave_one_last_item.last',
        'pseudouser.leave_one_last_item.last',
        'movielens.last',
        'movielens.leave_one_last_item.last',
        'amazon.leave_one_last_item.last',
    ]

    min_in_seq_len: int
    """minimum length of the input sequences (x). Sequences with x having length
    shorter than min_in_seq_len would be dropped.
    """

    max_in_seq_len: int | None = None
    """minimum length of the input sequences (x). Sequences with x having length
    longer than min_in_seq_len would be truncated.
    """

    repeatable: bool = False

    def _init_dataset(self, global_config, verbose=False):

        assert self.name.endswith('.last')
        data_getter = _recursive_getattr(
            final.data, self.name[:-len('.last')], None
        )
        assert data_getter is not None, f'dataset {self.name[:-len(".last")]} does not exist.'
        all_pids = data_getter.all_pids()
        dfs = data_getter.inters_df()
        pid_to_iid = {pid: iid for iid, pid in enumerate(all_pids)}
        results = final.utils.preparation_for_ui_based_last_item_prediction(
            {
                split:
                df.set_index('u')['loaded_pids'].
                map(lambda seq: [pid_to_iid[pid] for pid in seq])
                for split, df in dfs.items()
            },
            min_in_seq_len=self.min_in_seq_len,
            verbose=verbose,
        )

        datasets = {}
        for split, (x, y, y_pos) in results.items():
            if self.max_in_seq_len is not None:
                x = np.array([seq[-self.max_in_seq_len:] for seq in x])
            y = final.utils.seqs_to_onehot(y, len(all_pids))
            if not self.repeatable:
                prediction_domain = ~final.utils.seqs_to_onehot(
                    y_pos, len(all_pids)
                )
                prediction_domain[y] = True
            else:
                prediction_domain = np.ones_like(y, dtype=bool)
            datasets[split] = (x, y, prediction_domain)

        return datasets

    @model_validator(mode='after')
    def check_repetable(self):
        if self.repeatable:
            assert 'movielens' not in self.name
        else:
            # For movielens, repeatable must be False; for others, it's flexible
            if 'movielens' in self.name:
                pass  # movielens requires non-repeatable
            elif 'pseudouser' in self.name:
                pass  # pseudouser can be repeatable or not
        return self


class UserItemLastItemRetrievalDatasetConfig(
    UserItemLastItemPredictionDatasetConfig
):
    """Similar to last item prediction (standard sequential rec), but using
    second last item to predict the last item"""

    name: Literal['recipe.mini_leave_one_last_item.lastir',
                  'i3fresh.leave_one_last_item.lastir',
                  'movielens.global_temporal.lastir',
                  'movielens.leave_one_last_item.lastir',
                  'amazon.leave_one_last_item.lastir']

    min_in_seq_len: int = Field(1, le=1, ge=1, exclude=True)
    max_in_seq_len: int = Field(1, le=1, ge=1, exclude=True)

    def _init_dataset(self, global_config, verbose=False):

        assert self.name.endswith('.lastir')
        data_getter = _recursive_getattr(
            final.data, self.name[:-len('.lastir')], None
        )
        assert data_getter is not None, f'dataset {self.name[:-len(".lastir")]} does not exist.'
        all_pids = data_getter.all_pids()
        dfs = data_getter.inters_df()
        pid_to_iid = {pid: iid for iid, pid in enumerate(all_pids)}
        results = final.utils.preparation_for_ui_based_last_item_prediction(
            {
                split:
                df.set_index('u')['loaded_pids'].
                map(lambda seq: [pid_to_iid[pid] for pid in seq])
                for split, df in dfs.items()
            },
            min_in_seq_len=self.min_in_seq_len,
            verbose=verbose,
        )

        datasets = {}
        for split, (x, y, y_pos) in results.items():
            x = np.array([seq[-1] for seq in x])
            y = final.utils.seqs_to_onehot(y, len(all_pids))
            if not self.repeatable:
                prediction_domain = ~final.utils.seqs_to_onehot(
                    y_pos, len(all_pids)
                )
                prediction_domain[y] = True
            else:
                prediction_domain = np.ones_like(y, dtype=bool)
            datasets[split] = (x, y, prediction_domain)

        return datasets


DatasetConfigT = Annotated[
    UnseenLastItemsPredictionDatasetConfig | RetrievalTaskDatasetConfig
    | UserItemLastItemPredictionDatasetConfig
    | UserItemLastItemRetrievalDatasetConfig,
    Field(discriminator='name'),
]
