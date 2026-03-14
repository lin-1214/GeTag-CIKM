from __future__ import annotations
import json
import os
from itertools import product
from functools import cached_property
import hashlib
from filelock import FileLock
from typing import ClassVar
from pydantic import Field
import naive_flow as nf
import numpy as np
import pandas as pd
import final
from .predictor import PredictorConfigT
from .dataset import DatasetConfigT
from .base import BaseConfig
from .corpus import CorpusConfig


class ExperimentConfig(BaseConfig):

    exp_csv_name_template: ClassVar[str] = 'exps.v2.{}.csv'
    env_log_dir: ClassVar[str] = 'env_logs'
    corpus_log_dir: ClassVar[str] = 'corpus_logs'
    dataset_cache_dir: ClassVar[str] = 'datasets_cache'
    predictor_cache_dir: ClassVar[str] = 'predictor_cache'

    predictor_config: PredictorConfigT
    corpus_config: CorpusConfig
    dataset_config: DatasetConfigT
    ks: list[int] = [5, 10, 20]
    n_negative_samples: list[int] = [-1, 100]
    """a list of k for ranking metrics"""
    comment: str | None = Field(None)
    """comment to the experiment"""

    def eval(self, cache_dir=None, force_rerun=False, verbose=False):

        if not force_rerun:
            metrics = self.check_cache(cache_dir)
            if metrics is not None:
                if verbose:
                    print(f'cache for experiment: {self.hash} found.')
                return metrics

        self._datasets = self.dataset_config.init(
            self,
            cache_dir=(
                None if cache_dir is None else
                os.path.join(cache_dir, self.dataset_cache_dir)
            ),
            force_reload=force_rerun,
            verbose=verbose,
        )
        corpus = self.corpus_config.init(
            self,
            cache_dir=(
                None if cache_dir is None else
                os.path.join(cache_dir, self.corpus_log_dir)
            ),
            force_reload=force_rerun,
            verbose=verbose,
        )
        predictor = self.predictor_config.init(
            list(corpus.values()),
            self,
            cache_dir=(
                None if cache_dir is None else
                os.path.join(cache_dir, self.predictor_cache_dir)
            ),
            force_reload=force_rerun,
            verbose=verbose,
        )

        metrics = {}
        for split, xy in self._datasets.items():
            x, y, prediction_domain, *_ = (*xy, None)
            assert prediction_domain is not None
            if x.dtype == int and len(x.shape) == 1:
                assert 'ir' in self.dataset_config.name
                x = x.reshape(-1, 1)

            item_scores = predictor.predict_scores(x)

            for n_negatives in self.n_negative_samples:
                if n_negatives > 0:
                    assert len(prediction_domain.shape) == 2
                    tem = prediction_domain.copy()
                    tem[y] = False
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
                mets = final.ranking_metrics(
                    item_scores,
                    y,
                    prediction_domain=(y | negative_domain),
                    ks=self.ks,
                )
                mets = {
                    f'{key}{metrics_suffix}/{split}': val
                    for key, val in mets.items()
                }
                metrics.update(mets)

        metrics['hash'] = self.hash
        metrics['corpus_hash'] = self.corpus_config.hash
        metrics['comment'] = self.comment or ' '
        metrics['names'] = '.'.join(
            [
                self.dataset_config.name, self.corpus_config.name,
                self.predictor_config.name
            ]
        )
        if cache_dir is None:
            return metrics
        col_order = [
            f'{name}@{k}{suffix}/{split}'
            for suffix, split, name, k in product(
                [f'/{n}' if n > 0 else '' for n in self.n_negative_samples],
                self._datasets,
                ['hr', 'mrr', 'ndcg'],
                self.ks,
            )
        ]
        columns = ['hash', 'comment', 'names', 'corpus_hash', *col_order]

        df = pd.DataFrame([metrics], columns=columns).set_index('hash')

        out_file = os.path.join(cache_dir, self.exp_csv_name)
        with self.lock(out_file):
            if not os.path.exists(out_file):
                df.to_csv(out_file)
                return df.loc[self.hash]

            original_df = pd.read_csv(
                out_file, index_col='hash', usecols=columns
            )
            original_df.loc[self.hash] = df.loc[self.hash]
            original_df.to_csv(out_file)
        ENV_LOG_DIR = os.path.join(cache_dir, self.env_log_dir)
        os.makedirs(ENV_LOG_DIR, exist_ok=True)
        nf.dump_config(
            self, os.path.join(ENV_LOG_DIR, f'{self.hash}.env'),
            description='full'
        )
        self._datasets = {}
        return df.loc[self.hash]

    def check_cache(self, cache_dir):
        if cache_dir is None:
            return None
        path = os.path.join(cache_dir, self.exp_csv_name)
        if not os.path.exists(path):
            return None
        with self.lock(path):
            df = pd.read_csv(path, index_col='hash')
        if self.hash not in df.index:
            return None
        return df.loc[self.hash]

    @cached_property
    def hash(self):
        d = self.model_dump()
        del d['comment']
        s = json.dumps(d) + self.dataset_config.hash + self.corpus_config.hash
        return hashlib.sha1(s.encode()).hexdigest()

    @property
    def exp_csv_name(self):
        return self.exp_csv_name_template.format(self.dataset_config.name)

    def lock(self, out_file: str):
        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)
        basename = os.path.basename(out_file)
        return FileLock(os.path.join(out_dir, 'locks', f'{basename}.lock'))
