from __future__ import annotations
import os
import abc
from typing import Literal, Mapping, Annotated
from typing_extensions import deprecated
from pydantic import Field
import numpy as np
from ..models.base import RecPredictorT
from ..models import (
    BiRankRecPredictor,
    PageRankRecPredictor,
    BM25RecPredictor,
    BiRankIRPredictor,
    PageRankIRPredictor,
    BM25IRPredictor,
    RandomPredictor,
    PopularItemPredictor,
    ItemDistributionPredictor,
)

from .base import BaseConfig
from .corpus import CorpusConfig


class BasePredictorConfig(BaseConfig, abc.ABC):

    name: str
    n_jobs: int = Field(0, exclude=True)

    @abc.abstractmethod
    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ) -> RecPredictorT:
        ...


class QueryFnConfig(BaseConfig):

    query_type: Literal['union', 'sum']
    """determine how to convert a input sequnece into query vector.
    If 'union', a boolean query vector for entries indicating whether the item
        is in the sequence.
    If 'sum', a query vector for entries indicating how many times the item
        is in the sequence.
    """

    seq_weights: Literal['linear', 'log2', 'exp'] | None = None

    max_seq_len: int | None = None
    """The max len of seq. The query would be
    >>> query = reduce_fn(seq[-max_seq_len:])
    """


class BiRankRecPredictorConfig(BasePredictorConfig):

    name: Literal['birank', 'birank.rec'] = 'birank.rec'

    query_config: QueryFnConfig

    normalizer: Literal['BiRank'] = Field('BiRank', exclude=True)
    """backward"""

    alpha: float
    """dampling factor 1"""
    beta: float
    """dampling factor 2"""

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not isinstance(corpus, Mapping)
        if cache_dir is not None:
            cache_dir = os.path.join(
                cache_dir, global_config.corpus_config.hash
            )
            os.makedirs(cache_dir, exist_ok=True)
        return BiRankRecPredictor(
            corpus,
            query_type=self.query_config.query_type,
            seq_weights=self.query_config.seq_weights,
            max_seq_len=self.query_config.max_seq_len,
            alpha=self.alpha,
            beta=self.beta,
            cache_dir=cache_dir,
            force_reload=force_reload,
            verbose=verbose,
        )


class BiRankIRPredictorConfig(BasePredictorConfig):

    name: Literal['birank.ir'] = 'birank.ir'

    normalizer: Literal['BiRank'] = Field('BiRank', exclude=True)

    alpha: float
    """dampling factor 1"""
    beta: float
    """dampling factor 2"""

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not isinstance(corpus, Mapping)
        if cache_dir is not None:
            cache_dir = os.path.join(
                cache_dir, global_config.corpus_config.hash
            )
            os.makedirs(cache_dir, exist_ok=True)
        return BiRankIRPredictor(
            corpus,
            alpha=self.alpha,
            beta=self.beta,
            cache_dir=cache_dir,
            force_reload=force_reload,
            verbose=verbose,
        )


class BM25RecPredictorConfig(BasePredictorConfig):

    name: Literal['bm25', 'bm25.rec'] = 'bm25.rec'

    k1: float
    """
    k1 : float
        Constant used for influencing the term frequency saturation. After saturation is reached, additional
        presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
        that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
        the type of documents or queries.
    """

    b: float
    """
    b : float
        Constant used for influencing the effects of different document lengths relative to average document length.
        When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
        [1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
        depends on factors such as the type of documents or queries.
    """

    alpha: float
    """
		alpha: float
			IDF cutoff, terms with a lower idf score than alpha will be dropped. A higher alpha will lower the accuracy
			of BM25 but increase performance
    """

    query_config: QueryFnConfig

    document_corpus_config: CorpusConfig | None = Field(None)

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not isinstance(corpus, Mapping)

        document_corpus = (
            self.document_corpus_config
            and list(self.document_corpus_config.init(global_config).values())
        )
        if cache_dir is not None:
            cache_dir = os.path.join(
                cache_dir, global_config.corpus_config.hash
            )
            os.makedirs(cache_dir, exist_ok=True)

        return BM25RecPredictor(
            corpus,
            document_corpus=document_corpus,
            query_fn=self.query_config.query_type,
            seq_weights=self.query_config.seq_weights,
            max_seq_len=self.query_config.max_seq_len,
            k1=self.k1,
            b=self.b,
            alpha=self.alpha,
            cache_dir=cache_dir,
            force_reload=force_reload,
            verbose=verbose,
        )


class BM25IRPredictorConfig(BasePredictorConfig):

    name: Literal['bm25.ir'] = 'bm25.ir'

    k1: float
    """
    k1 : float
        Constant used for influencing the term frequency saturation. After saturation is reached, additional
        presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
        that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
        the type of documents or queries.
    """

    b: float
    """
    b : float
        Constant used for influencing the effects of different document lengths relative to average document length.
        When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
        [1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
        depends on factors such as the type of documents or queries.
    """

    alpha: float
    """
		alpha: float
			IDF cutoff, terms with a lower idf score than alpha will be dropped. A higher alpha will lower the accuracy
			of BM25 but increase performance
    """
    document_corpus_config: CorpusConfig | None = Field(None)

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not isinstance(corpus, Mapping)

        document_corpus = (
            self.document_corpus_config
            and list(self.document_corpus_config.init(global_config).values())
        )
        # XXX: fix this.

        return BM25IRPredictor(
            corpus,
            document_corpus=document_corpus,
            k1=self.k1,
            b=self.b,
            alpha=self.alpha,
        )


@deprecated('use BiRankRec instead')
class PageRankRecPredictorConfig(BasePredictorConfig):

    name: Literal['pagerank', 'pagerank.rec'] = 'pagerank.rec'

    query_type: Literal['union', 'sum']
    """determine how to convert a input sequnece into query vector.
    If 'union', a boolean query vector for entries indicating whether the item
        is in the sequence.
    If 'sum', a query vector for entries indicating how many times the item
        is in the sequence.
    """

    alpha: float
    """damping factor"""

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not isinstance(corpus, Mapping)
        return PageRankRecPredictor(
            corpus,
            query_type=self.query_type,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
        )


@deprecated('use BiRankIR instead')
class PageRankIRPredictorConfig(BasePredictorConfig):

    name: Literal['pagerank.ir'] = 'pagerank.ir'

    alpha: float
    """damping factor"""

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not isinstance(corpus, Mapping)
        return PageRankIRPredictor(
            corpus,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
        )


class RandomPredictorConfig(BasePredictorConfig):

    name: Literal['random'] = 'random'

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not corpus, (
            '"random" use no corpus while an non-empty corpus is passed. '
            'use name="empty" for the corpus instead.'
        )
        return RandomPredictor(corpus)


class ItemDistributionPredictorConfig(BasePredictorConfig):

    name: Literal['item_dist'] = 'item_dist'

    query_config: QueryFnConfig

    # interaction_df: str | pd.DataFrame

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not corpus, (
            '"random" use no corpus while an non-empty corpus is passed. '
            'use name="empty" for the corpus instead.'
        )

        # df = (
        #     self._get_interaction_df(self.interaction_df)
        #     if isinstance(self.interaction_df, str) else self.interaction_df
        # )
        train_x, train_y, *_ = global_config._datasets['train']

        def cat_xy():
            for x, y in zip(train_x, train_y):
                y = np.nonzero(y)[0]
                if len(x.shape) == 0:
                    x = x.reshape(1)
                yield np.concatenate([x, y])

        return ItemDistributionPredictor(
            None,
            user_behavior_seqs=list(cat_xy()),
            n_items=train_y.shape[-1],
            query_type=self.query_config.query_type,
            seq_weights=self.query_config.seq_weights,
            max_seq_len=self.query_config.max_seq_len,
        )


class PopularItemPredictorConfig(BasePredictorConfig):

    name: Literal['popular'] = 'popular'

    # interaction_df: str | pd.DataFrame

    def init(
        self,
        corpus: list[list[str]],
        global_config,
        cache_dir=None,
        force_reload=False,
        verbose=False,
    ):
        assert not corpus, (
            '"random" use no corpus while an non-empty corpus is passed. '
            'use name="empty" for the corpus instead.'
        )

        # df = (
        #     self._get_interaction_df(self.interaction_df)
        #     if isinstance(self.interaction_df, str) else self.interaction_df
        # )
        def cat_xy():
            train_x, train_y, *_ = global_config._datasets['train']
            for x, y in zip(train_x, train_y):
                y = np.nonzero(y)[0]
                if len(x.shape) == 0:
                    x = x.reshape(1)
                yield np.concatenate([x, y])

        return PopularItemPredictor(None, user_behavior_seqs=list(cat_xy()))

    # @classmethod
    # def _get_interaction_df(cls, name: str) -> pd.DataFrame:
    #     obj = data
    #     for n in name.split('.'):
    #         obj = getattr(obj, n)
    #     return obj()

    # @field_validator('interaction_df', mode='after')
    # @classmethod
    # def check_name_valid(cls, v: str):
    #     if isinstance(v, str):
    #         cls._get_interaction_df(v)
    #     return v


PredictorConfigT = Annotated[BM25RecPredictorConfig | BM25IRPredictorConfig
                             | BiRankRecPredictorConfig
                             | BiRankIRPredictorConfig
                             | PageRankRecPredictorConfig
                             | PageRankIRPredictorConfig
                             | RandomPredictorConfig
                             | PopularItemPredictorConfig
                             | ItemDistributionPredictorConfig,
                             Field(discriminator='name')]
