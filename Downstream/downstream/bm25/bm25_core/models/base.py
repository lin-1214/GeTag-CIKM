from typing import TypeVar
import abc
import numpy as np


class BaseRecPredictor(abc.ABC):

    def __init__(
        self,
        tag_corpous: list[list[str]],
    ):
        ...

    @abc.abstractmethod
    def predict_scores(self, x: list[np.ndarray]) -> np.ndarray:
        """Predict using sequence of item ids as query
        Args:
            x (list[list[int]]): list of item ids
        """


RecPredictorT = TypeVar('RecPredictorT', bound=BaseRecPredictor)


class BaseIRPredictor(abc.ABC):

    def __init__(
        self,
        tag_corpous: list[list[str]],
    ):
        ...

    @abc.abstractmethod
    def predict_scores(self, x: list[int]) -> np.ndarray:
        """Predict using tags of the specified item as query

        Args:
            x (list[int]): list of item id of first item
        """


IRPredictorT = TypeVar('IRPredictorT', bound=BaseIRPredictor)
