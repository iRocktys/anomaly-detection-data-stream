from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseThreshold(ABC):
    @abstractmethod
    def initialize(self, scores: Iterable[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def getThreshold(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, score: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def isReady(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def getState(self) -> dict[str, Any]:
        raise NotImplementedError
