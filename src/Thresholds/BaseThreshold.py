from abc import ABC, abstractmethod


class BaseThreshold(ABC):
    @abstractmethod
    def initialize(self, scores):
        raise NotImplementedError

    @abstractmethod
    def getThreshold(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, score, index=None):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def isReady(self):
        raise NotImplementedError

    @abstractmethod
    def getState(self):
        raise NotImplementedError