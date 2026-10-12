from abc import ABC, abstractmethod


class BaseThreshold(ABC):
    @abstractmethod
    def initialize(self, scores):
        # Define a calibração inicial obrigatória de uma estratégia de threshold.
        raise NotImplementedError

    @abstractmethod
    def getThreshold(self):
        # Define a leitura do limiar causal atualmente disponível.
        raise NotImplementedError

    @abstractmethod
    def update(self, score, index=None):
        # Define a atualização incremental após a decisão da instância atual.
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        # Define a limpeza completa do estado antes de uma nova execução.
        raise NotImplementedError

    @abstractmethod
    def isReady(self):
        # Define a indicação de que o threshold já pode produzir decisões.
        raise NotImplementedError

    @abstractmethod
    def getState(self):
        # Define a serialização resumida do estado interno do threshold.
        raise NotImplementedError
