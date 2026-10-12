from abc import ABC, abstractmethod


class BaseThreshold(ABC):
    @abstractmethod
    def initialize(self, scores):
        # Inicializa a estratégia de threshold com os scores de calibração disponíveis.
        raise NotImplementedError

    @abstractmethod
    def getThreshold(self):
        # Retorna o valor atual do threshold utilizado para decisão.
        raise NotImplementedError

    @abstractmethod
    def update(self, score, index=None):
        # Atualiza o estado incremental com os valores da instância atual.
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        # Reinicia o estado interno para permitir uma nova execução.
        raise NotImplementedError

    @abstractmethod
    def isReady(self):
        # Informa se a estratégia de threshold está pronta para avaliar novas instâncias.
        raise NotImplementedError

    @abstractmethod
    def getState(self):
        # Retorna um resumo do estado atual da estratégia de threshold.
        raise NotImplementedError
