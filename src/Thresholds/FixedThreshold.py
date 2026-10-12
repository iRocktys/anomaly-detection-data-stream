import math

from src.Thresholds.BaseThreshold import BaseThreshold


class FixedThreshold(BaseThreshold):
    def __init__(self, value=0.5):
        # Inicializa o estado e as dependências necessárias para a execução da classe.
        self.initialValue = float(value)

        if not math.isfinite(self.initialValue):
            raise ValueError("The fixed threshold must be a finite number.")

        self.reset()

    def initialize(self, scores=None):
        # Inicializa a estratégia de threshold com os scores de calibração disponíveis.
        return None

    def getThreshold(self):
        # Retorna o valor atual do threshold utilizado para decisão.
        return float(self.currentValue)

    def update(self, score, index=None):
        # Atualiza o estado incremental com os valores da instância atual.
        return None

    def reset(self):
        # Reinicia o estado interno para permitir uma nova execução.
        self.currentValue = self.initialValue

    def isReady(self):
        # Informa se a estratégia de threshold está pronta para avaliar novas instâncias.
        return True

    def getState(self):
        # Retorna um resumo do estado atual da estratégia de threshold.
        return {
            "name": "fixed",
            "ready": True,
            "threshold": float(self.currentValue),
        }
