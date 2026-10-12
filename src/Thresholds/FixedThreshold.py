import math

from src.Thresholds.BaseThreshold import BaseThreshold


class FixedThreshold(BaseThreshold):
    def __init__(self, value=0.5):
        # Inicializa um threshold constante e valida que seu valor seja finito.
        self.initialValue = float(value)
        if not math.isfinite(self.initialValue):
            raise ValueError("The fixed threshold must be a finite number.")
        self.reset()

    def initialize(self, scores=None):
        # Mantém compatibilidade com a interface sem exigir calibração para o threshold fixo.
        return None

    def getThreshold(self):
        # Retorna o valor constante utilizado para decisão.
        return float(self.currentValue)

    def update(self, score, index=None):
        # Ignora atualizações porque o limiar permanece constante durante toda a stream.
        return None

    def reset(self):
        # Restaura o valor inicial antes de uma nova execução.
        self.currentValue = self.initialValue

    def isReady(self):
        # Informa que o threshold fixo está sempre disponível desde a primeira instância.
        return True

    def getState(self):
        # Retorna o estado mínimo necessário para identificação e auditoria do threshold.
        return {"name": "fixed", "ready": True, "threshold": float(self.currentValue)}
