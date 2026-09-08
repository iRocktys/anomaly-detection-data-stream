"""Espaço de busca do AIF. Os limites abaixo são escolhas experimentais.

API conferida em https://capymoa.org/api/modules/capymoa.anomaly.AdaptiveIsolationForest.html
m_trees é a quantidade de árvores candidatas, não de árvores removidas.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class AifSearchSpaceConfig:
    windowSizes: tuple[int, ...] = (128, 256, 512, 1024)
    nTreesMinimum: int = 50
    nTreesMaximum: int = 200
    nTreesStep: int = 25
    heightMinimum: int = 6
    heightMaximum: int = 15
    mTreesMinimum: int = 5
    mTreesMaximum: int = 30
    mTreesStep: int = 5
    weightsMinimum: float = 0.0
    weightsMaximum: float = 1.0

    def __post_init__(self):
        if not self.windowSizes or any(
            not isinstance(v, Integral) or isinstance(v, bool) or v < 2
            for v in self.windowSizes
        ):
            raise ValueError("windowSizes deve conter inteiros maiores ou iguais a 2.")
        object.__setattr__(self, "windowSizes", tuple(sorted(set(self.windowSizes))))
        for prefix in ("nTrees", "height", "mTrees"):
            low, high = getattr(self, prefix + "Minimum"), getattr(self, prefix + "Maximum")
            step = getattr(self, prefix + "Step", 1)
            if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (low, high, step)):
                raise ValueError(f"O intervalo de {prefix} deve usar inteiros.")
            if low < 1 or high < low or step < 1 or (high - low) % step:
                raise ValueError(f"Intervalo/passo inválido para {prefix}.")
        if not 0 <= self.weightsMinimum <= self.weightsMaximum <= 1:
            raise ValueError("O intervalo de weights deve estar entre 0 e 1.")


class AifSearchSpace:
    parameterNames = ("window_size", "n_trees", "height", "m_trees", "weights")

    def __init__(self, config=None):
        self.config = config or AifSearchSpaceConfig()

    def suggest(self, trial, fixedParameters=None, initialWarmupSize=2000):
        fixed = dict(fixedParameters or {})
        cfg = self.config
        # A janela do AIF precisa terminar dentro do aquecimento fixo.
        windows = [v for v in cfg.windowSizes if v <= initialWarmupSize]
        if "window_size" not in fixed and not windows:
            raise ValueError("Nenhuma janela do AIF cabe no initialWarmupSize.")
        suggestions = {
            "window_size": lambda: trial.suggest_categorical("aif_window_size", windows),
            "n_trees": lambda: trial.suggest_int("aif_n_trees", cfg.nTreesMinimum, cfg.nTreesMaximum, step=cfg.nTreesStep),
            "height": lambda: trial.suggest_int("aif_height", cfg.heightMinimum, cfg.heightMaximum),
            "m_trees": lambda: trial.suggest_int("aif_m_trees", cfg.mTreesMinimum, cfg.mTreesMaximum, step=cfg.mTreesStep),
            "weights": lambda: trial.suggest_float("aif_weights", cfg.weightsMinimum, cfg.weightsMaximum),
        }
        parameters = {name: fixed[name] if name in fixed else suggest() for name, suggest in suggestions.items()}
        self.validateParameters(parameters, initialWarmupSize)
        return parameters

    @staticmethod
    def validateParameters(parameters, initialWarmupSize):
        for name in ("window_size", "n_trees", "m_trees"):
            value = parameters[name]
            minimum = 2 if name == "window_size" else 1
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"AIF: {name} deve ser inteiro >= {minimum}.")
        height = parameters.get("height")
        if height is not None and (not isinstance(height, Integral) or isinstance(height, bool) or height < 1):
            raise ValueError("AIF: height deve ser None ou inteiro positivo.")
        weights = parameters["weights"]
        if not isinstance(weights, Real) or not math.isfinite(weights) or not 0 <= weights <= 1:
            raise ValueError("AIF: weights deve estar entre 0 e 1.")
        if parameters["window_size"] > initialWarmupSize:
            raise ValueError("A janela do AIF não pode superar o warm-up da avaliação.")
