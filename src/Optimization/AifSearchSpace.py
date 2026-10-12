from dataclasses import dataclass


@dataclass(frozen=True)
class AifSearchSpaceConfig:
    windowSizes: tuple[int, ...] = (128, 256, 512)
    treeCounts: tuple[int, ...] = (50, 100, 150, 200)
    heightMinimum: int = 8
    heightMaximum: int = 15
    memberMinimum: int = 5
    memberMaximum: int = 20
    weightMinimum: float = 0.15
    weightMaximum: float = 0.75


class AifSearchSpace:
    def __init__(self, config=None):
        # Inicializa o espaço de busca dos hiperparâmetros do Adaptive Isolation Forest.
        self.config = config or AifSearchSpaceConfig()

    def suggest(self, trial, fixedParameters=None, modelWarmupSize=500):
        # Sugere somente parâmetros não fixados do AIF e respeita o tamanho disponível para aquecimento.
        fixed = dict(fixedParameters or {})
        valid_windows = [value for value in self.config.windowSizes if value <= int(modelWarmupSize)]
        if not valid_windows:
            raise ValueError("No AIF window_size fits inside the configured model warm-up.")
        window_size = int(fixed["window_size"]) if "window_size" in fixed else int(trial.suggest_categorical("aif_window_size", valid_windows))
        n_trees = int(fixed["n_trees"]) if "n_trees" in fixed else int(trial.suggest_categorical("aif_n_trees", list(self.config.treeCounts)))
        height = int(fixed["height"]) if "height" in fixed else int(trial.suggest_int("aif_height", self.config.heightMinimum, self.config.heightMaximum))
        m_trees = int(fixed["m_trees"]) if "m_trees" in fixed else int(trial.suggest_int("aif_m_trees", self.config.memberMinimum, self.config.memberMaximum))
        weights = float(fixed["weights"]) if "weights" in fixed else float(trial.suggest_float("aif_weights", self.config.weightMinimum, self.config.weightMaximum))
        if window_size > int(modelWarmupSize):
            raise ValueError("AIF window_size cannot exceed the model warm-up.")
        return {
            "window_size": window_size,
            "n_trees": n_trees,
            "height": height,
            "m_trees": m_trees,
            "weights": weights,
        }
