from dataclasses import dataclass
import math
from numbers import Integral, Real

import ProjectDefaults as defaults


@dataclass(frozen=True)
class AifSearchSpaceConfig:
    windowSizeMinimum: int = 64
    windowSizeMaximum: int = defaults.DEFAULT_AIF_WARMUP_SIZE
    windowSizeStep: int = 4
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
        # Valida um espaço AIF que sempre cabe nos 500 exemplos exclusivos de aquecimento do modelo.
        integerNames = (
            "windowSizeMinimum",
            "windowSizeMaximum",
            "windowSizeStep",
            "nTreesMinimum",
            "nTreesMaximum",
            "nTreesStep",
            "heightMinimum",
            "heightMaximum",
            "mTreesMinimum",
            "mTreesMaximum",
            "mTreesStep",
        )

        if any(
            not isinstance(
                getattr(
                    self,
                    name,
                ),
                Integral,
            )
            or isinstance(
                getattr(
                    self,
                    name,
                ),
                bool,
            )
            for name in integerNames
        ):
            raise ValueError(
                "Os limites e passos do AIF devem ser inteiros."
            )

        for prefix in (
            "windowSize",
            "nTrees",
            "height",
            "mTrees",
        ):
            low = getattr(
                self,
                prefix
                + "Minimum",
            )

            high = getattr(
                self,
                prefix
                + "Maximum",
            )

            step = getattr(
                self,
                prefix
                + "Step",
                1,
            )

            if (
                low < 1
                or high < low
                or step < 1
            ):
                raise ValueError(
                    f"Intervalo inválido para {prefix}."
                )

            if (
                prefix != "height"
                and (
                    high
                    - low
                )
                % step
            ):
                raise ValueError(
                    f"O intervalo de {prefix} deve ser divisível pelo passo."
                )

        if (
            self.windowSizeMaximum
            > defaults.DEFAULT_AIF_WARMUP_SIZE
        ):
            raise ValueError(
                "windowSizeMaximum não pode superar o warm-up padrão do AIF."
            )

        if not (
            0.0
            <= self.weightsMinimum
            <= self.weightsMaximum
            <= 1.0
        ):
            raise ValueError(
                "O intervalo de weights deve estar entre 0 e 1."
            )


class AifSearchSpace:
    parameterNames = (
        "window_size",
        "n_trees",
        "height",
        "m_trees",
        "weights",
    )

    def __init__(self, config=None):
        # Inicializa o espaço de busca do AIF usando limites fixos e compatíveis com o protocolo.
        self.config = (
            config
            or AifSearchSpaceConfig()
        )

    def suggest(self, trial, fixedParameters=None, modelWarmupSize=None):
        # Sugere todos os hiperparâmetros do AIF sem permitir janela maior que seu warm-up.
        fixed = dict(
            fixedParameters
            or {}
        )

        cfg = self.config

        warmup = int(
            modelWarmupSize
            if modelWarmupSize is not None
            else defaults.DEFAULT_AIF_WARMUP_SIZE
        )

        maximum_window = min(
            cfg.windowSizeMaximum,
            warmup,
        )

        if (
            maximum_window
            < cfg.windowSizeMinimum
            and "window_size"
            not in fixed
        ):
            raise ValueError(
                "O warm-up do AIF é menor que a menor janela permitida."
            )

        parameters = {
            "window_size": (
                fixed[
                    "window_size"
                ]
                if "window_size" in fixed
                else trial.suggest_int(
                    "aif_window_size",
                    cfg.windowSizeMinimum,
                    maximum_window,
                    step=cfg.windowSizeStep,
                )
            ),
            "n_trees": (
                fixed[
                    "n_trees"
                ]
                if "n_trees" in fixed
                else trial.suggest_int(
                    "aif_n_trees",
                    cfg.nTreesMinimum,
                    cfg.nTreesMaximum,
                    step=cfg.nTreesStep,
                )
            ),
            "height": (
                fixed[
                    "height"
                ]
                if "height" in fixed
                else trial.suggest_int(
                    "aif_height",
                    cfg.heightMinimum,
                    cfg.heightMaximum,
                )
            ),
            "m_trees": (
                fixed[
                    "m_trees"
                ]
                if "m_trees" in fixed
                else trial.suggest_int(
                    "aif_m_trees",
                    cfg.mTreesMinimum,
                    cfg.mTreesMaximum,
                    step=cfg.mTreesStep,
                )
            ),
            "weights": (
                fixed[
                    "weights"
                ]
                if "weights" in fixed
                else trial.suggest_float(
                    "aif_weights",
                    cfg.weightsMinimum,
                    cfg.weightsMaximum,
                )
            ),
        }

        self.validateParameters(
            parameters,
            warmup,
        )

        return parameters

    @staticmethod
    def validateParameters(parameters, modelWarmupSize):
        # Valida os parâmetros AIF antes de criar o modelo e evita trials estruturalmente inválidos.
        for name in (
            "window_size",
            "n_trees",
            "m_trees",
        ):
            value = parameters[
                name
            ]

            minimum = (
                2
                if name
                == "window_size"
                else 1
            )

            if (
                not isinstance(
                    value,
                    Integral,
                )
                or isinstance(
                    value,
                    bool,
                )
                or value
                < minimum
            ):
                raise ValueError(
                    f"AIF: {name} deve ser inteiro >= {minimum}."
                )

        height = parameters.get(
            "height"
        )

        if (
            height is not None
            and (
                not isinstance(
                    height,
                    Integral,
                )
                or isinstance(
                    height,
                    bool,
                )
                or height
                < 1
            )
        ):
            raise ValueError(
                "AIF: height deve ser None ou inteiro positivo."
            )

        weights = parameters[
            "weights"
        ]

        if (
            not isinstance(
                weights,
                Real,
            )
            or not math.isfinite(
                weights
            )
            or not (
                0.0
                <= weights
                <= 1.0
            )
        ):
            raise ValueError(
                "AIF: weights deve estar entre 0 e 1."
            )

        if (
            parameters[
                "window_size"
            ]
            > int(
                modelWarmupSize
            )
        ):
            raise ValueError(
                "A janela do AIF não pode superar seu warm-up."
            )
