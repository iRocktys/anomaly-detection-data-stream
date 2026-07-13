import re

from src.Anomaly.Thresholds.FixedThreshold import FixedThreshold
from src.Anomaly.Thresholds.Incremental.DspotThreshold import DspotThreshold
from src.Anomaly.Thresholds.Incremental.IncrementalMeanStdThreshold import IncrementalMeanStdThreshold
from src.Anomaly.Thresholds.Incremental.SpotThreshold import SpotThreshold
from src.Data.OnlineNormalizers import (
    IncrementalMinMaxNormalizer,
    IncrementalZScoreNormalizer,
    NoOnlineNormalizer,
    RollingMinMaxNormalizer,
    RollingZScoreNormalizer,
)
from src.Features.FeatureExtractors import NoFeatureExtractor
from src.Features.FeatureSmoothers import MovingAverageFeatureSmoother, NoFeatureSmoother
from src.Pipeline.DecisionComponents import ThresholdDecisionStrategy
from src.Scores.ScoreSmoothers import ExponentialScoreSmoother, MovingAverageScoreSmoother, NoScoreSmoother
from src.Training.TrainingStrategies import NoTrainingStrategy, TrainAllStrategy, TrainOracleNormalStrategy


class ComponentRegistry:
    builders = {
        "normalizer": {
            "none": NoOnlineNormalizer,
            "incrementalminmax": IncrementalMinMaxNormalizer,
            "incrementalzscore": IncrementalZScoreNormalizer,
            "rollingminmax": RollingMinMaxNormalizer,
            "rollingzscore": RollingZScoreNormalizer,
        },
        "featureExtractor": {
            "none": NoFeatureExtractor,
        },
        "featureSmoother": {
            "none": NoFeatureSmoother,
            "movingaverage": MovingAverageFeatureSmoother,
            "mean": MovingAverageFeatureSmoother,
        },
        "scoreSmoother": {
            "none": NoScoreSmoother,
            "movingaverage": MovingAverageScoreSmoother,
            "mean": MovingAverageScoreSmoother,
            "exponential": ExponentialScoreSmoother,
            "ewma": ExponentialScoreSmoother,
        },
        "threshold": {
            "fixed": FixedThreshold,
            "incrementalmeanstd": IncrementalMeanStdThreshold,
            "spot": SpotThreshold,
            "dspot": DspotThreshold,
        },
        "decision": {
            "threshold": ThresholdDecisionStrategy,
            "binary": ThresholdDecisionStrategy,
        },
        "training": {
            "all": TrainAllStrategy,
            "oraclenormal": TrainOracleNormalStrategy,
            "normalonly": TrainOracleNormalStrategy,
            "none": NoTrainingStrategy,
        },
    }

    @staticmethod
    def normalizeName(name):
        return re.sub(r"[^a-z0-9]", "", str(name or "").lower())

    @classmethod
    def create(cls, category, config):
        if category not in cls.builders:
            raise ValueError(f"Categoria de componente desconhecida: {category}.")
        normalizedName = cls.normalizeName(config.name)
        builder = cls.builders[category].get(normalizedName)
        if builder is None:
            available = ", ".join(sorted(cls.builders[category]))
            raise ValueError(
                f"Componente desconhecido em {category}: {config.name}. Disponíveis: {available}."
            )
        parameters = cls.normalizeParameters(category, normalizedName, config.parameters)
        try:
            return builder(**parameters)
        except TypeError as error:
            raise TypeError(
                f"Parâmetros inválidos para {category}/{config.name}: {parameters}."
            ) from error

    @classmethod
    def normalizeParameters(cls, category, name, parameters):
        resolved = dict(parameters or {})
        if "window_size" in resolved and "windowSize" not in resolved:
            resolved["windowSize"] = resolved.pop("window_size")
        return resolved

    @classmethod
    def describe(cls):
        return {
            category: sorted(builders.keys())
            for category, builders in cls.builders.items()
        }
