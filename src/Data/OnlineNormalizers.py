import re
from collections import deque

import numpy as np
from capymoa.instance import LabeledInstance


class BaseOnlineNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def transform(self, x):
        return np.asarray(x, dtype=np.float64)

    def update(self, x):
        return None

    def transform_instance(self, instance):
        x_normalized = self.transform(instance.x)

        return LabeledInstance.from_array(
            schema=instance.schema,
            x=x_normalized,
            y_index=int(instance.y_index),
        )

    def _clean_x(self, x):
        x = np.asarray(x, dtype=np.float64)

        return np.nan_to_num(
            x,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=np.finfo(np.float32).min,
        )


class NoOnlineNormalizer(BaseOnlineNormalizer):
    def transform(self, x):
        return np.asarray(x, dtype=np.float64)

    def update(self, x):
        return None

    def transform_instance(self, instance):
        return instance


class IncrementalMinMaxNormalizer(BaseOnlineNormalizer):
    def __init__(self, epsilon=1e-8, clip=True):
        super().__init__(epsilon=epsilon)
        self.clip = clip
        self.min_values = None
        self.max_values = None
        self.n_samples = 0

    def transform(self, x):
        x = self._clean_x(x)

        if self.n_samples == 0 or self.min_values is None or self.max_values is None:
            return np.zeros_like(x, dtype=np.float64)

        denominator = self.max_values - self.min_values
        safe_denominator = np.where(
            np.abs(denominator) < self.epsilon,
            1.0,
            denominator,
        )

        x_normalized = (x - self.min_values) / safe_denominator

        if self.clip:
            x_normalized = np.clip(x_normalized, 0.0, 1.0)

        return x_normalized.astype(np.float64)

    def update(self, x):
        x = self._clean_x(x)

        if self.n_samples == 0:
            self.min_values = x.copy()
            self.max_values = x.copy()
        else:
            self.min_values = np.minimum(self.min_values, x)
            self.max_values = np.maximum(self.max_values, x)

        self.n_samples += 1


class IncrementalZScoreNormalizer(BaseOnlineNormalizer):
    def __init__(self, epsilon=1e-8, clip=None):
        super().__init__(epsilon=epsilon)
        self.clip = clip
        self.n_samples = 0
        self.mean = None
        self.m2 = None

    def transform(self, x):
        x = self._clean_x(x)

        if self.n_samples < 2 or self.mean is None or self.m2 is None:
            return np.zeros_like(x, dtype=np.float64)

        variance = self.m2 / max(self.n_samples - 1, 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        safe_std = np.where(std < self.epsilon, 1.0, std)

        x_normalized = (x - self.mean) / safe_std

        if self.clip is not None:
            x_normalized = np.clip(
                x_normalized,
                -float(self.clip),
                float(self.clip),
            )

        return x_normalized.astype(np.float64)

    def update(self, x):
        x = self._clean_x(x)

        if self.n_samples == 0:
            self.n_samples = 1
            self.mean = x.copy()
            self.m2 = np.zeros_like(x, dtype=np.float64)
            return

        self.n_samples += 1

        delta = x - self.mean
        self.mean = self.mean + (delta / self.n_samples)
        delta_after_update = x - self.mean

        self.m2 = self.m2 + (delta * delta_after_update)


class RollingMinMaxNormalizer(BaseOnlineNormalizer):
    def __init__(self, window_size=200, epsilon=1e-8, clip=True):
        super().__init__(epsilon=epsilon)
        self.window_size = max(2, int(window_size))
        self.clip = clip
        self.window = deque(maxlen=self.window_size)

    def transform(self, x):
        x = self._clean_x(x)

        if len(self.window) == 0:
            return np.zeros_like(x, dtype=np.float64)

        window_array = np.asarray(self.window, dtype=np.float64)

        min_values = np.min(window_array, axis=0)
        max_values = np.max(window_array, axis=0)

        denominator = max_values - min_values
        safe_denominator = np.where(
            np.abs(denominator) < self.epsilon,
            1.0,
            denominator,
        )

        x_normalized = (x - min_values) / safe_denominator

        if self.clip:
            x_normalized = np.clip(x_normalized, 0.0, 1.0)

        return x_normalized.astype(np.float64)

    def update(self, x):
        x = self._clean_x(x)
        self.window.append(x.copy())

class RollingZScoreNormalizer(BaseOnlineNormalizer):
    def __init__(self, window_size=200, epsilon=1e-8, clip=None):
        super().__init__(epsilon=epsilon)
        self.window_size = max(2, int(window_size))
        self.clip = clip
        self.window = deque(maxlen=self.window_size)

    def transform(self, x):
        x = self._clean_x(x)

        if len(self.window) < 2:
            return np.zeros_like(x, dtype=np.float64)

        window_array = np.asarray(self.window, dtype=np.float64)

        mean = np.mean(window_array, axis=0)
        std = np.std(window_array, axis=0, ddof=1)
        safe_std = np.where(std < self.epsilon, 1.0, std)

        x_normalized = (x - mean) / safe_std

        if self.clip is not None:
            x_normalized = np.clip(
                x_normalized,
                -float(self.clip),
                float(self.clip),
            )

        return x_normalized.astype(np.float64)

    def update(self, x):
        x = self._clean_x(x)
        self.window.append(x.copy())


class OnlineNormalizers:
    def create(strategy="none", **kwargs):
        strategy = str(strategy or "none").strip().lower()

        if strategy == "none":
            return NoOnlineNormalizer(**kwargs)

        if strategy in ["incremental_minmax", "incremental_minmanx"]:
            return IncrementalMinMaxNormalizer(**kwargs)

        if strategy in ["incremental_z_score", "incremental_zscore"]:
            return IncrementalZScoreNormalizer(**kwargs)

        if strategy.startswith("rolling_minmax"):
            window_size = OnlineNormalizers._extract_window_size(
                strategy=strategy,
                default_window=200,
            )

            return RollingMinMaxNormalizer(
                window_size=window_size,
                **kwargs,
            )

        if strategy.startswith("rolling_z_score") or strategy.startswith("rolling_zscore"):
            window_size = OnlineNormalizers._extract_window_size(
                strategy=strategy,
                default_window=200,
            )

            return RollingZScoreNormalizer(
                window_size=window_size,
                **kwargs,
            )

        raise ValueError(
            "normalization_strategy deve ser uma destas: "
            "none, incremental_minmax, incremental_z_score, "
            "rolling_minmax_w200, rolling_z_score_w200."
        )

    def _extract_window_size(strategy, default_window=200):
        match = re.search(r"_w(\d+)$", strategy)

        if match is None:
            return int(default_window)

        return int(match.group(1))
# API em camelCase utilizada pelo pipeline modular.
def createNormalizer(strategy="none", **kwargs):
    if "windowSize" in kwargs:
        kwargs["window_size"] = kwargs.pop("windowSize")
    strategyMap = {
        "none": "none",
        "incrementalminmax": "incremental_minmax",
        "incrementalzscore": "incremental_z_score",
        "rollingminmax": "rolling_minmax",
        "rollingzscore": "rolling_z_score",
    }
    normalizedName = str(strategy or "none").replace("_", "").replace("-", "").strip().lower()
    mappedStrategy = strategyMap.get(normalizedName, strategy)
    return OnlineNormalizers.create(strategy=mappedStrategy, **kwargs)


OnlineNormalizers.createNormalizer = staticmethod(createNormalizer)
