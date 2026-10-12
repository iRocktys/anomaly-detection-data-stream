import numpy as np


class BaseOnlineNormalizer:
    name = "base"

    def __init__(self, epsilon=1e-8):
        # Inicializa o normalizador com tolerância numérica usada por implementações derivadas.
        self.epsilon = float(epsilon)

    def transform(self, values):
        # Retorna uma cópia validada quando nenhuma normalização adicional é definida.
        return self.prepareValues(values)

    def update(self, values, observedMask=None):
        # Mantém uma interface uniforme para normalizadores sem estado incremental.
        return None

    def reset(self):
        # Mantém uma interface uniforme para normalizadores sem estado incremental.
        return None

    def prepareValues(self, values):
        # Valida que a entrada seja unidimensional e inteiramente finita após a imputação.
        prepared_values = np.asarray(values, dtype=np.float64)
        if prepared_values.ndim != 1:
            raise ValueError("O normalizador aceita apenas vetores unidimensionais.")
        if np.any(~np.isfinite(prepared_values)):
            raise ValueError("The normalizer received non-finite values; imputation must occur first.")
        return prepared_values.copy()

    def prepareObservedMask(self, observedMask, featureCount):
        # Normaliza a máscara que impede valores originalmente ausentes de atualizar estatísticas.
        feature_count = int(featureCount)
        if observedMask is None:
            return np.ones(feature_count, dtype=bool)
        prepared_mask = np.asarray(observedMask, dtype=bool)
        if prepared_mask.ndim != 1 or prepared_mask.size != feature_count:
            raise ValueError("The observed-value mask must contain one position per feature.")
        return prepared_mask


class IncrementalZScoreNormalizer(BaseOnlineNormalizer):
    name = "incrementalZScore"

    def __init__(self, epsilon=1e-8, clip=None):
        # Inicializa o Z-score incremental com clipping opcional dos valores normalizados.
        super().__init__(epsilon)
        self.clip = clip
        self.reset()

    def transform(self, values):
        # Normaliza a instância usando média e desvio padrão calculados somente no passado.
        prepared_values = self.prepareValues(values)
        self.ensureState(prepared_values.size)
        normalized_values = np.zeros_like(prepared_values)
        ready_mask = self.counts >= 2
        if np.any(ready_mask):
            variance = self.squareDistances[ready_mask] / (self.counts[ready_mask] - 1)
            deviation = np.sqrt(np.maximum(variance, 0.0))
            safe_deviation = np.where(deviation < self.epsilon, 1.0, deviation)
            normalized_values[ready_mask] = (prepared_values[ready_mask] - self.means[ready_mask]) / safe_deviation
        if self.clip is not None:
            normalized_values = np.clip(normalized_values, -float(self.clip), float(self.clip))
        return normalized_values

    def update(self, values, observedMask=None):
        # Atualiza média e variância online somente com features originalmente observadas.
        prepared_values = self.prepareValues(values)
        self.ensureState(prepared_values.size)
        prepared_mask = self.prepareObservedMask(observedMask, prepared_values.size)
        if not np.any(prepared_mask):
            return
        updated_counts = self.counts[prepared_mask] + 1
        differences = prepared_values[prepared_mask] - self.means[prepared_mask]
        updated_means = self.means[prepared_mask] + differences / updated_counts
        second_differences = prepared_values[prepared_mask] - updated_means
        self.squareDistances[prepared_mask] += differences * second_differences
        self.means[prepared_mask] = updated_means
        self.counts[prepared_mask] = updated_counts

    def ensureState(self, featureCount):
        # Inicializa os acumuladores da primeira instância e valida a dimensionalidade nas seguintes.
        feature_count = int(featureCount)
        if self.counts is None:
            self.counts = np.zeros(feature_count, dtype=np.int64)
            self.means = np.zeros(feature_count, dtype=np.float64)
            self.squareDistances = np.zeros(feature_count, dtype=np.float64)
            return
        if self.counts.size != feature_count:
            raise ValueError("The number of features received by the normalizer changed during execution.")

    def reset(self):
        # Zera todas as estatísticas acumuladas antes de uma nova execução.
        self.counts = None
        self.means = None
        self.squareDistances = None
