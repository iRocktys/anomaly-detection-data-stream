import numpy as np


class BaseOnlineImputer:
    name = "base"

    def transform(self, values):
        # Define a transformação causal aplicada antes da atualização do estado do imputador.
        raise NotImplementedError

    def update(self, values):
        # Define a atualização incremental do estado usando a instância já processada.
        raise NotImplementedError

    def reset(self):
        # Define a limpeza completa do estado interno do imputador.
        raise NotImplementedError

    def prepareValues(self, values):
        # Converte a entrada para um vetor unidimensional de ponto flutuante.
        prepared_values = np.asarray(values, dtype=np.float64)
        if prepared_values.ndim != 1:
            raise ValueError("O imputador aceita apenas vetores unidimensionais.")
        return prepared_values.copy()


class IncrementalMeanImputer(BaseOnlineImputer):
    name = "incrementalMean"

    def __init__(self):
        # Inicializa o imputador incremental sem estatísticas previamente observadas.
        self.reset()

    def transform(self, values):
        # Substitui ausências pela média acumulada somente de observações anteriores.
        imputed_values = self.prepareValues(values)
        self.ensureState(imputed_values.size)
        missing_mask = ~np.isfinite(imputed_values)
        imputed_values[missing_mask] = self.means[missing_mask]
        return imputed_values

    def update(self, values):
        # Atualiza a média incremental de cada feature somente nas posições observadas.
        observed_values = self.prepareValues(values)
        self.ensureState(observed_values.size)
        observed_mask = np.isfinite(observed_values)
        if not np.any(observed_mask):
            return
        updated_counts = self.counts[observed_mask] + 1
        differences = observed_values[observed_mask] - self.means[observed_mask]
        self.means[observed_mask] += differences / updated_counts
        self.counts[observed_mask] = updated_counts

    def ensureState(self, featureCount):
        # Inicializa as estatísticas na primeira instância e valida estabilidade dimensional depois disso.
        feature_count = int(featureCount)
        if self.counts is None:
            self.counts = np.zeros(feature_count, dtype=np.int64)
            self.means = np.zeros(feature_count, dtype=np.float64)
            return
        if self.counts.size != feature_count:
            raise ValueError("The number of features received by the imputer changed during execution.")

    def reset(self):
        # Remove as médias e contagens acumuladas para iniciar uma nova stream independente.
        self.counts = None
        self.means = None
