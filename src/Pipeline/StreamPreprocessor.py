import numpy as np


class StreamPreprocessor:
    def __init__(self, imputer, normalizer):
        # Mantém imputação e normalização online em uma única etapa reutilizável pelos pipelines.
        self.imputer = imputer
        self.normalizer = normalizer

    def transform(self, raw_values):
        # Transforma uma instância usando apenas o estado acumulado nas instâncias anteriores.
        values = np.asarray(raw_values, dtype=np.float64)
        observed_mask = np.isfinite(values)
        imputed_values = self.imputer.transform(values)
        if np.any(~np.isfinite(imputed_values)):
            raise RuntimeError("The imputer produced non-finite values.")
        normalized_values = self.normalizer.transform(imputed_values)
        if np.any(~np.isfinite(normalized_values)):
            raise RuntimeError("The normalizer produced non-finite values.")
        return normalized_values, imputed_values, observed_mask

    def update(self, raw_values, imputed_values, observed_mask):
        # Atualiza os estados online somente depois da predição ou score da instância atual.
        self.imputer.update(np.asarray(raw_values, dtype=np.float64))
        self.normalizer.update(np.asarray(imputed_values, dtype=np.float64), observedMask=observed_mask)

    def reset(self):
        # Reinicia os componentes de pré-processamento antes de uma nova execução.
        self.imputer.reset()
        self.normalizer.reset()

    def getImputerName(self):
        # Retorna um nome persistível para o imputador em uso.
        if hasattr(self.imputer, "name"):
            return str(self.imputer.name)
        return self.imputer.__class__.__name__

    def getNormalizerName(self):
        # Retorna um nome persistível para o normalizador em uso.
        if hasattr(self.normalizer, "name"):
            return str(self.normalizer.name)
        return self.normalizer.__class__.__name__
