from src.Anomaly.Thresholds.FixedThreshold import FixedThreshold
from src.Anomaly.Thresholds.Incremental.IncrementalMeanStdThreshold import IncrementalMeanStdThreshold


class ThresholdFactory:
    @staticmethod
    def createThreshold(config):
        thresholdName = str(config.name).strip().lower()
        thresholdParameters = dict(config.parameters)

        if thresholdName == "fixed":
            return FixedThreshold(**thresholdParameters)
        if thresholdName in {"incrementalmeanstd", "incrementalMeanStd".lower()}:
            return IncrementalMeanStdThreshold(**thresholdParameters)
        if thresholdName == "spot":
            raise NotImplementedError("SPOT possui módulo reservado, mas ainda não está habilitado na fábrica.")
        if thresholdName == "dspot":
            raise NotImplementedError("DSPOT possui módulo reservado, mas ainda não está habilitado na fábrica.")

        raise ValueError(f"Threshold desconhecido: {config.name}")
