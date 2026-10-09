from src.Anomaly.Thresholds.FixedThreshold import FixedThreshold
from src.Anomaly.Thresholds.Incremental.DspotThreshold import DspotConfig, DspotThreshold


class ThresholdRegistry:
    @staticmethod
    def create(name, parameters=None):
        # Cria a estratégia de threshold usando uma configuração textual genérica.
        key = str(name).strip().lower().replace("_", "").replace("-", "")
        values = dict(parameters or {})
        if key == "fixed":
            return FixedThreshold(**values)
        if key == "dspot":
            if "calibrationWindow" in values:
                calibration_window = int(values.pop("calibrationWindow"))
                drift_depth = int(values.get("driftDepth", DspotConfig().driftDepth))
                values["calibrationSize"] = calibration_window - drift_depth
            return DspotThreshold(DspotConfig(**values))
        raise ValueError("Threshold desconhecido: %s. Disponíveis: ['dspot', 'fixed']" % name)
