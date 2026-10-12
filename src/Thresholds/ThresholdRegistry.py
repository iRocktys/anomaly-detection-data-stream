from src.Thresholds.FixedThreshold import FixedThreshold
from src.Thresholds.Incremental.DspotThreshold import DspotConfig, DspotThreshold


class ThresholdRegistry:
    @staticmethod
    def create(name, parameters=None):
        # Cria a estratégia de threshold configurada para o pipeline de detecção de anomalias.
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
        raise ValueError(f"Unknown threshold: {name}. Available: ['dspot', 'fixed']")
