import json
from pathlib import Path

import numpy as np
import pandas as pd

import ProjectDefaults as defaults
from src.Plots.DspotStudyPlot import DspotStudyPlot
from src.Thresholds.Incremental.DspotThreshold import DspotConfig, DspotThreshold


class DspotStudy:
    def __init__(self, outputRoot=None):
        # Inicializa o estudo utilizando exatamente a mesma implementação DSPOT usada nos experimentos.
        self.outputRoot = Path(outputRoot or defaults.DEFAULT_DSPOT_OUTPUT_ROOT)

    def run(self, experimentName, scores, config=None):
        # Executa calibração e atualização incremental do DSPOT e persiste o diagnóstico completo da série.
        experiment_name = str(experimentName).strip()
        if not experiment_name:
            raise ValueError("experimentName must be provided explicitly.")
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(values) == 0 or np.any(~np.isfinite(values)):
            raise ValueError("The score series must contain only finite values.")
        dspot_config = config if isinstance(config, DspotConfig) else DspotConfig(**dict(config or {}))
        if len(values) <= dspot_config.warmupSize:
            raise ValueError("The series must contain values beyond the DSPOT calibration window.")
        detector = DspotThreshold(dspot_config)
        warmup = values[:dspot_config.warmupSize]
        detector.initialize(warmup)
        calibration_peak_count = int(detector.peakCount)
        rows = self.buildWarmupRows(warmup, detector, dspot_config)
        for index in range(dspot_config.warmupSize, len(values)):
            score = float(values[index])
            threshold_before = float(detector.getThreshold())
            state = detector.update(score, index=index)
            rows.append({
                "index": int(index),
                "score": score,
                "phase": "stream",
                "classification": state["classification"],
                "drift": float(state["drift"]),
                "residual": float(state["residual"]),
                "thresholdBefore": threshold_before,
                "thresholdAfter": float(state["threshold"]),
                "initialResidualThreshold": float(detector.initialResidualThreshold),
                "extremeResidualThreshold": float(detector.extremeResidualThreshold),
                "shape": float(state["shape"]),
                "scale": float(state["scale"]),
                "updatedTail": int(bool(state["updatedTail"])),
            })
        result_frame = pd.DataFrame(rows)
        output_directory = self.outputRoot / experiment_name
        plot_directory = output_directory / "plots"
        output_directory.mkdir(parents=True, exist_ok=True)
        plot_directory.mkdir(parents=True, exist_ok=True)
        result_path = output_directory / "dspot_study.csv"
        metadata_path = output_directory / "dspot_metadata.json"
        result_frame.to_csv(result_path, index=False)
        metadata = {
            "driftDepth": dspot_config.driftDepth,
            "calibrationSize": dspot_config.calibrationSize,
            "warmupSize": dspot_config.warmupSize,
            "initialQuantile": dspot_config.initialQuantile,
            "risk": dspot_config.risk,
            "refitEvery": dspot_config.refitEvery,
            "optimizationStarts": dspot_config.optimizationStarts,
            "tolerance": dspot_config.tolerance,
            "calibrationPeakCount": calibration_peak_count,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        plot_path = DspotStudyPlot().plot(result_frame, plot_directory / "dspot_threshold.png", dspot_config.warmupSize)
        return {
            "outputDirectory": str(output_directory),
            "resultPath": str(result_path),
            "metadataPath": str(metadata_path),
            "plotPath": str(plot_path),
            "resultFrame": result_frame,
            "config": dspot_config,
        }

    @staticmethod
    def buildWarmupRows(warmup, detector, config):
        # Representa a janela inicial no mesmo formato do período incremental para facilitar inspeção e plotagem.
        rows = []
        for index, score in enumerate(warmup):
            phase = "driftWarmup" if index < config.driftDepth else "tailCalibration"
            rows.append({
                "index": int(index),
                "score": float(score),
                "phase": phase,
                "classification": "calibration",
                "drift": np.nan,
                "residual": np.nan,
                "thresholdBefore": np.nan,
                "thresholdAfter": np.nan,
                "initialResidualThreshold": float(detector.initialResidualThreshold),
                "extremeResidualThreshold": float(detector.extremeResidualThreshold),
                "shape": float(detector.shape),
                "scale": float(detector.scale),
                "updatedTail": 0,
            })
        return rows
