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

    def run(
        self,
        experimentName,
        scores,
        config=None,
        scoreColumn=None,
        attackNameColumn="labelName",
        attackFlagColumn="isAttack",
    ):
        # Executa o DSPOT preservando, quando disponíveis, os rótulos binários e nomes dos ataques usados no gráfico.
        experiment_name = str(experimentName).strip()
        if not experiment_name:
            raise ValueError("experimentName must be provided explicitly.")

        values, attack_frame = self.prepareScores(
            scores,
            scoreColumn,
            attackNameColumn,
            attackFlagColumn,
        )

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
        result_frame = self.attachAttackMetadata(
            result_frame,
            attack_frame,
            attackNameColumn,
            attackFlagColumn,
        )

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

        plot_path = DspotStudyPlot().plot(
            result_frame,
            plot_directory / "dspot_threshold.png",
            dspot_config.warmupSize,
            attackNameColumn=attackNameColumn,
            attackFlagColumn=attackFlagColumn,
        )

        return {
            "outputDirectory": str(output_directory),
            "resultPath": str(result_path),
            "metadataPath": str(metadata_path),
            "plotPath": str(plot_path),
            "resultFrame": result_frame,
            "config": dspot_config,
        }

    @staticmethod
    def prepareScores(scores, scoreColumn, attackNameColumn, attackFlagColumn):
        # Prepara a série numérica e mantém os metadados de ataque quando os scores forem fornecidos em DataFrame.
        if isinstance(scores, pd.DataFrame):
            frame = scores.reset_index(drop=True).copy()
            resolved_column = scoreColumn

            if resolved_column is None:
                if "score" in frame.columns:
                    resolved_column = "score"
                elif "rawScore" in frame.columns:
                    resolved_column = "rawScore"
                else:
                    raise ValueError("scoreColumn must be provided when the DataFrame has no 'score' or 'rawScore' column.")

            if resolved_column not in frame.columns:
                raise KeyError(f"Missing score column: {resolved_column}.")

            numeric_scores = pd.to_numeric(frame[resolved_column], errors="coerce")
            valid_mask = numeric_scores.notna() & np.isfinite(numeric_scores.to_numpy(dtype=np.float64))
            frame = frame.loc[valid_mask].reset_index(drop=True)
            values = numeric_scores.loc[valid_mask].to_numpy(dtype=np.float64)

            attack_columns = [column for column in (attackFlagColumn, attackNameColumn) if column in frame.columns]
            attack_frame = frame[attack_columns].copy() if attack_columns else None
        else:
            values = np.asarray(scores, dtype=np.float64).reshape(-1)
            attack_frame = None

        if len(values) == 0 or np.any(~np.isfinite(values)):
            raise ValueError("The score series must contain only finite values.")

        return values, attack_frame

    @staticmethod
    def attachAttackMetadata(resultFrame, attackFrame, attackNameColumn, attackFlagColumn):
        # Anexa aos resultados DSPOT as regiões reais de ataque sem alterar a lógica de detecção incremental.
        if attackFrame is None:
            return resultFrame

        result = resultFrame.copy()
        aligned = attackFrame.iloc[:len(result)].reset_index(drop=True)

        if attackFlagColumn in aligned.columns:
            result[attackFlagColumn] = aligned[attackFlagColumn].astype(bool).to_numpy()

        if attackNameColumn in aligned.columns:
            result[attackNameColumn] = aligned[attackNameColumn].fillna("Attack").astype(str).to_numpy()

        return result

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
