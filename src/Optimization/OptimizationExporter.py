import json
import math
from pathlib import Path

import pandas as pd

from src.Metrics.Metrics import Metrics


class OptimizationExporter:
    globalMetricNames = (
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "tp",
        "tn",
        "fp",
        "fn",
        "evaluatedInstances",
        "benignInstances",
        "attackInstances",
        "attackRatioPercent",
    )

    def __init__(self, outputDirectory, topK=10):
        # Inicializa a exportação mantendo somente trials concluídos nos CSVs finais.
        self.outputDirectory = Path(
            outputDirectory
        )

        self.topK = max(
            1,
            int(
                topK
            ),
        )

    def recordIdentity(self, trial, dataset, modelCode):
        # Registra a identidade explícita do dataset e do estudo no trial.
        trial.set_user_attr(
            "datasetKey",
            dataset.name,
        )

        trial.set_user_attr(
            "dataset",
            dataset.name,
        )

        trial.set_user_attr(
            "modelCode",
            modelCode,
        )

        trial.set_user_attr(
            "studyName",
            trial.study.study_name,
        )

    def recordConfiguration(self, trial, configuration, effectiveParameters):
        # Registra somente a configuração efetivamente testada para auditoria dos trials válidos.
        total_warmup = int(
            effectiveParameters[
                "execution"
            ][
                "initialWarmupSize"
            ]
        )

        payload = {
            "imputerName": configuration.imputerName,
            "thresholdScoreSource": configuration.thresholdScoreSource,
            "scoreWindowSizes": configuration.scoreWindowSizes,
            "modelWarmupSize": (
                total_warmup
                - configuration.calibrationWindow
            ),
            "thresholdCalibrationWindow": configuration.calibrationWindow,
            "driftDepth": configuration.driftDepth,
            "calibrationSize": configuration.calibrationSize,
            "initialQuantile": configuration.initialQuantile,
            "risk": configuration.risk,
            "refitEvery": configuration.refitEvery,
            "optimizationStarts": configuration.optimizationStarts,
            "tolerance": configuration.tolerance,
            "modelParameters": dict(
                configuration.modelParameters
            ),
            "normalizer": effectiveParameters[
                "preprocessing"
            ][
                "normalizer"
            ],
            "initialWarmupSize": total_warmup,
            "metricsWindowSize": effectiveParameters[
                "execution"
            ][
                "metricsWindowSize"
            ],
        }

        trial.set_user_attr(
            "effectiveConfiguration",
            payload,
        )

        trial.set_user_attr(
            "modelParameters",
            dict(
                configuration.modelParameters
            ),
        )

        for key, value in payload.items():
            if key not in {
                "modelParameters",
                "scoreWindowSizes",
            }:
                trial.set_user_attr(
                    key,
                    value,
                )

    def recordResult(self, trial, result):
        # Registra as métricas globais e as contagens por janela do trial concluído.
        metrics = result.streamMetrics

        if not math.isfinite(
            float(
                metrics[
                    "f1"
                ]
            )
        ):
            raise ValueError(
                "O trial produziu um F1 não finito."
            )

        countNames = {
            "tp",
            "tn",
            "fp",
            "fn",
            "evaluatedInstances",
            "benignInstances",
            "attackInstances",
        }

        for name in self.globalMetricNames:
            value = (
                int(
                    metrics[
                        name
                    ]
                )
                if name in countNames
                else float(
                    metrics[
                        name
                    ]
                )
            )

            trial.set_user_attr(
                name,
                value,
            )

        trial.set_user_attr(
            "windowCounts",
            [
                [
                    int(
                        window[
                            name
                        ]
                    )
                    for name in (
                        "windowIndex",
                        "windowStart",
                        "windowEnd",
                        "tp",
                        "tn",
                        "fp",
                        "fn",
                    )
                ]
                for window in result.windowMetrics
            ],
        )

    def export(self, studies):
        # Exporta somente trials COMPLETE e o F1 janelado do Top-K calculado exclusivamente pelo F1 global.
        self.outputDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        trialRows = []
        windowRows = []

        for datasetKey, study in studies.items():
            completeTrials = [
                trial
                for trial in study.trials
                if self._isComplete(
                    trial
                )
            ]

            ranking = self._rankTrials(
                completeTrials
            )

            for trial in sorted(
                completeTrials,
                key=lambda item: item.number,
            ):
                trialRows.append(
                    self._trialRow(
                        datasetKey,
                        study.study_name,
                        trial,
                        ranking.get(
                            trial.number
                        ),
                    )
                )

            for trial in completeTrials:
                rank = ranking.get(
                    trial.number
                )

                if (
                    rank is not None
                    and rank
                    <= self.topK
                ):
                    windowRows.extend(
                        self._windowRows(
                            datasetKey,
                            trial,
                            rank,
                        )
                    )

        trialsPath = (
            self.outputDirectory
            / "trials.csv"
        )

        windowsPath = (
            self.outputDirectory
            / "top10_windows.csv"
        )

        pd.DataFrame(
            trialRows
        ).to_csv(
            trialsPath,
            index=False,
        )

        pd.DataFrame(
            windowRows
        ).to_csv(
            windowsPath,
            index=False,
        )

        return {
            "trialsPath": str(
                trialsPath
            ),
            "top10WindowsPath": str(
                windowsPath
            ),
        }

    def _rankTrials(self, trials):
        # Ordena configurações únicas usando apenas o F1 global e o número do trial como desempate.
        ordered = sorted(
            trials,
            key=lambda trial: (
                -float(
                    trial.value
                ),
                int(
                    trial.number
                ),
            ),
        )

        ranking = {}
        signatures = set()

        for trial in ordered:
            signature = json.dumps(
                trial.user_attrs.get(
                    "effectiveConfiguration",
                    trial.params,
                ),
                sort_keys=True,
                default=str,
            )

            if signature in signatures:
                continue

            signatures.add(
                signature
            )

            ranking[
                trial.number
            ] = (
                len(
                    ranking
                )
                + 1
            )

        return ranking

    def _isComplete(self, trial):
        # Verifica se o trial terminou normalmente com F1 finito.
        return (
            getattr(
                trial.state,
                "name",
                str(
                    trial.state
                ),
            )
            == "COMPLETE"
            and trial.value is not None
            and math.isfinite(
                float(
                    trial.value
                )
            )
        )

    def _trialRow(self, datasetKey, studyName, trial, rank):
        # Converte um trial concluído em uma linha tabular para o CSV de resultados.
        attrs = trial.user_attrs

        row = {
            "datasetKey": datasetKey,
            "studyName": studyName,
            "trialNumber": int(
                trial.number
            ),
            "objectiveF1": float(
                trial.value
            ),
            "rank": rank,
            "isTop10": bool(
                rank is not None
                and rank
                <= self.topK
            ),
            "modelCode": attrs.get(
                "modelCode"
            ),
        }

        for name in self.globalMetricNames:
            row[
                name
            ] = attrs.get(
                name
            )

        for name, value in sorted(
            trial.params.items()
        ):
            row[
                f"param_{name}"
            ] = value

        for name, value in sorted(
            attrs.get(
                "modelParameters",
                {},
            ).items()
        ):
            row[
                f"model_{name}"
            ] = value

        return row

    def _windowRows(self, datasetKey, trial, rank):
        # Converte as contagens janeladas de um trial Top-K em métricas por janela.
        rows = []

        cumulative = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

        for values in trial.user_attrs.get(
            "windowCounts",
            [],
        ):
            (
                windowIndex,
                windowStart,
                windowEnd,
                tp,
                tn,
                fp,
                fn,
            ) = values

            for name, value in zip(
                (
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                ),
                (
                    tp,
                    tn,
                    fp,
                    fn,
                ),
            ):
                cumulative[
                    name
                ] += int(
                    value
                )

            metrics = Metrics.fromCounts(
                tp,
                tn,
                fp,
                fn,
            )

            cumulativeMetrics = Metrics.fromCounts(
                **cumulative
            )

            rows.append(
                {
                    "datasetKey": datasetKey,
                    "rank": int(
                        rank
                    ),
                    "trialNumber": int(
                        trial.number
                    ),
                    "globalF1": float(
                        trial.value
                    ),
                    "windowIndex": int(
                        windowIndex
                    ),
                    "windowStart": int(
                        windowStart
                    ),
                    "windowEnd": int(
                        windowEnd
                    ),
                    **metrics,
                    "cumulativeInstances": cumulativeMetrics[
                        "instances"
                    ],
                    "cumulativeF1": cumulativeMetrics[
                        "f1"
                    ],
                }
            )

        return rows
