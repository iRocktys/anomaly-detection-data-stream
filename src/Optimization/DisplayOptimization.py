import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.Plots.Plots import Plots


class DisplayOptimization:
    def __init__(
        self,
        trialsPath,
        datasetName,
    ):
        # Inicializa a análise manual de uma otimização já concluída para um único dataset.
        self.trialsPath = Path(
            trialsPath
        )

        self.datasetName = str(
            datasetName
        ).strip()

        if not self.trialsPath.exists():
            raise FileNotFoundError(
                f"Arquivo trials.csv não encontrado: {self.trialsPath}"
            )

        if not self.datasetName:
            raise ValueError(
                "datasetName deve ser informado."
            )

        self.optimizationDirectory = (
            self.trialsPath.parent
        )

        self.windowsPath = (
            self.optimizationDirectory
            / "top10_windows.csv"
        )

        self.datasetDirectory = (
            self.optimizationDirectory
            / self.normalizeName(
                self.datasetName
            )
        )

        self.plotDirectory = (
            self.datasetDirectory
            / "plots"
        )

    def getBestTrial(self):
        # Seleciona somente o trial concluído com maior objectiveF1 para o dataset informado.
        trials = pd.read_csv(
            self.trialsPath
        )

        datasetColumn = self.findDatasetColumn(
            trials
        )

        objectiveColumn = self.findObjectiveColumn(
            trials
        )

        selected = trials[
            trials[
                datasetColumn
            ].astype(
                str
            )
            == self.datasetName
        ].copy()

        if "state" in selected.columns:
            selected = selected[
                selected[
                    "state"
                ].astype(
                    str
                ).str.upper()
                == "COMPLETE"
            ]

        selected[
            objectiveColumn
        ] = pd.to_numeric(
            selected[
                objectiveColumn
            ],
            errors="coerce",
        )

        selected = selected.dropna(
            subset=[
                objectiveColumn
            ]
        )

        if selected.empty:
            raise ValueError(
                f"Nenhum trial válido foi encontrado para '{self.datasetName}'."
            )

        selected = selected.sort_values(
            [
                objectiveColumn,
                "trialNumber",
            ],
            ascending=[
                False,
                True,
            ],
        )

        return selected.iloc[
            0
        ].copy()

    def getBestParameters(self):
        # Reconstrói o dicionário de parâmetros pronto para uso no ExperimentRunner.
        best = self.getBestTrial()

        scoreMode = self.getValue(
            best,
            "param_scoreMode",
            "raw",
        )

        if (
            str(
                scoreMode
            )
            == "movingAverage"
        ):
            movingAverageWindow = int(
                self.getValue(
                    best,
                    "param_movingAverageWindow",
                    10,
                )
            )

            scoreSource = (
                f"ma{movingAverageWindow}"
            )

            scoreWindowSizes = [
                movingAverageWindow
            ]

        else:
            scoreSource = "raw"
            scoreWindowSizes = []

        driftDepth = int(
            self.getValue(
                best,
                "param_driftDepth",
                50,
            )
        )

        calibrationWindow = 500

        calibrationSize = (
            calibrationWindow
            - driftDepth
        )

        parameters = {
            "model": {
                "code": str(
                    self.getValue(
                        best,
                        "modelCode",
                        "AIF",
                    )
                ),
                "parameters": {
                    "window_size": int(
                        self.getModelValue(
                            best,
                            "window_size",
                        )
                    ),
                    "n_trees": int(
                        self.getModelValue(
                            best,
                            "n_trees",
                        )
                    ),
                    "height": int(
                        self.getModelValue(
                            best,
                            "height",
                        )
                    ),
                    "m_trees": int(
                        self.getModelValue(
                            best,
                            "m_trees",
                        )
                    ),
                    "weights": float(
                        self.getModelValue(
                            best,
                            "weights",
                        )
                    ),
                },
            },
            "preprocessing": {
                "imputer": "incrementalMean",
                "normalizer": "incrementalZScore",
            },
            "training": {
                "strategy": "all",
            },
            "threshold": {
                "name": "dspot",
                "scoreSource": scoreSource,
                "scoreWindowSizes": scoreWindowSizes,
                "parameters": {
                    "driftDepth": driftDepth,
                    "calibrationSize": calibrationSize,
                    "initialQuantile": float(
                        self.getValue(
                            best,
                            "param_initialQuantile",
                        )
                    ),
                    "risk": float(
                        self.getValue(
                            best,
                            "param_risk",
                        )
                    ),
                    "refitEvery": int(
                        self.getValue(
                            best,
                            "param_refitEvery",
                        )
                    ),
                    "optimizationStarts": int(
                        self.getValue(
                            best,
                            "param_optimizationStarts",
                            10,
                        )
                    ),
                    "tolerance": float(
                        self.getValue(
                            best,
                            "param_tolerance",
                            1e-8,
                        )
                    ),
                },
            },
        }

        return parameters

    def showBest(self):
        # Exibe o número do melhor trial, seu F1 e devolve os parâmetros prontos para treinamento.
        best = self.getBestTrial()

        objectiveColumn = self.findObjectiveColumn(
            pd.DataFrame(
                [
                    best
                ]
            )
        )

        print(
            "Dataset:",
            self.datasetName,
        )

        print(
            "Trial:",
            int(
                best[
                    "trialNumber"
                ]
            ),
        )

        print(
            "F1:",
            float(
                best[
                    objectiveColumn
                ]
            ),
        )

        parameters = self.getBestParameters()

        print(
            "\nParâmetros:"
        )

        return parameters

    def plotBest(self):
        # Gera manualmente os gráficos janelados do melhor trial e salva-os na pasta do dataset.
        if not self.windowsPath.exists():
            raise FileNotFoundError(
                "O arquivo top10_windows.csv não foi encontrado ao lado de trials.csv."
            )

        best = self.getBestTrial()

        trialNumber = int(
            best[
                "trialNumber"
            ]
        )

        windows = pd.read_csv(
            self.windowsPath
        )

        datasetColumn = self.findDatasetColumn(
            windows
        )

        selected = windows[
            (
                windows[
                    datasetColumn
                ].astype(
                    str
                )
                == self.datasetName
            )
            & (
                pd.to_numeric(
                    windows[
                        "trialNumber"
                    ],
                    errors="coerce",
                )
                == trialNumber
            )
        ].copy()

        if selected.empty:
            raise ValueError(
                "As métricas janeladas do melhor trial não foram encontradas."
            )

        selected = selected.sort_values(
            "windowIndex"
            if "windowIndex" in selected.columns
            else "windowEnd"
        ).reset_index(
            drop=True
        )

        self.datasetDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.plotDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        bestWindowsPath = (
            self.datasetDirectory
            / "best_windows.csv"
        )

        selected.to_csv(
            bestWindowsPath,
            index=False,
        )

        if "windowSize" in selected.columns:
            windowSize = int(
                selected[
                    "windowSize"
                ].iloc[
                    0
                ]
            )

        elif len(
            selected
        ) > 0:
            first = int(
                selected[
                    "windowStart"
                ].iloc[
                    0
                ]
            )

            last = int(
                selected[
                    "windowEnd"
                ].iloc[
                    0
                ]
            )

            windowSize = (
                last
                - first
                + 1
            )

        else:
            windowSize = 100

        plots = Plots()

        errorPath = plots.plotWindowErrors(
            selected,
            attackSource=None,
            outputPath=(
                self.plotDirectory
                / "fp_fn_windows.png"
            ),
            windowSize=windowSize,
        )

        metricsPath = plots.plotWindowMetrics(
            selected,
            attackSource=None,
            outputPath=(
                self.plotDirectory
                / "metrics_windows.png"
            ),
            windowSize=windowSize,
        )

        return {
            "datasetDirectory": str(
                self.datasetDirectory
            ),
            "bestWindowsPath": str(
                bestWindowsPath
            ),
            "fpFnWindowPath": str(
                errorPath
            ),
            "metricsWindowPath": str(
                metricsPath
            ),
        }

    def getValue(
        self,
        row,
        column,
        default=None,
    ):
        # Lê um valor do trial tratando campos ausentes ou NaN.
        value = row.get(
            column,
            default,
        )

        if pd.isna(
            value
        ):
            return default

        return value

    def getModelValue(
        self,
        row,
        name,
    ):
        # Recupera um parâmetro AIF preferindo o valor efetivo salvo pelo exporter.
        candidates = [
            f"model_{name}",
            f"param_aif_{name}",
        ]

        for column in candidates:
            value = self.getValue(
                row,
                column,
                None,
            )

            if value is not None:
                return value

        raise ValueError(
            f"Parâmetro do modelo não encontrado: {name}."
        )

    def findDatasetColumn(
        self,
        frame,
    ):
        # Localiza a coluna de identificação do dataset em versões novas ou antigas do exporter.
        for column in (
            "datasetKey",
            "dataset",
            "scenario",
        ):
            if column in frame.columns:
                return column

        raise ValueError(
            "Nenhuma coluna de identificação do dataset foi encontrada."
        )

    def findObjectiveColumn(
        self,
        frame,
    ):
        # Localiza a coluna usada como objetivo F1 no arquivo de trials.
        for column in (
            "objectiveF1",
            "f1",
        ):
            if column in frame.columns:
                return column

        raise ValueError(
            "Nenhuma coluna de F1 objetivo foi encontrada."
        )

    def normalizeName(
        self,
        value,
    ):
        # Sanitiza o nome lógico do dataset somente para criação segura de diretórios.
        asciiName = (
            unicodedata.normalize(
                "NFKD",
                str(
                    value
                ).strip(),
            )
            .encode(
                "ascii",
                "ignore",
            )
            .decode(
                "ascii"
            )
        )

        safeName = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            asciiName,
        ).strip(
            "_"
        )

        return (
            safeName
            or "dataset"
        )
