import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SimilarityPreprocessor:
    metadataColumns = {"Timestamp", "Attack", "Label"}
    diagnosticColumns = {"AttackCount", "BenignCount", "OtherAttackCount", "AttackRatio"}

    def __init__(self, nearConstantRatio=0.999, correlationThreshold=0.995, randomSeed=42, missingDisplayThreshold=30.0, noFlowLabel="NOLABEL"):
        # Inicializa balanceamento, diagnóstico de ausências, limpeza de features e padronização global em lote.
        self.nearConstantRatio = float(nearConstantRatio)
        self.correlationThreshold = float(correlationThreshold)
        self.randomSeed = int(randomSeed)
        self.missingDisplayThreshold = float(missingDisplayThreshold)
        self.noFlowLabel = str(noFlowLabel).strip().upper()

    def loadAnalysisFrame(self, datasetPath):
        # Reabre um dataset bruto e mantém somente janelas realmente rotuladas com algum ataque.
        frame = pd.read_csv(datasetPath, low_memory=False)
        if "Label" not in frame.columns:
            raise KeyError("Missing Label column.")
        if "Attack" not in frame.columns:
            frame["Attack"] = frame["Label"].astype(str)
        frame["Label"] = frame["Label"].astype(str).str.upper()
        frame["Attack"] = frame["Attack"].astype(str).str.upper()
        excluded_labels = {"BENIGN", self.noFlowLabel}
        attack_frame = frame.loc[~frame["Label"].isin(excluded_labels)].copy()
        if attack_frame.empty:
            raise ValueError("The dataset contains no attack-labeled windows.")
        attack_frame["Attack"] = attack_frame["Label"]
        return attack_frame.reset_index(drop=True)

    def balanceAttackWindows(self, frame):
        # Aplica random undersampling sem reposição para usar a mesma quantidade de janelas em todos os ataques.
        counts = frame["Attack"].value_counts().sort_index()
        target_count = int(counts.min())
        blocks = []
        summary = {}
        for attack in sorted(counts.index):
            group = frame.loc[frame["Attack"] == attack].copy()
            selected = group.sample(n=target_count, replace=False, random_state=self.randomSeed)
            blocks.append(selected)
            summary[str(attack)] = {
                "availableWindows": int(len(group)),
                "selectedWindows": target_count,
                "droppedWindows": int(len(group) - target_count),
            }
        balanced = pd.concat(blocks, ignore_index=True)
        sort_columns = [column for column in ["Attack", "Timestamp"] if column in balanced.columns]
        if sort_columns:
            balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        return balanced, summary

    def selectNumericFeatures(self, frame, selectedFeatures=None):
        # Seleciona somente features numéricas comportamentais e exclui metadados e métricas derivadas do rótulo.
        if selectedFeatures is not None:
            features = [str(feature) for feature in selectedFeatures]
            missing = [feature for feature in features if feature not in frame.columns]
            if missing:
                raise KeyError(f"Missing selected features: {missing}")
            return features
        excluded = self.metadataColumns | self.diagnosticColumns
        features = []
        for column in frame.columns:
            if column in excluded:
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                features.append(column)
        return features

    def buildMissingSummary(self, frame, features):
        # Resume somente combinações ataque-feature cuja porcentagem de ausências ultrapassa o limiar configurado.
        rows = []
        all_missing = []
        for attack, group in frame.groupby("Attack", sort=True):
            for feature in features:
                values = pd.to_numeric(group[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
                missing = int(values.isna().sum())
                percentage = float(100.0 * missing / len(values)) if len(values) else 100.0
                if missing == len(values):
                    all_missing.append(feature)
                if percentage >= self.missingDisplayThreshold:
                    rows.append({
                        "Attack": attack,
                        "Feature": feature,
                        "MissingCount": missing,
                        "TotalCount": int(len(values)),
                        "MissingPercentage": percentage,
                    })
        return pd.DataFrame(rows), sorted(set(all_missing))

    def imputeByAttack(self, frame, features):
        # Substitui valores inválidos pela mediana da própria feature dentro de cada ataque.
        data = frame.copy()
        unusable = []
        for feature in features:
            data[feature] = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            feature_unusable = False
            for attack, indexes in data.groupby("Attack").groups.items():
                values = data.loc[indexes, feature]
                valid = values.dropna()
                if valid.empty:
                    feature_unusable = True
                    continue
                data.loc[indexes, feature] = values.fillna(float(valid.median()))
            if feature_unusable:
                unusable.append(feature)
        return data, sorted(set(unusable))

    def removeLowInformationFeatures(self, frame, features):
        # Remove features constantes ou quase constantes no dataset balanceado completo.
        kept = []
        removed = {}
        for feature in features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            if values.nunique(dropna=True) <= 1:
                removed[feature] = "constant"
                continue
            top_ratio = float(values.value_counts(normalize=True, dropna=False).iloc[0])
            if top_ratio >= self.nearConstantRatio:
                removed[feature] = f"near_constant:{top_ratio:.6f}"
                continue
            kept.append(feature)
        return kept, removed

    def removeHighlyCorrelatedFeatures(self, frame, features):
        # Remove apenas redundâncias extremas entre features usando correlação absoluta acima do limiar configurado.
        if len(features) < 2:
            return list(features), {}
        correlation = frame[features].corr().abs()
        kept = []
        removed = {}
        for feature in features:
            conflict = None
            for retained in kept:
                value = correlation.loc[feature, retained]
                if pd.notna(value) and float(value) >= self.correlationThreshold:
                    conflict = retained
                    break
            if conflict is None:
                kept.append(feature)
            else:
                removed[feature] = {
                    "correlatedWith": conflict,
                    "absoluteCorrelation": float(correlation.loc[feature, conflict]),
                }
        return kept, removed

    def standardize(self, frame, features):
        # Aplica um único StandardScaler global em lote sobre o dataset balanceado completo.
        scaler = StandardScaler()
        values = scaler.fit_transform(frame[features].to_numpy(dtype=float))
        standardized = frame[[column for column in ["Timestamp", "Attack", "Label"] if column in frame.columns]].copy()
        standardized[features] = values
        return standardized

    def saveFeatureSelection(self, reportDirectory, candidates, unusable, lowRemoved, correlationRemoved, finalFeatures, balanceSummary, missingSummary):
        # Salva em um único JSON a seleção de features, o balanceamento e os diagnósticos necessários à análise.
        payload = {
            "candidateFeatures": list(candidates),
            "removedAllMissingInAtLeastOneAttack": list(unusable),
            "removedLowInformation": dict(lowRemoved),
            "removedHighCorrelation": dict(correlationRemoved),
            "mahalanobisFeatures": list(finalFeatures),
            "mahalanobisFeatureCount": len(finalFeatures),
            "randomUndersampling": {
                "randomSeed": self.randomSeed,
                "perAttack": balanceSummary,
            },
            "missingDisplayThresholdPercentage": self.missingDisplayThreshold,
            "featuresAboveMissingThreshold": missingSummary.to_dict(orient="records"),
        }
        path = Path(reportDirectory) / "feature_selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def runFromFile(self, datasetPath, outputDirectory, selectedFeatures=None):
        # Reabre o dataset bruto, balanceia, limpa, imputa e salva apenas os artefatos necessários para a análise posterior.
        output_directory = Path(outputDirectory)
        dataset_directory = output_directory / "datasets"
        report_directory = output_directory / "reports"
        dataset_directory.mkdir(parents=True, exist_ok=True)
        report_directory.mkdir(parents=True, exist_ok=True)
        attack_frame = self.loadAnalysisFrame(datasetPath)
        balanced, balance_summary = self.balanceAttackWindows(attack_frame)
        candidates = self.selectNumericFeatures(balanced, selectedFeatures)
        missing_summary, missing_everywhere = self.buildMissingSummary(balanced, candidates)
        imputed, unusable = self.imputeByAttack(balanced, candidates)
        unusable = sorted(set(unusable) | set(missing_everywhere))
        eligible = [feature for feature in candidates if feature not in unusable]
        low_kept, low_removed = self.removeLowInformationFeatures(imputed, eligible)
        final_features, correlation_removed = self.removeHighlyCorrelatedFeatures(imputed, low_kept)
        if not final_features:
            raise ValueError("No feature remained available for similarity analysis.")
        standardized = self.standardize(imputed, final_features)
        balanced_path = dataset_directory / "balanced_attack_windows.csv"
        imputed_path = dataset_directory / "imputed_attack_windows.csv"
        standardized_path = dataset_directory / "standardized_attack_windows.csv"
        balanced.to_csv(balanced_path, index=False)
        imputed.to_csv(imputed_path, index=False)
        standardized.to_csv(standardized_path, index=False)
        feature_selection_path = self.saveFeatureSelection(
            report_directory,
            candidates,
            unusable,
            low_removed,
            correlation_removed,
            final_features,
            balance_summary,
            missing_summary,
        )
        return {
            "balancedDatasetPath": str(balanced_path),
            "imputedDatasetPath": str(imputed_path),
            "standardizedDatasetPath": str(standardized_path),
            "featureSelectionPath": str(feature_selection_path),
            "features": final_features,
            "missingSummary": missing_summary,
            "balanceSummary": balance_summary,
        }
