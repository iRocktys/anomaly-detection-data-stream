import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SimilarityPreprocessor:
    """Preparação em lote; None desativa cada técnica opcional explicitamente."""

    metadataColumns = {"Timestamp", "Attack", "Label", "OriginalTimestamp", "FlowStart", "FlowEnd",
                       "OriginalFlowStart", "OriginalFlowEnd", "SourceFile", "SourceRow", "BlockIndex",
                       "StreamIndex", "WindowIndex", "ScenarioPart", "AttackBlock",
                       "Flow_ID", "Source_IP", "Destination_IP", "SimillarHTTP"}
    diagnosticColumns = {"AttackCount", "BenignCount", "OtherAttackCount", "AttackRatio"}

    def __init__(self, nearConstantRatio=0.999, correlationThreshold=0.995, randomSeed=42,
                 missingDisplayThreshold=30.0, noFlowLabel="NOLABEL", downsample="random",
                 removeZeroColumns=True, removeAllMissingColumns=True, removeConstantColumns=True,
                 imputation="median", normalization="standard", sampleUnit="window"):
        if downsample not in (None, "random"):
            raise ValueError("downsample must be 'random' or None.")
        if imputation not in (None, "median") or normalization not in (None, "standard"):
            raise ValueError("Use imputation='median'/None and normalization='standard'/None.")
        for name, value in (("nearConstantRatio", nearConstantRatio), ("correlationThreshold", correlationThreshold)):
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1] or None.")
        for value in (removeZeroColumns, removeAllMissingColumns, removeConstantColumns):
            if value is not True and value is not None:
                raise ValueError("Removal options accept True or None.")
        if missingDisplayThreshold is not None and not 0 <= missingDisplayThreshold <= 100:
            raise ValueError("missingDisplayThreshold must be in [0, 100] or None.")
        if sampleUnit not in ("window", "flow"):
            raise ValueError("sampleUnit must be window or flow.")
        self.sampleUnit = sampleUnit
        self.nearConstantRatio = nearConstantRatio
        self.correlationThreshold = correlationThreshold
        self.randomSeed = randomSeed
        self.missingDisplayThreshold = missingDisplayThreshold
        self.noFlowLabel = noFlowLabel.upper()
        self.downsample = downsample
        self.removeZeroColumns = removeZeroColumns
        self.removeAllMissingColumns = removeAllMissingColumns
        self.removeConstantColumns = removeConstantColumns
        self.imputation = imputation
        self.normalization = normalization

    def loadAnalysisFrame(self, datasetPath):
        frame = pd.read_csv(datasetPath, low_memory=False)
        if "Label" not in frame:
            raise KeyError("Missing Label column.")
        frame["Label"] = frame["Label"].astype(str).str.strip().str.upper()
        if "Attack" in frame:
            target = frame["Attack"].astype(str).str.strip().str.upper()
            foreign = ~frame["Label"].isin({"BENIGN", "NORMAL", "0", self.noFlowLabel}) & frame["Label"].ne(target)
            if foreign.any():
                raise ValueError("A sample contains a label different from its target Attack.")
        frame = frame.loc[~frame["Label"].isin({"BENIGN", "NORMAL", "0", self.noFlowLabel})].copy()
        if frame.empty:
            raise ValueError("The dataset contains no attack-labeled samples.")
        frame["Attack"] = frame["Label"]
        return frame.reset_index(drop=True)

    def balanceAttackWindows(self, frame):
        counts = frame["Attack"].value_counts().sort_index()
        blocks, summary = [], {}
        for attack in counts.index:
            group = frame.loc[frame["Attack"] == attack]
            selected = group.sample(n=int(counts.min()), random_state=self.randomSeed) if self.downsample == "random" else group
            blocks.append(selected)
            unit = "Flows" if self.sampleUnit == "flow" else "Windows"
            summary[attack] = {f"available{unit}": len(group), f"selected{unit}": len(selected),
                              f"dropped{unit}": len(group) - len(selected)}
        # Índices de origem conservam a ordem do Stream e garantem as mesmas janelas Mixed/Attack-Only.
        selected = pd.concat(blocks).sort_index(kind="stable").reset_index(drop=True)
        return selected, summary

    def selectNumericFeatures(self, frame, selectedFeatures=None):
        excluded = self.metadataColumns | self.diagnosticColumns | {c for c in frame if c.startswith("Unnamed") or c.lower() in {"index", "flow_id"}}
        if selectedFeatures is not None:
            features = list(dict.fromkeys(selectedFeatures))
            if any(f not in frame or f in excluded for f in features):
                raise ValueError("Selected features must exist and cannot contain metadata or label diagnostics.")
            return features
        return [c for c in frame if c not in excluded and
                (pd.api.types.is_numeric_dtype(frame[c]) or pd.to_numeric(frame[c], errors="coerce").notna().any()
                 or frame[c].isna().all())]

    def removeLowInformationFeatures(self, frame, features):
        kept, removed = [], {}
        for f in features:
            values = frame[f].dropna()
            if self.removeConstantColumns is not None and values.nunique() == 1:
                removed[f] = "constant"
            elif self.nearConstantRatio is not None and not values.empty and values.value_counts(normalize=True).iloc[0] >= self.nearConstantRatio:
                removed[f] = "near_constant"
            else:
                kept.append(f)
        return kept, removed

    def removeHighlyCorrelatedFeatures(self, frame, features):
        if self.correlationThreshold is None:
            return list(features), {}
        correlation = frame[features].corr().abs()
        kept, removed = [], {}
        for f in features:
            conflicts = [k for k in kept if pd.notna(correlation.loc[f, k]) and correlation.loc[f, k] >= self.correlationThreshold]
            if conflicts:
                removed[f] = {"correlatedWith": conflicts[0], "absoluteCorrelation": float(correlation.loc[f, conflicts[0]])}
            else:
                kept.append(f)
        return kept, removed

    def standardize(self, frame, features):
        output = frame[[c for c in frame if c in self.metadataColumns]].copy()
        output[features] = (StandardScaler().fit_transform(frame[features]) if self.normalization == "standard"
                            else frame[features].to_numpy())
        return output

    def runFromFile(self, datasetPath, outputDirectory, selectedFeatures=None):
        root = Path(outputDirectory)
        (root / "datasets").mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        balanced, balance = self.balanceAttackWindows(self.loadAnalysisFrame(datasetPath))
        candidates = self.selectNumericFeatures(balanced, selectedFeatures)
        data = balanced.copy()
        for f in candidates:
            data[f] = pd.to_numeric(data[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_rows = []
        if self.missingDisplayThreshold is not None:
            for attack, group in data.groupby("Attack", sort=True):
                for f in candidates:
                    percentage = float(group[f].isna().mean() * 100)
                    if percentage >= self.missingDisplayThreshold:
                        missing_rows.append({"Attack": attack, "Feature": f, "MissingCount": int(group[f].isna().sum()),
                                             "TotalCount": len(group), "MissingPercentage": percentage})
        missing = pd.DataFrame(missing_rows, columns=["Attack", "Feature", "MissingCount", "TotalCount", "MissingPercentage"])
        all_missing = [f for f in candidates if data[f].notna().sum() == 0] if self.removeAllMissingColumns is not None else []
        zeros = [f for f in candidates if data[f].notna().any() and data[f].dropna().eq(0).all()] if self.removeZeroColumns is not None else []
        eligible = [f for f in candidates if f not in set(all_missing + zeros)]
        eligible, low_removed = self.removeLowInformationFeatures(data, eligible)
        if not eligible:
            raise ValueError("No feature remained after pre-imputation cleaning.")
        medians = data[eligible].median() if self.imputation == "median" else pd.Series(dtype=float)
        if self.imputation == "median":
            data[eligible] = data[eligible].fillna(medians)
        # Armazena somente features elegíveis e metadados; colunas removidas não reaparecem na seleção RF.
        imputed = data[[c for c in data if c in self.metadataColumns] + eligible].copy()
        final, correlation_removed = self.removeHighlyCorrelatedFeatures(imputed, eligible)
        if not final:
            raise ValueError("No feature remained after selection.")
        standardized = self.standardize(imputed, final)
        suffix = "flows" if self.sampleUnit == "flow" else "windows"
        paths = {"balancedDatasetPath": root / f"datasets/balanced_attack_{suffix}.csv",
                 "imputedDatasetPath": root / f"datasets/imputed_attack_{suffix}.csv",
                 "standardizedDatasetPath": root / f"datasets/standardized_attack_{suffix}.csv",
                 "featureSelectionPath": root / "reports/feature_selection.json"}
        balanced.to_csv(paths["balancedDatasetPath"], index=False)
        imputed.to_csv(paths["imputedDatasetPath"], index=False)
        standardized.to_csv(paths["standardizedDatasetPath"], index=False)
        payload = {"sourceDataset": str(Path(datasetPath).resolve()), "sampleUnit": self.sampleUnit, "windowSeconds": 1 if self.sampleUnit == "window" else None,
                   "candidateFeatures": candidates, "preparedFeatures": eligible, "removedZeroColumns": zeros,
                   "removedAllMissingGlobally": all_missing, "removedLowInformation": low_removed,
                   "removedHighCorrelation": correlation_removed, "mahalanobisFeatures": final,
                   "mahalanobisFeatureCount": len(final), "globalMedians": {k: None if pd.isna(v) else float(v) for k, v in medians.items()},
                   "randomUndersampling": {"enabled": self.downsample is not None, "method": self.downsample,
                                             "randomSeed": self.randomSeed, "perAttack": balance},
                   "settings": {k: getattr(self, k) for k in ("nearConstantRatio", "correlationThreshold", "removeZeroColumns",
                       "removeAllMissingColumns", "removeConstantColumns", "imputation", "normalization", "missingDisplayThreshold")},
                   "featuresAboveMissingThreshold": missing.to_dict(orient="records")}
        paths["featureSelectionPath"].write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return {**{k: str(v) for k, v in paths.items()}, "features": final, "preparedFeatures": eligible,
                "missingSummary": missing, "balanceSummary": balance}
