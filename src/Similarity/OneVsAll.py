import json
from collections import Counter
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class SimilarityOneVsAll:
    """Saída: features por ataque e seleção comum, sem avaliação de classificação."""

    def __init__(self, featuresPerAttack=10, extractionMode="union", minAttackOccurrences=2,
                 nEstimators=300, randomSeed=42, nJobs=-1):
        if extractionMode not in (None, "union", "recurrence"):
            raise ValueError("extractionMode must be 'union', 'recurrence' or None.")
        if not isinstance(minAttackOccurrences, Integral) or minAttackOccurrences < 1:
            raise ValueError("minAttackOccurrences must be a positive integer.")
        self.featuresPerAttack = featuresPerAttack
        self.extractionMode = extractionMode
        self.minAttackOccurrences = int(minAttackOccurrences)
        self.nEstimators = nEstimators
        self.randomSeed = randomSeed
        self.nJobs = nJobs

    def runFromFile(self, datasetPath, featureSelectionPath, outputDirectory):
        frame = pd.read_csv(datasetPath, low_memory=False)
        payload = json.loads(Path(featureSelectionPath).read_text(encoding="utf-8"))
        features = payload["preparedFeatures"]
        attacks = sorted(frame["Attack"].unique())
        if len(attacks) < 2:
            raise ValueError("One-Vs-All requires at least two attacks.")
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError("One-Vs-All input has missing/infinite values. Enable imputation or clean the input.")
        per_attack = {}
        if self.extractionMode is None:
            final = list(features)
        else:
            if isinstance(self.featuresPerAttack, dict) and set(self.featuresPerAttack) != set(attacks):
                raise ValueError("featuresPerAttack must define a quantity for every attack.")
            for attack in attacks:
                k = self.featuresPerAttack[attack] if isinstance(self.featuresPerAttack, dict) else self.featuresPerAttack
                if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
                    raise ValueError("featuresPerAttack must be a positive integer or a dictionary of integers.")
                model = RandomForestClassifier(n_estimators=self.nEstimators, random_state=self.randomSeed,
                                               n_jobs=self.nJobs, class_weight="balanced")
                model.fit(frame[features], frame["Attack"].eq(attack).astype(int))
                # Desempate determinístico pela ordem de entrada. Importância zero não é selecionada.
                indices = np.argsort(-model.feature_importances_, kind="stable")
                per_attack[attack] = [features[i] for i in indices if model.feature_importances_[i] > 0][:k]
            frequency = Counter(f for selected in per_attack.values() for f in selected)
            minimum = 1 if self.extractionMode == "union" else self.minAttackOccurrences
            final = [f for f in features if frequency[f] >= minimum]
            if not final:
                raise ValueError("No feature meets the configured recurrence/importance criterion; change K or minimum recurrence.")
        frequency = Counter(f for selected in per_attack.values() for f in selected)
        result = {"extractionMode": self.extractionMode, "featuresPerAttack": self.featuresPerAttack,
                  "minAttackOccurrences": self.minAttackOccurrences, "selectedFeaturesByAttack": per_attack,
                  "featureOccurrences": dict(frequency), "selectedFeatures": final, "featureCount": len(final),
                  "randomForest": {"nEstimators": self.nEstimators, "randomSeed": self.randomSeed,
                                   "classWeight": "balanced", "zeroImportanceExcluded": True},
                  "requestedVsSelected": {a: {"requested": self.featuresPerAttack[a] if isinstance(self.featuresPerAttack, dict)
                                                  else self.featuresPerAttack, "selected": len(fs)} for a, fs in per_attack.items()}}
        root = Path(outputDirectory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "one_vs_all_selected_features.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return {**result, "selectionPath": str(path)}
