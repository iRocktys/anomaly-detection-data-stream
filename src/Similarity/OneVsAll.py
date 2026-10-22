from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.Data.FeatureAudit import buildFeatureRemovalReport
from src.Similarity.Preprocessing import SimilarityPreprocessor


class SimilarityOneVsAll:
    # Define o Top-K por classe e a execução serial padrão do Random Forest.
    def __init__(self, featuresPerAttack=15, nEstimators=300, randomSeed=42, nJobs=1):
        if isinstance(nEstimators, bool) or not isinstance(nEstimators, Integral) or nEstimators < 1:
            raise ValueError("nEstimators deve ser um inteiro positivo.")
        if isinstance(nJobs, bool) or not isinstance(nJobs, Integral) or nJobs == 0:
            raise ValueError("nJobs deve ser um inteiro diferente de zero.")
        self.featuresPerAttack, self.nEstimators = featuresPerAttack, nEstimators
        self.randomSeed, self.nJobs = randomSeed, nJobs

    # Seleciona a união dos Top-K usando todas as classes e mantém os resultados em memória.
    def run(self, prepared, saveIntermediate=False, intermediateDirectory=None):
        frame, features = prepared["frame"], list(prepared["features"])
        classes = sorted(frame.Label.unique())
        if "BENIGN" not in classes or len(classes) < 2:
            raise ValueError("O RF precisa receber BENIGN e pelo menos uma classe de ataque.")
        values = frame[features].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("O RF recebeu características não finitas.")
        if isinstance(self.featuresPerAttack, dict) and set(self.featuresPerAttack) != set(classes):
            raise ValueError("Informe o Top-K de todas as classes, incluindo BENIGN.")
        perClass = {}
        for label in classes:
            k = self.featuresPerAttack[label] if isinstance(self.featuresPerAttack, dict) else self.featuresPerAttack
            if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
                raise ValueError("O Top-K deve ser um inteiro positivo.")
            target = frame.Label.eq(label).to_numpy(dtype=np.int8)
            model = RandomForestClassifier(n_estimators=self.nEstimators, random_state=self.randomSeed,
                                           n_jobs=self.nJobs, class_weight="balanced")
            model.fit(values, target)
            importances = np.asarray(model.feature_importances_, dtype=float)
            order = np.argsort(-importances, kind="stable")
            perClass[label] = [features[i] for i in order if importances[i] > 0][:int(k)]
        if any(not selected for selected in perClass.values()):
            raise ValueError("Uma classe não apresentou características com importância positiva.")
        union = set().union(*(set(selected) for selected in perClass.values()))
        selected = [feature for feature in features if feature in union]
        removed = buildFeatureRemovalReport([{"Stage": "One-vs-All", "Feature": f,
                    "Reason": "Não selecionada no Top-K de nenhuma classe"} for f in features if f not in union])
        table = pd.DataFrame([{"Class": label, "SelectedCount": len(fs), "Features": ", ".join(fs)}
                              for label, fs in perClass.items()])
        selectedFrame = frame[["Timestamp"] + selected + ["Label"]].copy()
        SimilarityPreprocessor.saveIntermediateFrame(selectedFrame, "03_Selected", saveIntermediate, intermediateDirectory)
        return {"frame": selectedFrame, "selectedFeatures": selected,
                "selectedFeaturesByClass": perClass, "featureTable": table, "normalization": prepared["normalization"],
                "removedFeatures": removed}
