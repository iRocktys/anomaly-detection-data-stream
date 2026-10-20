
import numpy as np
import pandas as pd

from src.Data.FeatureAudit import buildFeatureRemovalReport


class SimilarityPreprocessor:
    metadataColumns = {"Timestamp", "WindowStart", "WindowEnd", "WindowSeconds", "Attack", "Label",
                       "OriginalTimestamp", "OriginalWindowStart", "OriginalWindowEnd", "FlowStart", "FlowEnd",
                       "SourceFile", "SourceRow", "SourceDataset", "SourceWindowRow", "TargetAttack", "InstanceIndex",
                       "BlockIndex", "ScenarioPart", "AttackBlock", "Flow_ID", "Source_IP", "Destination_IP", "SimillarHTTP",
                       "TemporalStratum", "SamplingSeed"}
    diagnosticColumns = {"AttackCount", "BenignCount", "OtherAttackCount", "AttackRatio"}

    # Configura a limpeza de todas as classes sem realizar amostragem antes do Random Forest.
    def __init__(self, nearConstantRatio=0.999,
                 removeZeroColumns=True, removeAllMissingColumns=True, removeConstantColumns=True,
                 imputation="median", normalization="standard", windowSeconds=1.0):
        if imputation not in (None, "median") or normalization not in (None, "standard"):
            raise ValueError("Use imputation='median'/None e normalization='standard'/None.")
        if nearConstantRatio is not None and not 0 < nearConstantRatio <= 1:
            raise ValueError("nearConstantRatio deve estar em (0, 1] ou ser None.")
        if any(v is not True and v is not None for v in (removeZeroColumns, removeAllMissingColumns, removeConstantColumns)):
            raise ValueError("As opções de remoção aceitam True ou None.")
        self.windowSeconds = float(windowSeconds)
        if not np.isfinite(self.windowSeconds) or self.windowSeconds != 1.0:
            raise ValueError("A similaridade utiliza janelas de 1 segundo.")
        self.nearConstantRatio = nearConstantRatio
        self.removeZeroColumns, self.removeAllMissingColumns = removeZeroColumns, removeAllMissingColumns
        self.removeConstantColumns, self.imputation, self.normalization = removeConstantColumns, imputation, normalization

    # Carrega todas as janelas do cenário e mantém BENIGN como uma classe explícita.
    def loadAnalysisFrame(self, datasetPath):
        frame = pd.read_csv(datasetPath, low_memory=False)
        if not frame.columns.is_unique or not {"Label", "Timestamp"}.issubset(frame):
            raise ValueError("O dataset precisa ter colunas distintas, Timestamp e Label.")
        labels = frame.Label.astype("string").str.strip().str.upper()
        if (labels.isna() | labels.fillna("").isin({"", "NAN", "NONE", "NULL", "NOLABEL"})).any():
            raise ValueError("Existem rótulos inválidos ou janelas sem observações.")
        frame["Label"] = labels.replace({"NORMAL": "BENIGN", "0": "BENIGN"}).astype(str)
        times = pd.to_datetime(frame.Timestamp, format="mixed", errors="raise")
        if times.isna().any() or times.duplicated().any():
            raise ValueError("Os timestamps reconstruídos precisam ser válidos e únicos.")
        if len(times) > 1 and not times.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all():
            raise ValueError("O dataset precisa ser contínuo com janelas de 1 segundo.")
        frame["Timestamp"] = times
        if "BENIGN" not in set(frame.Label) or frame.Label.nunique() < 2:
            raise ValueError("O dataset precisa conter BENIGN e pelo menos um tipo de ataque.")
        return frame.reset_index(drop=True)

    # Registra identificadores, diagnósticos e exclusões explícitas antes de selecionar as candidatas.
    def selectNumericFeatures(self, frame, selectedFeatures=None, removedFeatures=None):
        selected = None if selectedFeatures is None else list(dict.fromkeys(selectedFeatures))
        removed = set(removedFeatures or [])
        if selected is not None and any(f not in frame or f in self.metadataColumns | self.diagnosticColumns for f in selected):
            raise ValueError("As candidatas informadas devem existir e não podem ser alvos ou metadados.")
        candidates, records = [], []
        for feature in frame:
            reason = None
            if feature == "Label":
                reason = "Alvo supervisionado; não é um preditor"
            elif feature in self.metadataColumns or feature.startswith("Unnamed") or feature.lower() == "index":
                reason = "Identificador ou metadado"
            elif feature in self.diagnosticColumns:
                reason = "Diagnóstico derivado da rotulagem"
            elif feature in removed:
                reason = "Exclusão explícita em removedFeatures"
            elif selected is not None and feature not in selected:
                reason = "Fora da lista selectedFeatures"
            elif not (pd.api.types.is_numeric_dtype(frame[feature]) or pd.to_numeric(frame[feature], errors="coerce").notna().any() or frame[feature].isna().all()):
                reason = "Coluna sem valores numéricos"
            if reason:
                records.append({"Stage": "Preprocessing", "Feature": feature, "Reason": reason})
            else:
                candidates.append(feature)
        return candidates, records

    # Prepara todas as classes em memória sem gerar arquivos intermediários.
    def runFromFile(self, datasetPath, selectedFeatures=None, removedFeatures=None):
        frame = self.loadAnalysisFrame(datasetPath)
        candidates, records = self.selectNumericFeatures(frame, selectedFeatures, removedFeatures)
        data, eligible = frame.copy(), []
        for feature in candidates:
            data[feature] = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            values, reason = data[feature].dropna(), None
            if self.removeAllMissingColumns and values.empty:
                reason = "Todos os valores ausentes ou não finitos"
            elif self.removeZeroColumns and not values.empty and values.eq(0).all():
                reason = "Todos os valores válidos iguais a zero"
            elif self.removeConstantColumns and values.nunique() == 1:
                reason = "Característica constante"
            elif self.nearConstantRatio is not None and not values.empty and values.value_counts(normalize=True).iloc[0] >= self.nearConstantRatio:
                reason = f"Característica quase constante; frequência dominante >= {self.nearConstantRatio}"
            if reason:
                records.append({"Stage": "Preprocessing", "Feature": feature, "Reason": reason})
            else:
                eligible.append(feature)
        if not eligible:
            raise ValueError("Nenhuma característica permaneceu após a limpeza.")
        medians = data[eligible].median() if self.imputation == "median" else pd.Series(dtype=float)
        if self.imputation == "median":
            data[eligible] = data[eligible].fillna(medians)
        invalid = [f for f in eligible if not np.isfinite(data[f].to_numpy(dtype=float)).all()]
        if invalid:
            raise ValueError(f"Features ainda não finitas após a imputação: {invalid}")
        return {"frame": data[["Timestamp"] + eligible + ["Label"]].copy(), "features": eligible,
                "normalization": self.normalization, "removedFeatures": buildFeatureRemovalReport(records)}
