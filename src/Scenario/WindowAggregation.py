from collections import Counter

import numpy as np
import pandas as pd

from src.Scenario.feature_rules import FEATURE_RULES


class CompletedFlowWindowAggregator:
    diversityColumns = {"Source_IP": "SourceIP", "Destination_IP": "DestinationIP",
                        "Source_Port": "SourcePort", "Destination_Port": "DestinationPort",
                        "Protocol": "Protocol"}

    # Define a duração das janelas alinhadas ao relógio original da captura.
    def __init__(self, windowSeconds=1.0):
        self.windowSeconds = float(windowSeconds)
        if not np.isfinite(self.windowSeconds) or self.windowSeconds <= 0:
            raise ValueError("windowSeconds deve ser positivo e finito.")
        self.frequency = pd.to_timedelta(self.windowSeconds, unit="s")
        if self.frequency.value < 1:
            raise ValueError("A janela deve ter pelo menos um nanossegundo.")
        self.windows = {}
        self.presentFeatures = set()
        self.presentDiversity = set()

    @staticmethod
    # Combina contagem, média e dispersão sem depender das fronteiras dos lotes.
    def mergeMoments(previous, count, mean, squaredDeviations):
        if count <= 0:
            return previous
        if previous is None:
            return [float(count), float(mean), float(squaredDeviations)]
        oldCount, oldMean, oldDeviations = previous
        total = oldCount + count
        difference = mean - oldMean
        return [total, oldMean + difference * count / total,
                oldDeviations + squaredDeviations + difference ** 2 * oldCount * count / total]

    # Atualiza os acumuladores de uma característica conforme sua regra de agregação.
    def updateFeature(self, state, feature, rule, data):
        values = data[feature]
        method = rule["method"]
        valid = values[np.isfinite(values)]
        previous = state["features"].get(feature)
        if method in {"sum", "mean"}:
            count, total = previous or (0, 0.0)
            state["features"][feature] = [count + len(valid), total + float(valid.sum())]
        elif method in {"min", "max"} and valid.size:
            value = float(valid.min() if method == "min" else valid.max())
            state["features"][feature] = value if previous is None else (min(previous, value) if method == "min" else max(previous, value))
        elif method == "weighted_mean" and rule["weight"] in data:
            weights = data[rule["weight"]]
            mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            weight, total = previous or (0.0, 0.0)
            state["features"][feature] = [weight + float(weights[mask].sum()),
                                          total + float((values[mask] * weights[mask]).sum())]
        elif method == "pooled_std" and rule["count"] in data and rule["mean"] in data:
            counts, means = data[rule["count"]], data[rule["mean"]]
            mask = np.isfinite(values) & np.isfinite(counts) & np.isfinite(means) & (counts > 0)
            n, m, s = counts[mask], means[mask], values[mask]
            total = float(n.sum())
            if total > 0:
                mean = float(np.sum(n * m) / total)
                deviations = float(np.sum(np.maximum(n - 1, 0) * s ** 2 + n * (m - mean) ** 2))
                state["features"][feature] = self.mergeMoments(previous, total, mean, deviations)

    # Insere os fluxos de um lote nas janelas determinadas pelo seu encerramento.
    def update(self, flows):
        if flows.empty:
            return
        data = flows.copy()
        numeric = (set(FEATURE_RULES) | {"Source_Port", "Destination_Port", "Protocol"} | {r.get(k) for r in FEATURE_RULES.values() for k in ("weight", "count", "mean")}) & set(data.columns)
        for column in numeric:
            data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fwd = data.get("Total_Fwd_Packets", pd.Series(0, index=data.index)).fillna(0)
        bwd = data.get("Total_Backward_Packets", pd.Series(0, index=data.index)).fillna(0)
        data["__TotalPackets"] = fwd + bwd
        data["__FlowIATCount"] = (fwd + bwd - 1).clip(lower=0)
        data["__FwdIATCount"] = (fwd - 1).clip(lower=0)
        data["__BwdIATCount"] = (bwd - 1).clip(lower=0)
        self.presentFeatures.update(set(FEATURE_RULES) & set(data.columns))
        self.presentDiversity.update(set(self.diversityColumns) & set(data.columns))
        for start, group in data.groupby(data.FlowEnd.dt.floor(self.frequency), sort=False):
            state = self.windows.setdefault(start, {"labels": Counter(), "features": {}, "diversity": {}})
            state["labels"].update(group.Label.value_counts().to_dict())
            for column in self.presentDiversity:
                if column in group:
                    counter = state["diversity"].setdefault(column, Counter())
                    counter.update(group[column].dropna().value_counts().to_dict())
            arrays = {column: group[column].to_numpy(dtype=float, copy=False)
                      for column in numeric | {"__TotalPackets", "__FlowIATCount", "__FwdIATCount", "__BwdIATCount"}}
            for feature in self.presentFeatures:
                if feature in group:
                    self.updateFeature(state, feature, FEATURE_RULES[feature], arrays)

    # Converte os acumuladores de uma janela em estatísticas e rótulo definitivo.
    def finalizeWindow(self, start, state, targetAttack):
        labels = state["labels"]
        foreign = set(labels) - {"BENIGN", targetAttack}
        if foreign:
            raise ValueError(f"Ataques diferentes encontrados na mesma base: {sorted(foreign)}")
        count = sum(labels.values())
        attackCount = 0 if targetAttack == "BENIGN" else labels.get(targetAttack, 0)
        row = {"Timestamp": start + self.frequency, "WindowStart": start,
               "WindowEnd": start + self.frequency, "WindowSeconds": self.windowSeconds,
               "Attack": targetAttack, "Label": targetAttack if attackCount else "BENIGN",
               "FlowCount": count, "AttackCount": attackCount, "BenignCount": labels.get("BENIGN", 0),
               "AttackRatio": attackCount / count}
        for column in sorted(self.presentDiversity):
            suffix = self.diversityColumns[column]
            counter = state["diversity"].get(column, Counter())
            frequencies = np.array(list(counter.values()), dtype=float)
            probabilities = frequencies / frequencies.sum() if frequencies.size else frequencies
            uniqueName = {"SourceIP": "UniqueSourceIPs", "DestinationIP": "UniqueDestinationIPs",
                          "SourcePort": "UniqueSourcePorts", "DestinationPort": "UniqueDestinationPorts",
                          "Protocol": "UniqueProtocols"}[suffix]
            row[uniqueName] = len(counter)
            row[f"{suffix}Entropy"] = float(-np.sum(probabilities * np.log2(probabilities)))
        for feature, rule in FEATURE_RULES.items():
            if feature not in self.presentFeatures:
                continue
            value, method = state["features"].get(feature), rule["method"]
            result = np.nan
            if value is not None:
                if method in {"min", "max"}:
                    result = value
                elif method == "sum":
                    result = value[1] if value[0] else np.nan
                elif method in {"mean", "weighted_mean"}:
                    result = value[1] / value[0] if value[0] else np.nan
                elif method == "pooled_std":
                    result = np.sqrt(max(value[2] / (value[0] - 1), 0)) if value[0] > 1 else 0.0
            if method == "variance":
                result = row.get(rule["std"], np.nan) ** 2
            row[feature] = result
        return row

    # Retorna somente janelas ocupadas em ordem temporal preservando os intervalos originais.
    def toFrame(self, targetAttack):
        frame = pd.DataFrame([self.finalizeWindow(start, self.windows[start], targetAttack)
                              for start in sorted(self.windows)])
        if frame.empty:
            return frame
        frame = frame.drop(columns=["WindowStart", "WindowEnd", "WindowSeconds", "Attack"])
        return frame[[column for column in frame if column != "Label"] + ["Label"]]
