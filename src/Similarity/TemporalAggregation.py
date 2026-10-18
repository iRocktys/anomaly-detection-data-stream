from pathlib import Path

import numpy as np
import pandas as pd

from src.Scenario.aggregation import aggregate_window
from src.Scenario.entropy import calculate_entropies


class SimilarityTemporalAggregator:
    uniqueColumns = {
        "Source_IP": "UniqueSourceIPs",
        "Destination_IP": "UniqueDestinationIPs",
        "Source_Port": "UniqueSourcePorts",
        "Destination_Port": "UniqueDestinationPorts",
        "Protocol": "UniqueProtocols",
    }

    def __init__(self, windowSeconds=1.0, labelColumn="Label", normalLabels=None, fillEmptyWindows=True, noFlowLabel="NOLABEL"):
        # Inicializa a agregação temporal pelo término do fluxo e define o rótulo exclusivo para janelas sem fluxo.
        self.windowSeconds = float(windowSeconds)
        if not np.isfinite(self.windowSeconds) or self.windowSeconds <= 0:
            raise ValueError("windowSeconds must be positive and finite.")
        self.windowFrequency = pd.to_timedelta(self.windowSeconds, unit="s")
        self.labelColumn = str(labelColumn)
        self.normalLabels = {str(value).strip().upper() for value in (normalLabels or ["BENIGN", "NORMAL", "0"])}
        self.fillEmptyWindows = bool(fillEmptyWindows)
        self.noFlowLabel = str(noFlowLabel).strip().upper()

    def calculateUniqueCounts(self, flows):
        # Conta a quantidade de valores distintos de endereços, portas e protocolos dentro da janela.
        result = {}
        for source, target in self.uniqueColumns.items():
            if source in flows.columns:
                result[target] = int(flows[source].dropna().nunique())
        return result

    def emptyWindow(self, targetAttack, windowEnd):
        # Cria uma janela NOLABEL para representar explicitamente um intervalo sem qualquer fluxo encerrado.
        row = {
            "Timestamp": windowEnd,
            "Attack": str(targetAttack).upper(),
            "Label": self.noFlowLabel,
            "FlowCount": 0,
            "AttackCount": 0,
            "BenignCount": 0,
            "OtherAttackCount": 0,
            "AttackRatio": np.nan,
        }
        for target in self.uniqueColumns.values():
            row[target] = 0
        for target in ("SourceIPEntropy", "DestinationIPEntropy", "SourcePortEntropy", "DestinationPortEntropy", "ProtocolEntropy"):
            row[target] = np.nan
        return row

    def aggregateWindow(self, flows, targetAttack, windowEnd):
        # Resume os fluxos encerrados na janela e calcula composição, diversidade, entropia e features agregadas.
        labels = flows[self.labelColumn].astype(str).str.upper()
        target = str(targetAttack).upper()
        attack_count = int((labels == target).sum())
        benign_count = int(labels.isin(self.normalLabels).sum())
        other_attack_count = int(len(flows) - attack_count - benign_count)
        row = {
            "Timestamp": windowEnd,
            "Attack": target,
            "Label": target if attack_count > 0 else "BENIGN",
            "FlowCount": int(len(flows)),
            "AttackCount": attack_count,
            "BenignCount": benign_count,
            "OtherAttackCount": other_attack_count,
            "AttackRatio": float(attack_count / len(flows)) if len(flows) else np.nan,
        }
        row.update(self.calculateUniqueCounts(flows))
        row.update(calculate_entropies(flows))
        row.update(aggregate_window(flows))
        return row

    def aggregateAttack(self, flows, targetAttack, cutStart, cutEnd, mode="mixed_traffic"):
        # Agrega um recorte em modo somente ataque ou tráfego misto usando FlowEnd para posicionar cada fluxo.
        data = flows.copy()
        if "FlowEnd" not in data.columns:
            raise KeyError("Missing FlowEnd column.")
        data["FlowEnd"] = pd.to_datetime(data["FlowEnd"], errors="coerce")
        data = data.dropna(subset=["FlowEnd"]).copy()
        target = str(targetAttack).upper()
        mode_value = str(mode).strip().lower()
        if mode_value not in {"attack_only", "mixed_traffic"}:
            raise ValueError("mode must be 'attack_only' or 'mixed_traffic'.")
        labels = data[self.labelColumn].astype(str).str.strip().str.upper()
        data = data.loc[labels.isin(self.normalLabels | {target})].copy()
        if mode_value == "attack_only":
            data = data.loc[data[self.labelColumn].astype(str).str.upper() == target].copy()
        data = data.loc[(data["FlowEnd"] >= cutStart) & (data["FlowEnd"] < cutEnd)].copy()
        origin = pd.Timestamp(cutStart)
        expected_count = int(np.ceil((pd.Timestamp(cutEnd) - origin).total_seconds() / self.windowSeconds))
        if data.empty:
            grouped = {}
        else:
            delta_seconds = (data["FlowEnd"] - origin).dt.total_seconds()
            data["WindowIndex"] = np.floor(delta_seconds / self.windowSeconds).astype(int)
            grouped = {int(index): group for index, group in data.groupby("WindowIndex", sort=True)}
        indexes = range(expected_count) if self.fillEmptyWindows else sorted(grouped)
        rows = []
        for index in indexes:
            window_end = origin + ((index + 1) * self.windowFrequency)
            group = grouped.get(index)
            rows.append(self.emptyWindow(target, window_end) if group is None or group.empty else self.aggregateWindow(group, target, window_end))
        return pd.DataFrame(rows)

    def aggregateDirectory(self, cutsDirectory, cutSummaryPath, mode, perAttackDirectory, combinedOutputPath, summaryOutputPath):
        # Reabre os recortes salvos, agrega cada ataque no modo solicitado e persiste o dataset bruto combinado.
        cuts_directory = Path(cutsDirectory)
        cut_summary = pd.read_csv(cutSummaryPath)
        per_attack_directory = Path(perAttackDirectory)
        per_attack_directory.mkdir(parents=True, exist_ok=True)
        frames = []
        summaries = []
        for row in cut_summary.itertuples(index=False):
            attack = str(row.Attack).upper()
            cut_path = cuts_directory / f"{attack}.csv"
            if not cut_path.exists():
                raise FileNotFoundError(f"Cut file was not found: {cut_path}")
            cut = pd.read_csv(cut_path, low_memory=False)
            cut["FlowEnd"] = pd.to_datetime(cut["FlowEnd"], errors="coerce")
            labels = cut[self.labelColumn].astype(str).str.strip().str.upper()
            excluded_other = int((~labels.isin(self.normalLabels | {attack})).sum())
            frame = self.aggregateAttack(cut, attack, pd.Timestamp(row.CutStart), pd.Timestamp(row.CutEnd), mode=mode)
            frame.to_csv(per_attack_directory / f"{attack}_{self.windowSeconds:g}s.csv", index=False)
            frames.append(frame)
            summaries.append({
                "Attack": attack,
                "Mode": mode,
                "ExcludedOtherAttackFlows": excluded_other,
                "WindowCount": int(len(frame)),
                "AttackWindows": int((frame["Label"] == attack).sum()),
                "BenignWindows": int((frame["Label"] == "BENIGN").sum()),
                "NoFlowWindows": int((frame["Label"] == self.noFlowLabel).sum()),
                "TotalFlows": int(frame["FlowCount"].sum()),
                "AttackFlows": int(frame["AttackCount"].sum()),
                "BenignFlows": int(frame["BenignCount"].sum()),
                "OtherAttackFlows": int(frame["OtherAttackCount"].sum()),
                "MeanAttackRatio": float(frame["AttackRatio"].mean(skipna=True)),
                "StdAttackRatio": float(frame["AttackRatio"].std(ddof=0, skipna=True)),
            })
        combined = pd.concat(frames, ignore_index=True).sort_values(["Attack", "Timestamp"], kind="stable").reset_index(drop=True)
        combined_path = Path(combinedOutputPath)
        summary_path = Path(summaryOutputPath)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        return str(combined_path), str(summary_path)
