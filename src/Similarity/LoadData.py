import re
from pathlib import Path

import pandas as pd


class SimilarityDataLoader:
    def __init__(self, timestampColumn="Timestamp", durationColumn="Flow_Duration", labelColumn="Label", normalLabels=None, chunkSize=100_000):
        # Inicializa a leitura temporal dos CSVs originais do CICDDoS2019.
        self.timestampColumn = str(timestampColumn)
        self.durationColumn = str(durationColumn)
        self.labelColumn = str(labelColumn)
        self.normalLabels = {self.normalizeLabel(value) for value in (normalLabels or ["BENIGN", "NORMAL", "0"])}
        self.chunkSize = int(chunkSize)

    @staticmethod
    def cleanColumnName(name):
        # Padroniza nomes de colunas para o formato utilizado pelo projeto.
        value = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip())
        return value.strip("_")

    @staticmethod
    def normalizeLabel(value):
        # Normaliza os rótulos diretamente na coluna Label para nomes curtos de ataque.
        label = str(value).strip().upper().replace(" ", "_")
        for prefix in ("DRDOS_", "DRDOS-", "DDOS_", "DDOS-"):
            if label.startswith(prefix):
                label = label[len(prefix):]
        aliases = {
            "UDP_LAG": "UDP-LAG",
            "UDPLAG": "UDP-LAG",
            "WEB_DDOS": "WEBDDOS",
            "NORMAL": "BENIGN",
            "0": "BENIGN",
        }
        return aliases.get(label, label)

    def normalizeColumns(self, frame):
        # Padroniza os nomes das colunas e aplica a nova rotulagem diretamente em Label.
        data = frame.copy()
        data.columns = [self.cleanColumnName(column) for column in data.columns]
        label_column = self.cleanColumnName(self.labelColumn)
        if label_column not in data.columns:
            raise KeyError(f"Missing label column: {label_column}.")
        data[label_column] = data[label_column].map(self.normalizeLabel)
        return data

    def prepareChunk(self, frame):
        # Cria FlowStart e FlowEnd para representar o início e o término de cada fluxo.
        data = self.normalizeColumns(frame)
        timestamp_column = self.cleanColumnName(self.timestampColumn)
        duration_column = self.cleanColumnName(self.durationColumn)
        if timestamp_column not in data.columns:
            raise KeyError(f"Missing timestamp column: {timestamp_column}.")
        if duration_column not in data.columns:
            raise KeyError(f"Missing duration column: {duration_column}.")
        data["FlowStart"] = pd.to_datetime(data[timestamp_column], errors="coerce", dayfirst=True, format="mixed")
        duration = pd.to_numeric(data[duration_column], errors="coerce").fillna(0.0).clip(lower=0.0)
        data["FlowEnd"] = data["FlowStart"] + pd.to_timedelta(duration.to_numpy(dtype=float), unit="us")
        return data.dropna(subset=["FlowStart", "FlowEnd"]).copy()

    def findAttackStart(self, csvPath, attackLabel):
        # Localiza o primeiro término de fluxo do ataque solicitado dentro do arquivo original.
        target = self.normalizeLabel(attackLabel)
        label_column = self.cleanColumnName(self.labelColumn)
        starts = []
        for chunk in pd.read_csv(csvPath, chunksize=self.chunkSize, low_memory=False):
            prepared = self.prepareChunk(chunk)
            matches = prepared.loc[prepared[label_column] == target, "FlowEnd"]
            if not matches.empty:
                starts.append(matches.min())
        if starts:
            return min(starts)
        raise ValueError(f"Attack '{target}' was not found in {csvPath}.")

    def loadTemporalCut(self, csvPath, attackLabel, durationMinutes=5.0):
        # Carrega um intervalo contínuo mantendo tráfego benigno e outros rótulos presentes no mesmo período.
        csv_path = Path(csvPath)
        target = self.normalizeLabel(attackLabel)
        start = self.findAttackStart(csv_path, target)
        end = start + pd.to_timedelta(float(durationMinutes), unit="m")
        parts = []
        for chunk in pd.read_csv(csv_path, chunksize=self.chunkSize, low_memory=False):
            prepared = self.prepareChunk(chunk)
            selected = prepared.loc[(prepared["FlowEnd"] >= start) & (prepared["FlowEnd"] < end)].copy()
            if not selected.empty:
                parts.append(selected)
        if not parts:
            raise ValueError(f"No flows were found in the selected interval for attack '{target}'.")
        cut = pd.concat(parts, ignore_index=True).sort_values("FlowEnd", kind="stable").reset_index(drop=True)
        label_column = self.cleanColumnName(self.labelColumn)
        attack_count = int((cut[label_column] == target).sum())
        if attack_count == 0:
            raise ValueError(f"The selected interval contains no '{target}' flows.")
        summary = {
            "Attack": target,
            "SourceFile": str(csv_path),
            "CutStart": start,
            "CutEnd": end,
            "DurationMinutes": float(durationMinutes),
            "FlowCount": int(len(cut)),
            "AttackFlowCount": attack_count,
            "BenignFlowCount": int(cut[label_column].isin(self.normalLabels).sum()),
            "OtherAttackFlowCount": int((~cut[label_column].isin(self.normalLabels | {target})).sum()),
        }
        return cut, summary

    def createCuts(self, dataDirectory, attackFiles, durationMinutes, outputDirectory, summaryPath):
        # Cria e salva os recortes temporais de todos os ataques para reutilização nas agregações seguintes.
        data_directory = Path(dataDirectory)
        output_directory = Path(outputDirectory)
        output_directory.mkdir(parents=True, exist_ok=True)
        summaries = []
        for attack, filename in attackFiles.items():
            path = data_directory / filename
            if not path.exists():
                raise FileNotFoundError(f"Attack file was not found: {path}")
            cut, summary = self.loadTemporalCut(path, attack, durationMinutes)
            attack_name = self.normalizeLabel(attack)
            cut.to_csv(output_directory / f"{attack_name}.csv", index=False)
            summaries.append(summary)
        summary_frame = pd.DataFrame(summaries)
        summary_path = Path(summaryPath)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_frame.to_csv(summary_path, index=False)
        return str(summary_path)
