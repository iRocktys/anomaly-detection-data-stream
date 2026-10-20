from pathlib import Path

import numpy as np
import pandas as pd

from src.Scenario.FlowCSVReader import FlowCSVReader
from src.Scenario.WindowAggregation import CompletedFlowWindowAggregator


class WindowDatasetBuilder:
    # Configura exclusivamente a agregação dos fluxos já disponíveis em arquivos CSV.
    def __init__(self, dataDirectory, outputDirectory="data", chunkSize=100_000):
        self.dataDirectory = Path(dataDirectory)
        self.outputDirectory = Path(outputDirectory)
        self.reader = FlowCSVReader(chunkSize)

    # Valida as durações de janela utilizadas para produzir as bases agregadas.
    def validateWindowDurations(self, windowSeconds):
        durations = list(dict.fromkeys(float(value) for value in windowSeconds))
        if not durations or any(not np.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("Informe durações de janela positivas e finitas.")
        return durations

    # Agrega um benign_flows.csv existente sem executar novamente a extração dos CSVs originais.
    def aggregateBenignFlows(self, benignFlowsPath, windowSeconds=(1.0,)):
        source = Path(benignFlowsPath)
        if not source.is_file():
            raise FileNotFoundError(f"Arquivo benigno não encontrado: {source}. Ajuste o caminho ou execute a extração opcional.")
        aggregators = {value: CompletedFlowWindowAggregator(value) for value in self.validateWindowDurations(windowSeconds)}
        for chunk in self.reader.iterChunks(source):
            if not chunk.Label.eq("BENIGN").all():
                raise ValueError("O arquivo benign_flows.csv deve conter somente fluxos benignos.")
            for aggregator in aggregators.values():
                aggregator.update(chunk)
        rows, paths = [], {}
        for seconds, aggregator in aggregators.items():
            frame = aggregator.toFrame("BENIGN")
            if frame.empty:
                raise ValueError("O arquivo benigno não contém fluxos para agregar.")
            key = f"{seconds:g}s"
            path = self.outputDirectory / f"benign/benign_{key}.csv"
            if path.resolve() == source.resolve():
                raise ValueError("O destino agregado não pode substituir o arquivo de fluxos de origem.")
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            paths[key] = str(path)
            rows.append({"Dataset": "BENIGN", "Duration": key, "Windows": len(frame), "AttackWindows": 0,
                         "BenignWindows": len(frame), "Flows": int(frame.FlowCount.sum()),
                         "OccupiedSeconds": len(frame) * seconds, "Path": str(path)})
        summary = pd.DataFrame(rows)
        summary.to_csv(self.outputDirectory / "benign/window_summary.csv", index=False)
        return {"paths": paths, "summary": summary}

    # Localiza os limites de encerramento dos fluxos maliciosos no arquivo de um ataque.
    def findAttackBounds(self, csvPath, attack):
        first, last = None, None
        for chunk in self.reader.iterChunks(csvPath):
            times = chunk.loc[chunk.Label.eq(attack), "FlowEnd"]
            if not times.empty:
                first = times.min() if first is None else min(first, times.min())
                last = times.max() if last is None else max(last, times.max())
        if first is None:
            raise ValueError(f"O ataque {attack} não foi encontrado em {csvPath}.")
        return first, last

    # Agrega separadamente cada ataque junto com os benignos do respectivo período original.
    def aggregateAttackFlows(self, attackFiles, windowSeconds=(1.0,), cutMinutes=None, cutStartMinutes=None):
        durations = self.validateWindowDurations(windowSeconds)
        if cutMinutes is not None and (not np.isfinite(cutMinutes) or cutMinutes <= 0):
            raise ValueError("cutMinutes deve ser positivo ou None para utilizar o período completo.")
        files = {self.reader.normalizeLabel(a): self.dataDirectory / name for a, name in attackFiles.items()}
        if not files or "BENIGN" in files or len(files) != len(attackFiles):
            raise ValueError("Informe nomes distintos de ataques e seus arquivos.")
        if any(not path.is_file() for path in files.values()):
            raise FileNotFoundError("Um dos arquivos de ataque não foi encontrado.")
        offsets = {self.reader.normalizeLabel(a): float(v) for a, v in (cutStartMinutes or {}).items()}
        if set(offsets) - set(files) or any(not np.isfinite(v) or v < 0 for v in offsets.values()):
            raise ValueError("Informe deslocamentos não negativos apenas para os ataques selecionados.")
        paths, rows = {f"{seconds:g}s": {} for seconds in durations}, []
        for attack, source in files.items():
            first, last = self.findAttackBounds(source, attack)
            start = first.floor("s") + pd.to_timedelta(offsets.get(attack, 0), unit="m")
            end = last.floor("s") + pd.Timedelta(seconds=1) if cutMinutes is None else start + pd.to_timedelta(cutMinutes, unit="m")
            if start > last:
                raise ValueError(f"O recorte de {attack} começa após seu término.")
            aggregators = {value: CompletedFlowWindowAggregator(value) for value in durations}
            for chunk in self.reader.iterChunks(source):
                selected = chunk.loc[chunk.Label.isin({"BENIGN", attack}) & chunk.FlowEnd.ge(start) & chunk.FlowEnd.lt(end)]
                for aggregator in aggregators.values():
                    aggregator.update(selected)
            for seconds, aggregator in aggregators.items():
                frame = aggregator.toFrame(attack)
                if frame.empty or not frame.Label.eq(attack).any():
                    raise ValueError(f"O recorte de {attack} não contém janelas maliciosas.")
                key = f"{seconds:g}s"
                path = self.outputDirectory / f"attacks/{key}/{attack}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                paths[key][attack] = str(path)
                rows.append({"Dataset": attack, "Duration": key, "Windows": len(frame),
                             "AttackWindows": int(frame.Label.eq(attack).sum()), "BenignWindows": int(frame.Label.eq("BENIGN").sum()),
                             "Flows": int(frame.FlowCount.sum()), "CutStart": start, "CutEnd": end,
                             "OriginalSpanSeconds": (frame.Timestamp.max() - frame.Timestamp.min()).total_seconds() + seconds,
                             "OccupiedSeconds": len(frame) * seconds, "Path": str(path)})
        summary = pd.DataFrame(rows)
        summary.to_csv(self.outputDirectory / "attacks/window_summary.csv", index=False)
        return {"paths": paths, "summary": summary}
