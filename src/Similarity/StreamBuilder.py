from numbers import Integral
from pathlib import Path
import os
import tempfile

import pandas as pd

from src.Similarity.LoadData import SimilarityDataLoader


class SimilarityStreamBuilder:
    """Seleciona os primeiros N fluxos encontrados por ataque e ordena só a seleção."""

    def runFromFiles(self, dataDirectory, attackFiles, outputPath, summaryOutputPath,
                     samplesPerAttack=10_000, chunkSize=100_000,
                     timestampColumn="Timestamp", dayFirst=True):
        for name, value in (("samplesPerAttack", samplesPerAttack), ("chunkSize", chunkSize)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        loader = SimilarityDataLoader(chunkSize=chunkSize)
        files = {loader.normalizeLabel(a): Path(dataDirectory) / f for a, f in attackFiles.items()}
        if len(files) != len(attackFiles) or not files or "BENIGN" in files:
            raise ValueError("attackFiles must contain distinct non-benign attack labels.")
        output, summary = Path(outputPath), Path(summaryOutputPath)
        inputs = {p.resolve() for p in files.values()}
        if output.resolve() == summary.resolve() or {output.resolve(), summary.resolve()} & inputs:
            raise ValueError("Output paths must be distinct from inputs and each other.")
        timestamp = loader.cleanColumnName(timestampColumn)
        # A chave temporal é somente interna: não se torna uma feature do CSV.
        sort_key = "__stream_timestamp__"
        columns, blocks, reports = None, [], []
        for attack, path in files.items():
            header = pd.read_csv(path, nrows=0).columns.tolist()
            cleaned = [loader.cleanColumnName(c) for c in header]
            if len(cleaned) != len(set(cleaned)) or not {"Label", timestamp}.issubset(cleaned):
                raise ValueError(f"Invalid columns or missing Label/{timestamp} in {path}.")
            if sort_key in cleaned:
                raise ValueError(f"Reserved internal column in {path}: {sort_key}.")
            if columns is None:
                columns = cleaned
            elif set(columns) != set(cleaned):
                raise ValueError(f"Original flow schemas differ in {path}.")
            parts, remaining, inspected = [], int(samplesPerAttack), 0
            # Interrompe a leitura assim que encontra N registros do rótulo-alvo.
            with pd.read_csv(path, chunksize=chunkSize, low_memory=False) as reader:
                for chunk in reader:
                    data = loader.normalizeColumns(chunk)
                    inspected += len(data)
                    target = data.loc[data["Label"].eq(attack)].head(remaining).copy()
                    if not target.empty:
                        parts.append(target)
                        remaining -= len(target)
                    if remaining == 0:
                        break
            if remaining:
                found = int(samplesPerAttack) - remaining
                raise ValueError(f"Requested {samplesPerAttack} flows per attack; {attack} has {found}. No output written.")
            selected = pd.concat(parts, ignore_index=True)
            times = pd.to_datetime(selected[timestamp], errors="coerce", dayfirst=dayFirst, format="mixed")
            if times.isna().any():
                raise ValueError(f"Invalid {timestamp} in selected flows for {attack}; no output written.")
            selected[sort_key] = times
            blocks.append(selected)
            reports.append({"Attack": attack, "SourceFile": str(path.resolve()),
                            "inspectedRows": inspected, "requestedFlows": int(samplesPerAttack),
                            "selectedFlows": len(selected),
                            "FirstTimestamp": selected[sort_key].min(),
                            "LastTimestamp": selected[sort_key].max()})
        # A cronologia real define a ordem final, inclusive se houver ataques simultâneos.
        stream = pd.concat(blocks, ignore_index=True).sort_values(sort_key, kind="stable")
        stream = stream.reindex(columns=columns)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".stream_", suffix=".csv", dir=output.parent)
        os.close(fd)
        try:
            stream.to_csv(temporary, index=False)
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        pd.DataFrame(reports).to_csv(summary, index=False)
        return {"datasetPath": str(output), "rawFlowDatasetPath": str(output),
                "summaryPath": str(summary), "selectedFlowsPerAttack": int(samplesPerAttack),
                "totalFlows": len(stream), "timestampColumn": timestamp,
                "ordering": "original_timestamp", "selection": "first_encountered", "sampleUnit": "flow"}
