import re
from pathlib import Path

import numpy as np
import pandas as pd


class FlowCSVReader:
    # Configura a leitura dos fluxos completos em lotes de tamanho limitado.
    def __init__(self, chunkSize=100_000):
        if isinstance(chunkSize, bool) or int(chunkSize) != chunkSize or chunkSize < 1:
            raise ValueError("chunkSize deve ser um inteiro positivo.")
        self.chunkSize = int(chunkSize)

    @staticmethod
    # Padroniza os nomes das colunas dos arquivos originais.
    def normalizeColumnName(value):
        return re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")

    @staticmethod
    # Converte os rótulos originais para os nomes utilizados nos cenários.
    def normalizeLabel(value):
        if pd.isna(value) or str(value).strip().upper() in {"", "NAN", "NONE", "NULL", "<NA>"}:
            raise ValueError("Existe um fluxo sem rótulo válido.")
        label = str(value).strip().upper().replace(" ", "_")
        for prefix in ("DRDOS_", "DRDOS-", "DDOS_", "DDOS-"):
            if label.startswith(prefix):
                label = label[len(prefix):]
                break
        return {"NORMAL": "BENIGN", "0": "BENIGN", "UDP_LAG": "UDP-LAG",
                "UDPLAG": "UDP-LAG", "WEB_DDOS": "WEBDDOS"}.get(label, label)

    # Prepara os horários de início e encerramento sem repartir as estatísticas dos fluxos.
    def prepareChunk(self, frame, sourceFile, firstRow):
        data = frame.copy()
        data.columns = [self.normalizeColumnName(column) for column in data]
        if not data.columns.is_unique:
            raise ValueError(f"Colunas duplicadas após normalização: {sourceFile}")
        missing = {"Timestamp", "Flow_Duration", "Label"} - set(data.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes em {sourceFile}: {sorted(missing)}")
        data["Label"] = data.Label.map(self.normalizeLabel)
        data["FlowStart"] = pd.to_datetime(data.Timestamp, format="mixed", dayfirst=True, errors="coerce")
        duration = pd.to_numeric(data.Flow_Duration, errors="coerce").replace([np.inf, -np.inf], np.nan)
        data["FlowEnd"] = data.FlowStart + pd.to_timedelta(duration.fillna(0).clip(lower=0), unit="us")
        if data.FlowStart.isna().any() or data.FlowEnd.isna().any():
            raise ValueError(f"Horário inválido em {sourceFile}; corrija o CSV antes de agregar.")
        if ("SourceFile" in data) != ("SourceRow" in data):
            raise ValueError("SourceFile e SourceRow precisam estar presentes juntos.")
        if "SourceFile" not in data:
            data["SourceFile"] = str(Path(sourceFile).resolve())
            data["SourceRow"] = np.arange(firstRow, firstRow + len(data))
        elif data.SourceFile.isna().any() or data.SourceRow.isna().any():
            raise ValueError("Identificação original incompleta.")
        return data

    # Percorre um CSV preservando a identificação e a ordem original de suas linhas.
    def iterChunks(self, csvPath):
        firstRow = 0
        for chunk in pd.read_csv(csvPath, chunksize=self.chunkSize, low_memory=False):
            prepared = self.prepareChunk(chunk, csvPath, firstRow)
            firstRow += len(chunk)
            yield prepared
