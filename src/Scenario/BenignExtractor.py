from pathlib import Path

from src.Scenario.FlowCSVReader import FlowCSVReader


class BenignFlowExtractor:
    # Configura a leitura dos CSVs originais para a extração independente do tráfego normal.
    def __init__(self, dataDirectory, chunkSize=100_000):
        self.dataDirectory = Path(dataDirectory)
        self.reader = FlowCSVReader(chunkSize)

    # Salva somente fluxos benignos sem executar agregação nem sobrescrever arquivos existentes por padrão.
    def extract(self, outputPath="data/benign/benign_flows.csv", sourceFiles=None, overwrite=False):
        path = Path(outputPath)
        if path.exists() and not overwrite:
            raise FileExistsError(f"O arquivo {path} já existe; reutilize-o ou informe overwrite=True.")
        sources = [p for p in self.dataDirectory.iterdir() if p.is_file() and p.suffix.lower() == ".csv"] if sourceFiles is None else [self.dataDirectory / name for name in sourceFiles]
        sources = sorted({source.resolve() for source in sources})
        if not sources or any(not source.is_file() for source in sources):
            raise FileNotFoundError("Informe arquivos CSV de origem existentes.")
        if path.resolve() in sources:
            raise ValueError("O arquivo de destino não pode fazer parte dos arquivos de origem.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.partial")
        columns, count = None, 0
        for source in sources:
            for chunk in self.reader.iterChunks(source):
                benign = chunk.loc[chunk.Label.eq("BENIGN")].drop(columns=["WindowStart", "WindowEnd", "WindowSeconds", "Attack"], errors="ignore")
                if benign.empty:
                    continue
                selected = [c for c in benign if c != "Label"] + ["Label"]
                columns = selected if columns is None else columns
                if set(columns) != set(benign.columns):
                    raise ValueError("Os arquivos de origem precisam apresentar as mesmas colunas.")
                benign[columns].to_csv(temporary, index=False, header=count == 0, mode="w" if count == 0 else "a")
                count += len(benign)
        if not count:
            raise ValueError("Não foram encontrados fluxos benignos.")
        temporary.replace(path)
        return {"datasetPath": str(path), "flows": count}
