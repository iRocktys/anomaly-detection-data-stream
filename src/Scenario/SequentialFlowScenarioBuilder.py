import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from src.Scenario.FlowCSVReader import FlowCSVReader


class SequentialFlowScenarioBuilder:
    """Monta cenários por instâncias sem agregar as estatísticas originais."""

    def __init__(self, benignPath, dataDirectory=".", attackFiles=None, chunkSize=100_000):
        self.benignPath = Path(benignPath)
        self.dataDirectory = Path(dataDirectory)
        self.reader = FlowCSVReader(chunkSize)
        self.attackFiles = {self.reader.normalizeLabel(k): self.dataDirectory / v for k, v in (attackFiles or {}).items()}

    @staticmethod
    def positiveInteger(value, name, allowZero=False):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if allowZero else 1):
            raise ValueError(f"{name} deve ser inteiro {'não negativo' if allowZero else 'positivo'}.")
        return int(value)

    @staticmethod
    def identities(frame):
        return list(zip(frame.SourceFile.astype(str), frame.SourceRow.astype(int)))

    @staticmethod
    def chronological(frame):
        return frame.sort_values(["FlowEnd", "SourceFile", "SourceRow"], kind="stable").reset_index(drop=True)

    @staticmethod
    def generator(name, seed):
        words = np.frombuffer(hashlib.sha256(name.encode()).digest(), dtype="<u4").tolist()
        return np.random.default_rng(np.random.SeedSequence([seed, *words]))

    @staticmethod
    def keepSmallest(pool, group, capacity, generator):
        # Prioridades aleatórias permitem amostrar em lotes com memória limitada.
        if group.empty or capacity == 0:
            return pool
        group = group.copy()
        group["__Priority"] = generator.random(len(group))
        candidate = pd.concat([pool, group], ignore_index=True) if pool is not None else group
        return candidate.nsmallest(capacity, "__Priority").reset_index(drop=True)

    def loadAttackPool(self, attack, capacity, seed):
        """Guarda o prefixo necessário na ordem original das linhas do CSV."""
        pieces, count, retained = [], 0, 0
        first = last = None
        started = False
        for chunk in self.reader.iterChunks(self.attackFiles[attack]):
            mask = chunk.Label.eq(attack)
            times = chunk.loc[mask, "FlowEnd"]
            count += int(mask.sum())
            if len(times):
                first = times.min() if first is None else min(first, times.min())
                last = times.max() if last is None else max(last, times.max())
            if retained >= capacity:
                continue
            positions = np.flatnonzero(mask.to_numpy())
            if not started:
                if not len(positions):
                    continue
                chunk = chunk.iloc[positions[0]:].copy()
                started = True
            positions = np.flatnonzero(chunk.Label.eq(attack).to_numpy())
            remaining = capacity - retained
            if len(positions) >= remaining:
                chunk = chunk.iloc[:positions[remaining - 1] + 1].copy()
            if not chunk.Label.isin({attack, "BENIGN"}).all():
                raise ValueError(f"{attack}: outro rótulo dentro do trecho sequencial.")
            retained += int(chunk.Label.eq(attack).sum())
            pieces.append(chunk)
        if count == 0:
            raise ValueError(f"Ataque {attack} ausente em {self.attackFiles[attack]}.")
        frame = pd.concat(pieces, ignore_index=True)
        return {"frame": frame, "count": count, "first": first, "last": last}

    def selectBlocks(self, sequence, count, pools, repeat):
        for attack in set(sequence):
            required = count * (1 if repeat else sequence.count(attack))
            available = pools[attack]["count"]
            if required > available:
                raise ValueError(f"{attack}: necessários {required} ataques; disponíveis {available}.")
        cursors, originals, blocks = {}, {}, []
        for index, attack in enumerate(sequence):
            if repeat and attack in originals:
                originalIndex, original = originals[attack]
                blocks.append((attack, original.copy(), originalIndex))
                continue
            frame = pools[attack]["frame"]
            positions = np.flatnonzero(frame.Label.eq(attack).to_numpy())
            offset = cursors.get(attack, 0)
            start = int(positions[offset])
            end = int(positions[offset + count - 1])
            block = frame.iloc[start:end + 1].copy()
            if int(block.Label.eq(attack).sum()) != count:
                raise ValueError("Contagem de ataques inconsistente no trecho selecionado.")
            cursors[attack] = offset + count
            originals[attack] = (index, block)
            blocks.append((attack, block, None))
        return blocks

    def sampleBenign(self, count, seed):
        pool, available, seen = None, 0, set()
        generator = self.generator("BENIGN", seed)
        for chunk in self.reader.iterChunks(self.benignPath):
            if not chunk.Label.eq("BENIGN").all():
                raise ValueError("A referência benigna deve conter somente BENIGN.")
            mask = []
            for key in self.identities(chunk):
                mask.append(key not in seen)
                seen.add(key)
            group = chunk.loc[mask].copy()
            available += len(group)
            pool = self.keepSmallest(pool, group, count, generator)
        if available < count:
            raise ValueError(f"Referência benigna inviável: necessários {count} registros únicos, disponíveis {available}.")
        return self.chronological(pool.drop(columns="__Priority"))

    def writeScenario(self, config, count, blocks, benign, seed, repeat):
        initial, gap = config["initialBenignInstances"], config["betweenBenignInstances"]
        cursor = position = 0
        pieces, summaries = [], []
        for index in range(len(blocks) + 1):
            n = len(benign) - cursor if index == len(blocks) else (initial if index == 0 else gap)
            selections = [("BENIGN", "benign", benign.iloc[cursor:cursor+n].copy(), None)]
            cursor += n
            if index < len(blocks):
                attack, block, repeatOf = blocks[index]
                selections.append((attack, "mixed_attack", block.copy(), repeatOf))
            for target, kind, frame, repeatOf in selections:
                if frame.empty:
                    continue
                attackMask = frame.Label.ne("BENIGN")
                summaries.append({
                    "BlockIndex": len(summaries),
                    "Label": target,
                    "InstanceStart": position,
                    "InstanceEnd": position + len(frame) - 1,
                    "Instances": len(frame),
                    "AttackInstances": int(attackMask.sum()),
                    "BenignInstances": int((~attackMask).sum()),
                })
                frame["InstanceIndex"] = np.arange(position, position + len(frame))
                position += len(frame)
                pieces.append(frame)
        full = pd.concat(pieces, ignore_index=True).drop(columns=["__Priority"], errors="ignore")
        full = full[[c for c in full if c != "Label"] + ["Label"]]
        path = Path(config["outputDirectory"]) / config["scenarioName"] / f"{config['scenarioName']}_{count}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        blocksPath = path.with_name(path.stem + "_blocks.csv")
        full.to_csv(path, index=False)
        pd.DataFrame(summaries).to_csv(blocksPath, index=False)
        return {"Path": str(path), "BlocksPath": str(blocksPath), "Instances": len(full),
                "AttackInstances": int(full.Label.ne("BENIGN").sum())}

    def buildCollection(self, scenarios, blockInstances=(25, 200, 1000), benignSampleCount=15000,
                        initialBenignInstances=3600, betweenBenignInstances=1200,
                        randomSeed=42, repeatAttackBlocks=False, benignReferencePath=None, useSavedBenign=False):
        """Planeja a coleção e usa a mesma referência benigna em todas as variantes."""
        seed = self.positiveInteger(randomSeed, "randomSeed", True)
        countBenign = self.positiveInteger(benignSampleCount, "benignSampleCount")
        if not isinstance(repeatAttackBlocks, bool):
            raise ValueError("repeatAttackBlocks deve ser booleano.")
        sizes = list(dict.fromkeys(self.positiveInteger(v, "blockInstances") for v in blockInstances))
        if not sizes:
            raise ValueError("Informe pelo menos um tamanho de bloco.")
        configs, capacity, targets = [], {}, set()
        for config in scenarios:
            config = dict(config)
            name = str(config["scenarioName"])
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
                raise ValueError("scenarioName inválido.")
            config["sequence"] = [self.reader.normalizeLabel(v) for v in config["sequence"]]
            if not config["sequence"] or set(config["sequence"]) - set(self.attackFiles) or "BENIGN" in config["sequence"]:
                raise ValueError("Informe ataques existentes na configuração attackFiles.")
            for key, default in (("initialBenignInstances", initialBenignInstances), ("betweenBenignInstances", betweenBenignInstances)):
                config[key] = self.positiveInteger(config.get(key, default), key, True)
            identity = str((Path(config["outputDirectory"]) / name).resolve())
            if identity in targets:
                raise ValueError("Dois cenários possuem o mesmo diretório de saída.")
            targets.add(identity)
            configs.append(config)
            for attack in set(config["sequence"]):
                occurrences = 1 if repeatAttackBlocks else config["sequence"].count(attack)
                demand = max(sizes) * occurrences
                capacity[attack] = max(capacity.get(attack, 0), demand)
        if not configs:
            raise ValueError("Informe pelo menos um cenário.")
        pools, errors, availability = {}, {}, []
        for attack in sorted(capacity):
            try:
                pools[attack] = self.loadAttackPool(attack, capacity[attack], seed)
                data = pools[attack]
                availability.append({"Attack": attack, "AvailableAttackFlows": data["count"],
                                     "OriginalStart": data["first"], "OriginalEnd": data["last"]})
            except ValueError as error:
                errors[attack] = str(error)
                availability.append({"Attack": attack, "Reason": str(error)})
        plans, rows = [], []
        for config in configs:
            for count in sizes:
                row = {"Scenario": config["scenarioName"], "Family": Path(config["outputDirectory"]).name,
                       "BlockInstances": count, "RepeatAttackBlocks": repeatAttackBlocks,
                       "Status": "Inviável", "Reason": "", "Path": None, "BlocksPath": None}
                try:
                    for attack in config["sequence"]:
                        if attack in errors:
                            raise ValueError(errors[attack])
                    required = config["initialBenignInstances"] + (len(config["sequence"])-1) * config["betweenBenignInstances"]
                    if required > countBenign:
                        raise ValueError(f"Início e intervalos exigem {required} benignos; referência configurada: {countBenign}.")
                    blocks = self.selectBlocks(config["sequence"], count, pools, repeatAttackBlocks)
                    plans.append((config, count, blocks, row))
                except ValueError as error:
                    row["Reason"] = str(error)
                rows.append(row)
        if plans:
            try:
                if useSavedBenign:
                    benign = pd.concat(list(self.reader.iterChunks(self.benignPath)), ignore_index=True)
                    if len(benign) != countBenign or not benign.Label.eq("BENIGN").all():
                        raise ValueError("Referência salva deve conter exatamente a quantidade configurada de BENIGN.")
                    identities = self.identities(benign)
                    if len(set(identities)) != len(identities):
                        raise ValueError("Referência salva contém registros duplicados.")
                    benign = self.chronological(benign)
                else:
                    benign = self.sampleBenign(countBenign, seed)
            except ValueError as error:
                for _, _, _, row in plans:
                    row["Reason"] = str(error)
            else:
                if benignReferencePath is not None:
                    path = Path(benignReferencePath)
                    if path.resolve() == self.benignPath.resolve():
                        raise ValueError("A referência amostrada não pode sobrescrever benign_flows.csv.")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    benign.to_csv(path, index=False)
                for config, count, blocks, row in plans:
                    row.update(self.writeScenario(config, count, blocks, benign, seed, repeatAttackBlocks))
                    row["Status"] = "Gerado"
        return {"availability": pd.DataFrame(availability), "results": pd.DataFrame(rows)}


    def createBenignReference(self, outputPath, sampleCount=15000, randomSeed=42):
        """Amostra somente o CSV benigno existente, sem ler arquivos de ataque."""
        count = self.positiveInteger(sampleCount, "sampleCount")
        seed = self.positiveInteger(randomSeed, "randomSeed", True)
        path = Path(outputPath)
        if path.resolve() == self.benignPath.resolve():
            raise ValueError("A saída não pode sobrescrever a base benigna original.")
        benign = self.sampleBenign(count, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        benign.to_csv(path, index=False)
        return pd.DataFrame([{
            "Dataset": "BENIGN", "Path": str(path), "Instances": len(benign),
            "SamplingSeed": seed,
        }])

    def buildVariants(self, scenarioName, sequence, outputDirectory="data/scenario",
                      blockInstances=(25, 200, 1000), initialBenignInstances=3600,
                      betweenBenignInstances=1200, randomSeed=42, repeatAttackBlocks=False,
                      benignSampleCount=15000):
        """Lê a referência pronta e gera somente as variantes deste cenário."""
        return self.buildCollection(
            scenarios=[{"scenarioName": scenarioName, "sequence": sequence,
                        "outputDirectory": outputDirectory}],
            blockInstances=blockInstances, benignSampleCount=benignSampleCount,
            initialBenignInstances=initialBenignInstances,
            betweenBenignInstances=betweenBenignInstances, randomSeed=randomSeed,
            repeatAttackBlocks=repeatAttackBlocks, useSavedBenign=True,
        )


    @classmethod
    def prepareBenignReference(cls, dataDirectory, excludedFiles, filteredPath,
                              outputPath, sampleCount=15000, randomSeed=42,
                              chunkSize=100_000, overwrite=False):
        """Extrai BENIGN somente dos arquivos permitidos e sorteia a referência."""
        from src.Scenario.BenignExtractor import BenignFlowExtractor

        directory = Path(dataDirectory)
        excluded = {str(name).casefold() for name in excludedFiles}
        files = sorted(path for path in directory.iterdir()
                       if path.is_file() and path.suffix.lower() == ".csv")
        unknown = excluded - {path.name.casefold() for path in files}
        if unknown:
            raise ValueError(f"Arquivos excluídos não encontrados: {sorted(unknown)}")
        sources = [path.name for path in files if path.name.casefold() not in excluded]
        if not sources:
            raise ValueError("Nenhum CSV permitido para extração do BENIGN.")
        if Path(filteredPath).resolve() == Path(outputPath).resolve():
            raise ValueError("Use arquivos diferentes para a extração e a amostra.")
        if Path(outputPath).resolve() in {path.resolve() for path in files}:
            raise ValueError("A amostra não pode sobrescrever um CSV de origem.")
        if Path(outputPath).exists() and not overwrite:
            raise FileExistsError(f"{outputPath} já existe; informe overwrite=True.")
        cls.positiveInteger(sampleCount, "sampleCount")
        cls.positiveInteger(randomSeed, "randomSeed", True)
        extraction = BenignFlowExtractor(directory, chunkSize).extract(
            outputPath=filteredPath, sourceFiles=sources, overwrite=overwrite)
        result = cls(filteredPath, chunkSize=chunkSize).createBenignReference(
            outputPath, sampleCount, randomSeed)
        result["EligibleBenignInstances"] = extraction["flows"]
        result["SourceFiles"] = len(sources)
        return result
