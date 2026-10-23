import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from src.Scenario.FlowCSVReader import FlowCSVReader


class WindowScenarioBuilder:
    metadataColumns = {"Timestamp", "WindowStart", "WindowEnd", "WindowSeconds", "Attack", "Label",
                       "AttackCount", "BenignCount", "OtherAttackCount", "AttackRatio"}

    # Configura as bases janeladas utilizadas na montagem das sequências.
    def __init__(self, benignPath, attacksDirectory, windowSeconds=1.0):
        self.benignPath = Path(benignPath)
        self.attacksDirectory = Path(attacksDirectory)
        self.windowSeconds = float(windowSeconds)
        if not np.isfinite(self.windowSeconds) or self.windowSeconds <= 0:
            raise ValueError("windowSeconds deve ser positivo e finito.")

    # Converte uma duração reconstruída em quantidade inteira de janelas ocupadas.
    def durationToWindows(self, seconds):
        value = float(seconds) / self.windowSeconds
        if not np.isfinite(value) or value < 0 or not np.isclose(value, round(value), rtol=0, atol=1e-8):
            raise ValueError("As durações devem ser múltiplos não negativos da janela.")
        return int(round(value))

    # Carrega uma base temporal preservando todas as suas janelas benignas internas.
    def loadWindows(self, path, target):
        frame = pd.read_csv(path, low_memory=False)
        required = {"Timestamp", "Label", "FlowCount"}
        if not required.issubset(frame):
            raise ValueError(f"Base janelada incompleta: {path}")
        if frame.empty:
            raise ValueError(f"Base vazia ou janela incompatível: {path}")
        if not frame.Label.isin({"BENIGN", target}).all() or not frame.FlowCount.gt(0).all():
            raise ValueError(f"Rótulos ou janelas vazias inválidos em {path}")
        frame["Timestamp"] = pd.to_datetime(frame.Timestamp, format="mixed", errors="raise")
        if frame.Timestamp.isna().any() or frame.Timestamp.duplicated().any():
            raise ValueError(f"Janelas duplicadas em {path}")
        frequency = pd.to_timedelta(self.windowSeconds, unit="s")
        if (frame.Timestamp.astype("int64") % frequency.value).ne(0).any():
            raise ValueError(f"Timestamps incompatíveis com a duração de janela configurada: {path}")
        frame = frame.drop(columns=["WindowStart", "WindowEnd", "WindowSeconds", "Attack"], errors="ignore")
        return frame.sort_values("Timestamp", kind="stable").reset_index(drop=True)

    # Divide a duração original da captura em três estratos usando o início original das janelas.
    def buildTemporalStrata(self, frame):
        frequency = pd.to_timedelta(self.windowSeconds, unit="s").value
        starts = frame.Timestamp.astype("int64").to_numpy() - frequency
        origin, end = int(starts.min()), int(frame.Timestamp.astype("int64").max())
        span = end - origin
        boundaries = np.array([origin + span // 3, origin + 2 * span // 3], dtype=np.int64)
        return np.searchsorted(boundaries, starts, side="right")

    # Embaralha cada estrato com semente estável independente da ordem dos outros ataques.
    def createStratumPermutation(self, attack, indices, stratum, randomSeed):
        digest = hashlib.sha256(attack.encode("utf-8")).digest()
        attackSeed = np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()
        generator = np.random.default_rng(np.random.SeedSequence([int(randomSeed), *attackSeed, stratum]))
        return generator.permutation(indices)

    def inspectAvailability(self, sequence):
        # Consulta as bases agregadas e resume os três estratos antes da montagem.
        names = [FlowCSVReader.normalizeLabel(name) for name in sequence]
        rows = []
        for attack in dict.fromkeys(names):
            frame = self.loadWindows(self.attacksDirectory / f"{attack}.csv", attack)
            strata = self.buildTemporalStrata(frame)
            rows.append({"Attack": attack, "Occurrences": names.count(attack),
                         "OriginalSpanSeconds": (frame.Timestamp.max() - frame.Timestamp.min()).total_seconds() + self.windowSeconds,
                         "AvailableWindows": len(frame),
                         "BeginningWindows": int((strata == 0).sum()),
                         "MiddleWindows": int((strata == 1).sum()),
                         "EndWindows": int((strata == 2).sum()),
                         "InternalBenignWindows": int(frame.Label.eq("BENIGN").sum())})
        return pd.DataFrame(rows)

    def buildVariants(self, scenarioName, sequence, outputDirectory="data/scenario",
                      blockSeconds=(60, 120, 180), initialBenignSeconds=2000,
                      betweenBenignSeconds=120, randomSeed=42, repeatAttackBlocks=False):
        # Executa variantes independentes e mantém o notebook apenas como configuração.
        name = str(scenarioName).strip()
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("scenarioName deve conter somente letras, números, '_' ou '-'.")
        durations = list(dict.fromkeys(float(value) for value in blockSeconds))
        if not durations:
            raise ValueError("Informe pelo menos uma duração por bloco.")
        for value in durations:
            if self.durationToWindows(value) < 3:
                raise ValueError("Cada bloco precisa conter pelo menos três janelas.")
        availability = self.inspectAvailability(sequence)
        results, summaries = [], {}
        for seconds in durations:
            path = Path(outputDirectory) / name / f"{name}_{self.windowSeconds:g}s_{seconds:g}s.csv"
            row = {"Scenario": name, "WindowSeconds": self.windowSeconds, "BlockSeconds": seconds}
            try:
                result = self.build(sequence=sequence, outputPath=path, blockSeconds=seconds,
                                    initialBenignSeconds=initialBenignSeconds,
                                    betweenBenignSeconds=betweenBenignSeconds, randomSeed=randomSeed,
                                    benignStartSeconds=0, allowBenignReuse=False, repeatAttackBlocks=repeatAttackBlocks)
            except ValueError as error:
                row.update(Status="Inviável", Reason=str(error), Path=None, BlocksPath=None)
            else:
                row.update(Status="Gerado", Reason="", Path=result["datasetPath"],
                           BlocksPath=result["blocksPath"], Windows=result["windows"],
                           AttackWindows=result["attackWindows"])
                summaries[seconds] = result["summary"]
            results.append(row)
        return {"availability": availability, "results": pd.DataFrame(results), "blocks": summaries}

    # Planeja todas as ocorrências e distribui janelas distintas de início, meio e fim entre os blocos.
    def selectAttackBlocks(self, sequence, blockSeconds, randomSeed=42, repeatAttackBlocks=False):
        if not isinstance(repeatAttackBlocks, bool):
            raise ValueError("repeatAttackBlocks deve ser booleano.")
        if isinstance(randomSeed, bool) or not isinstance(randomSeed, Integral) or randomSeed < 0:
            raise ValueError("randomSeed deve ser um inteiro não negativo.")
        names = [FlowCSVReader.normalizeLabel(a) for a in sequence]
        durations = list(blockSeconds) if isinstance(blockSeconds, (list, tuple)) else [blockSeconds] * len(names)
        if not names or len(durations) != len(names) or "BENIGN" in names:
            raise ValueError("Informe uma sequência de ataques e uma duração por ocorrência.")
        quotas = []
        for seconds in durations:
            count = self.durationToWindows(seconds)
            if count < 3:
                raise ValueError("Cada bloco precisa conter pelo menos três janelas para representar início, meio e fim.")
            base, remainder = divmod(count, 3)
            quotas.append(np.array([base + int(index < remainder) for index in range(3)], dtype=int))
        prepared, cursors = {}, {}
        for attack in sorted(set(names)):
            frame = self.loadWindows(self.attacksDirectory / f"{attack}.csv", attack)
            strata = self.buildTemporalStrata(frame)
            attackQuotas = [quota for name, quota in zip(names, quotas) if name == attack]
            if repeatAttackBlocks and any(not np.array_equal(q, attackQuotas[0]) for q in attackQuotas):
                raise ValueError("Repetição exige o mesmo tamanho nas ocorrências de cada ataque.")
            requested = attackQuotas[0] if repeatAttackBlocks else np.sum(attackQuotas, axis=0)
            pools = []
            for stratum, title in enumerate(("início", "meio", "fim")):
                indices = np.flatnonzero(strata == stratum)
                if requested[stratum] > len(indices):
                    raise ValueError(f"{attack}, estrato {title}: necessárias {requested[stratum]} janelas distintas, disponíveis {len(indices)}. Reduza os blocos ou a quantidade de ocorrências.")
                pools.append(self.createStratumPermutation(attack, indices, stratum, randomSeed))
            prepared[attack] = (frame, strata, pools)
            cursors[attack] = np.zeros(3, dtype=int)
        blocks = []
        for attack, quota in zip(names, quotas):
            frame, strata, pools = prepared[attack]
            selected = []
            for stratum in range(3):
                first = 0 if repeatAttackBlocks else cursors[attack][stratum]
                last = first + quota[stratum]
                selected.append(pools[stratum][first:last])
                cursors[attack][stratum] = last
            sourceRows = np.sort(np.concatenate(selected))
            block = frame.iloc[sourceRows].copy()
            if not block.Label.eq(attack).any():
                raise ValueError(f"O bloco sorteado de {attack} contém somente benignos. Revise a base ou escolha outra semente.")
            blocks.append((attack, sourceRows, block, strata[sourceRows]))
        return blocks

    # Monta a sequência com intervalos configuráveis e utiliza todo o restante benigno após o último ataque.
    def build(self, sequence, outputPath, blockSeconds=120, initialBenignSeconds=1200,
              betweenBenignSeconds=120, randomSeed=42,
              attackPositionsSeconds=None, benignStartSeconds=0, allowBenignReuse=True,
              startTimestamp="2021-01-01 00:00:00", repeatAttackBlocks=False):
        blocks = self.selectAttackBlocks(sequence, blockSeconds, randomSeed, repeatAttackBlocks)
        benign = self.loadWindows(self.benignPath, "BENIGN")
        gaps = [self.durationToWindows(initialBenignSeconds)]
        gaps.extend([self.durationToWindows(betweenBenignSeconds)] * (len(blocks) - 1))
        if attackPositionsSeconds is not None:
            if len(attackPositionsSeconds) != len(blocks):
                raise ValueError("Informe uma posição para cada bloco.")
            cursor = 0
            for index, position in enumerate(attackPositionsSeconds):
                start = self.durationToWindows(position)
                if start < cursor:
                    raise ValueError("As posições dos blocos não podem se sobrepor.")
                gaps[index] = start - cursor
                cursor = start + len(blocks[index][2])
        benignCursor = self.durationToWindows(benignStartSeconds)
        pieces, summaries, position = [], [], 0
        firstAttackBlock = {}
        for index in range(len(blocks) + 1):
            count = max(len(benign) - benignCursor, 0) if index == len(blocks) else gaps[index]
            if not allowBenignReuse and benignCursor + count > len(benign):
                raise ValueError("Não há janelas benignas suficientes sem reutilização.")
            indices = np.arange(benignCursor, benignCursor + count) % len(benign)
            selections = [("BENIGN", "benign", index, indices, benign.iloc[indices].copy(), np.full(count, -1, dtype=int))]
            benignCursor += count
            if index < len(blocks):
                attack, sourceRows, block, strata = blocks[index]
                selections.append((attack, "mixed_attack", index, sourceRows, block, strata))
            for target, kind, blockIndex, sourceRows, frame, strata in selections:
                if frame.empty:
                    continue
                origin = pd.Timestamp(startTimestamp)
                frequency = pd.to_timedelta(self.windowSeconds, unit="s")
                repeatOf = firstAttackBlock.get(target) if repeatAttackBlocks and kind == "mixed_attack" else None
                if kind == "mixed_attack":
                    firstAttackBlock.setdefault(target, blockIndex)
                summaries.append({
                    "BlockIndex": blockIndex, "ScenarioPart": kind, "TargetAttack": target,
                    "RepeatAttackBlocks": repeatAttackBlocks, "RepeatOfBlock": repeatOf,
                    "Windows": len(frame), "Start": origin + position * frequency,
                    "End": origin + (position + len(frame) - 1) * frequency,
                    "InstanceStart": position, "InstanceEnd": position + len(frame) - 1,
                    "AttackWindows": int(frame.Label.ne("BENIGN").sum()),
                    "BenignWindows": int(frame.Label.eq("BENIGN").sum()),
                    "BeginningWindows": int((strata == 0).sum()),
                    "MiddleWindows": int((strata == 1).sum()),
                    "EndWindows": int((strata == 2).sum()),
                    "OccupiedSeconds": len(frame) * self.windowSeconds,
                    "WindowSeconds": self.windowSeconds, "SamplingSeed": int(randomSeed),
                    "InitialBenignSeconds": initialBenignSeconds,
                    "BetweenBenignSeconds": betweenBenignSeconds,
                    "BenignStartSeconds": benignStartSeconds, "AllowBenignReuse": allowBenignReuse,
                    "SourceDataset": str(self.benignPath if kind == "benign" else self.attacksDirectory / f"{target}.csv"),
                    "OriginalStart": frame.Timestamp.min(), "OriginalEnd": frame.Timestamp.max(),
                })
                position += len(frame)
                pieces.append(frame)
        full = pd.concat(pieces, ignore_index=True)
        features = [column for column in full if column not in self.metadataColumns]
        dataset = full[["Timestamp"] + features + ["Label"]].copy()
        dataset["Timestamp"] = pd.date_range(startTimestamp, periods=len(dataset), freq=pd.to_timedelta(self.windowSeconds, unit="s"))
        path = Path(outputPath)
        path.parent.mkdir(parents=True, exist_ok=True)
        blocksPath = path.with_name(f"{path.stem}_blocks.csv")
        summary = pd.DataFrame(summaries)
        dataset.to_csv(path, index=False)
        summary.to_csv(blocksPath, index=False)
        return {"datasetPath": str(path), "blocksPath": str(blocksPath), "summary": summary,
                "windows": len(dataset), "attackWindows": int(dataset.Label.ne("BENIGN").sum())}


class FlowScenarioBuilder:
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
    def strata(times, first, last):
        # Três intervalos de igual duração; empates de horário permanecem no mesmo estrato.
        origin = pd.Timestamp(first).value
        span = pd.Timestamp(last).value - origin + 1
        boundaries = [origin + span // 3, origin + 2 * span // 3]
        return np.searchsorted(boundaries, times.astype("int64").to_numpy(), side="right")

    @staticmethod
    def quotas(count):
        base, remainder = divmod(count, 3)
        return np.array([base + int(i < remainder) for i in range(3)], dtype=int)

    @staticmethod
    def generators(name, seed):
        words = np.frombuffer(hashlib.sha256(name.encode()).digest(), dtype="<u4").tolist()
        return [np.random.default_rng(np.random.SeedSequence([seed, *words, i])) for i in range(3)]

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
        path = self.attackFiles[attack]
        first = last = None
        for chunk in self.reader.iterChunks(path):
            times = chunk.loc[chunk.Label.eq(attack), "FlowEnd"]
            if len(times):
                first = times.min() if first is None else min(first, times.min())
                last = times.max() if last is None else max(last, times.max())
        if first is None:
            raise ValueError(f"Ataque {attack} ausente em {path}.")
        pools, benignPieces, counts = [None] * 3, [], np.zeros(3, dtype=int)
        generators = self.generators(attack, seed)
        for chunk in self.reader.iterChunks(path):
            selected = chunk.loc[chunk.Label.eq(attack)].copy()
            selected["__Stratum"] = self.strata(selected.FlowEnd, first, last)
            for i in range(3):
                group = selected.loc[selected["__Stratum"].eq(i)]
                counts[i] += len(group)
                pools[i] = self.keepSmallest(pools[i], group, int(capacity[i]), generators[i])
            benign = chunk.loc[chunk.Label.eq("BENIGN") & chunk.FlowEnd.between(first, last)].copy()
            if len(benign):
                benign["__Stratum"] = self.strata(benign.FlowEnd, first, last)
                benignPieces.append(benign)
        pools = [p.drop(columns="__Priority") if p is not None else pd.DataFrame() for p in pools]
        benign = pd.concat(benignPieces, ignore_index=True) if benignPieces else pd.DataFrame()
        return {"pools": pools, "benign": benign, "counts": counts, "first": first, "last": last}

    def selectBlocks(self, sequence, count, pools, repeat):
        quota = self.quotas(count)
        for attack in set(sequence):
            required = quota * (1 if repeat else sequence.count(attack))
            available = pools[attack]["counts"]
            for i, title in enumerate(("início", "meio", "fim")):
                if required[i] > available[i]:
                    raise ValueError(f"{attack}, {title}: necessários {required[i]} fluxos de ataque distintos; disponíveis {available[i]}.")
        cursors, originals, usedBenign, blocks = {}, {}, set(), []
        for index, attack in enumerate(sequence):
            if repeat and attack in originals:
                firstIndex, original = originals[attack]
                blocks.append((attack, original.copy(), firstIndex))
                continue
            cursor = cursors.setdefault(attack, np.zeros(3, dtype=int))
            parts = []
            for i in range(3):
                group = pools[attack]["pools"][i].iloc[cursor[i]:cursor[i] + quota[i]].copy()
                cursor[i] += quota[i]
                parts.append(group)
                benign = pools[attack]["benign"]
                if not benign.empty:
                    internal = benign.loc[benign["__Stratum"].eq(i) & benign.FlowEnd.between(group.FlowEnd.min(), group.FlowEnd.max())].copy()
                    internal = internal.loc[[key not in usedBenign for key in self.identities(internal)]]
                    usedBenign.update(self.identities(internal))
                    parts.append(internal)
            block = self.chronological(pd.concat(parts, ignore_index=True))
            originals[attack] = (index, block)
            blocks.append((attack, block, None))
        return blocks

    def sampleBenign(self, count, seed):
        first = last = None
        for chunk in self.reader.iterChunks(self.benignPath):
            if not chunk.Label.eq("BENIGN").all():
                raise ValueError("A referência benigna deve conter somente BENIGN.")
            first = chunk.FlowEnd.min() if first is None else min(first, chunk.FlowEnd.min())
            last = chunk.FlowEnd.max() if last is None else max(last, chunk.FlowEnd.max())
        if first is None:
            raise ValueError("A base benigna está vazia.")
        pools, available, seen = [None] * 3, np.zeros(3, dtype=int), set()
        generators = self.generators("BENIGN", seed)
        for chunk in self.reader.iterChunks(self.benignPath):
            mask = []
            for key in self.identities(chunk):
                accept = key not in seen
                mask.append(accept)
                seen.add(key)
            group = chunk.loc[mask].copy()
            group["__Stratum"] = self.strata(group.FlowEnd, first, last)
            for i in range(3):
                part = group.loc[group["__Stratum"].eq(i)]
                available[i] += len(part)
                pools[i] = self.keepSmallest(pools[i], part, count, generators[i])
        if available.sum() < count:
            raise ValueError(f"Referência benigna inviável: necessários {count} registros únicos, disponíveis {available.sum()}.")
        quotas = np.minimum(self.quotas(count), available)
        while quotas.sum() < count:
            for i in range(3):
                if quotas[i] < available[i] and quotas.sum() < count:
                    quotas[i] += 1
        pieces = [p.iloc[:quotas[i]].drop(columns="__Priority") for i, p in enumerate(pools) if p is not None and quotas[i]]
        return self.chronological(pd.concat(pieces, ignore_index=True))

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
                    "BlockIndex": index, "ScenarioPart": kind, "TargetAttack": target,
                    "InstanceStart": position, "InstanceEnd": position + len(frame) - 1,
                    "Instances": len(frame), "AttackInstances": int(attackMask.sum()),
                    "BenignInstances": int((~attackMask).sum()), "RequestedAttackInstances": count if kind == "mixed_attack" else 0,
                    "BeginningAttackInstances": int((attackMask & frame["__Stratum"].eq(0)).sum()),
                    "MiddleAttackInstances": int((attackMask & frame["__Stratum"].eq(1)).sum()),
                    "EndAttackInstances": int((attackMask & frame["__Stratum"].eq(2)).sum()),
                    "OriginalStart": frame.FlowEnd.min(), "OriginalEnd": frame.FlowEnd.max(),
                    "SamplingSeed": seed, "BenignReferenceInstances": len(benign),
                    "InitialBenignInstances": initial, "BetweenBenignInstances": gap,
                    "RepeatAttackBlocks": repeat, "RepeatOfBlock": repeatOf,
                    "SourceDataset": str(self.benignPath if kind == "benign" else self.attackFiles[target]),
                })
                frame["InstanceIndex"] = np.arange(position, position + len(frame))
                position += len(frame)
                pieces.append(frame)
        full = pd.concat(pieces, ignore_index=True).drop(columns=["__Stratum", "__Priority"], errors="ignore")
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
        if not sizes or min(sizes) < 3:
            raise ValueError("Cada bloco deve ter pelo menos três registros de ataque.")
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
                demand = self.quotas(max(sizes)) * occurrences
                capacity[attack] = np.maximum(capacity.get(attack, np.zeros(3, dtype=int)), demand)
        if not configs:
            raise ValueError("Informe pelo menos um cenário.")
        pools, errors, availability = {}, {}, []
        for attack in sorted(capacity):
            try:
                pools[attack] = self.loadAttackPool(attack, capacity[attack], seed)
                data = pools[attack]
                availability.append({"Attack": attack, "BeginningFlows": data["counts"][0],
                                     "MiddleFlows": data["counts"][1], "EndFlows": data["counts"][2],
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
                    benign["__Stratum"] = self.strata(benign.FlowEnd, benign.FlowEnd.min(), benign.FlowEnd.max())
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
                    benign.drop(columns="__Stratum").to_csv(path, index=False)
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
        benign.drop(columns="__Stratum").to_csv(path, index=False)
        return pd.DataFrame([{
            "Dataset": "BENIGN", "Path": str(path), "Instances": len(benign),
            "BeginningInstances": int(benign["__Stratum"].eq(0).sum()),
            "MiddleInstances": int(benign["__Stratum"].eq(1).sum()),
            "EndInstances": int(benign["__Stratum"].eq(2).sum()),
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
