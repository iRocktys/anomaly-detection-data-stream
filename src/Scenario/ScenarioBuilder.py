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

    # Planeja todas as ocorrências e distribui janelas distintas de início, meio e fim entre os blocos.
    def selectAttackBlocks(self, sequence, blockSeconds, randomSeed=42):
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
            requested = np.sum([quota for name, quota in zip(names, quotas) if name == attack], axis=0)
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
                first = cursors[attack][stratum]
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
              startTimestamp="2021-01-01 00:00:00"):
        blocks = self.selectAttackBlocks(sequence, blockSeconds, randomSeed)
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
        pieces, manifests, position = [], [], 0
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
                columns = [c for c in ("Timestamp", "WindowStart", "WindowEnd", "Label", "FlowCount", "AttackCount", "BenignCount") if c in frame]
                manifest = frame[columns].copy().rename(columns={"Timestamp": "OriginalTimestamp", "WindowStart": "OriginalWindowStart", "WindowEnd": "OriginalWindowEnd"})
                manifest["SourceWindowRow"] = sourceRows
                manifest["TemporalStratum"] = np.array(["beginning", "middle", "end"], dtype=object)[np.maximum(strata, 0)] if kind == "mixed_attack" else "benign"
                manifest["SamplingSeed"] = int(randomSeed) if kind == "mixed_attack" else pd.NA
                manifest["SourceDataset"] = str(self.benignPath if kind == "benign" else self.attacksDirectory / f"{target}.csv")
                manifest["TargetAttack"], manifest["ScenarioPart"], manifest["BlockIndex"] = target, kind, blockIndex
                manifest["InstanceIndex"] = np.arange(position, position + len(frame))
                position += len(frame)
                pieces.append(frame)
                manifests.append(manifest)
        full = pd.concat(pieces, ignore_index=True)
        manifest = pd.concat(manifests, ignore_index=True)
        features = [column for column in full if column not in self.metadataColumns]
        dataset = full[["Timestamp"] + features + ["Label"]].copy()
        dataset["Timestamp"] = pd.date_range(startTimestamp, periods=len(dataset), freq=pd.to_timedelta(self.windowSeconds, unit="s"))
        manifest.insert(0, "Timestamp", dataset.Timestamp)
        path = Path(outputPath)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifestPath = path.with_name(f"{path.stem}_manifest.csv")
        dataset.to_csv(path, index=False)
        manifest[[c for c in manifest if c != "Label"] + ["Label"]].to_csv(manifestPath, index=False)
        summary = manifest.groupby(["BlockIndex", "ScenarioPart", "TargetAttack"], sort=False).agg(
            Windows=("Label", "size"), Start=("Timestamp", "min"), End=("Timestamp", "max"),
            AttackWindows=("Label", lambda values: int(values.ne("BENIGN").sum())),
            BeginningWindows=("TemporalStratum", lambda values: int(values.eq("beginning").sum())),
            MiddleWindows=("TemporalStratum", lambda values: int(values.eq("middle").sum())),
            EndWindows=("TemporalStratum", lambda values: int(values.eq("end").sum()))).reset_index()
        summary["OccupiedSeconds"] = summary.Windows * self.windowSeconds
        summary.to_csv(path.with_name(f"{path.stem}_blocks.csv"), index=False)
        return {"datasetPath": str(path), "manifestPath": str(manifestPath), "summary": summary,
                "windows": len(dataset), "attackWindows": int(dataset.Label.ne("BENIGN").sum())}
