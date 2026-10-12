from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import ProjectDefaults as defaults
from src.Scenario.attack_blocks import build_attack_block
from src.Scenario.baseline import build_benign_baseline
from src.Scenario.scenarios import build_scenario, prepare_scenario_attack_segments


@dataclass(frozen=True)
class ScenarioConfig:
    dataDirectory: Path
    benignSourceFiles: tuple[str, ...]
    attackFiles: dict[str, str]
    scenarios: dict[str, tuple[str, ...]]
    experimentName: str = "Scenarios"
    outputRoot: Path = defaults.DEFAULT_SCENARIO_OUTPUT_ROOT
    window: str = "1s"
    attackRatioThreshold: float = 0.5
    preferredAttackDuration: str = "5min"
    chunkSize: int = 100_000

    def __post_init__(self):
        # Normaliza caminhos e valida as regras necessárias para construir cenários binários reproduzíveis.
        if not str(self.experimentName).strip():
            raise ValueError("experimentName must be provided explicitly.")
        if not self.benignSourceFiles or not self.attackFiles or not self.scenarios:
            raise ValueError("Benign files, attacks, and scenarios must be provided.")
        if not 0.0 <= float(self.attackRatioThreshold) <= 1.0:
            raise ValueError("attackRatioThreshold must be between 0 and 1.")
        object.__setattr__(self, "dataDirectory", Path(self.dataDirectory))
        object.__setattr__(self, "outputRoot", Path(self.outputRoot))
        object.__setattr__(self, "benignSourceFiles", tuple(self.benignSourceFiles))
        object.__setattr__(self, "attackFiles", {str(key): str(value) for key, value in self.attackFiles.items()})
        object.__setattr__(self, "scenarios", {str(key): tuple(value) for key, value in self.scenarios.items()})


class ScenarioBuilder:
    def __init__(self, config):
        # Inicializa o orquestrador que concentra toda a criação de baseline, blocos e cenários fora do notebook.
        if not isinstance(config, ScenarioConfig):
            raise TypeError("config must be a ScenarioConfig.")
        self.config = config

    def run(self):
        # Constrói baseline benigno, blocos de ataque, segmentos necessários e CSVs finais com rótulo binário.
        root = self.config.outputRoot / self.config.experimentName
        window_tag = self.getWindowTag()
        baseline_directory = root / "Baseline"
        block_directory = root / "AttackBlocks"
        scenario_directory = root / "Datasets"
        baseline = build_benign_baseline(
            data_dir=self.config.dataDirectory,
            source_files=self.config.benignSourceFiles,
            output_path=baseline_directory / f"BENIGN_{window_tag}.csv",
            window=self.config.window,
            attack_ratio_threshold=self.config.attackRatioThreshold,
            chunk_size=self.config.chunkSize,
        )
        attack_blocks = {}
        for attack_name, file_name in self.config.attackFiles.items():
            source_path = self.resolveSourcePath(file_name)
            attack_blocks[attack_name] = build_attack_block(
                csv_path=source_path,
                attack_name=attack_name,
                output_path=block_directory / f"{attack_name}_{window_tag}.csv",
                window=self.config.window,
                attack_ratio_threshold=self.config.attackRatioThreshold,
                chunk_size=self.config.chunkSize,
            )
        attack_segments, segment_summary = prepare_scenario_attack_segments(
            attack_blocks=attack_blocks,
            scenarios=self.config.scenarios,
            preferred_duration=self.config.preferredAttackDuration,
        )
        scenario_outputs = {}
        summary_rows = []
        for scenario_name, sequence in self.config.scenarios.items():
            output_path = scenario_directory / f"{scenario_name}_{window_tag}.csv"
            frame = build_scenario(
                scenario_name=scenario_name,
                sequence=sequence,
                baseline=baseline,
                attack_segments=attack_segments,
                output_path=output_path,
                window=self.config.window,
            )
            scenario_outputs[scenario_name] = str(output_path)
            summary_rows.append({
                "scenario": scenario_name,
                "instances": int(len(frame)),
                "benignInstances": int((frame["Label"] == "BENIGN").sum()),
                "attackInstances": int((frame["Label"] == "ATTACK").sum()),
                "attackRatioPercent": float((frame["Label"] == "ATTACK").mean() * 100.0),
                "path": str(output_path),
            })
        summary = pd.DataFrame(summary_rows)
        summary_path = root / "scenario_summary.csv"
        segment_summary_path = root / "attack_segment_summary.csv"
        root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        segment_summary.to_csv(segment_summary_path, index=False)
        return {
            "outputDirectory": str(root),
            "baselinePath": str(baseline_directory / f"BENIGN_{window_tag}.csv"),
            "scenarioPaths": scenario_outputs,
            "scenarioSummary": summary,
            "scenarioSummaryPath": str(summary_path),
            "attackSegmentSummary": segment_summary,
            "attackSegmentSummaryPath": str(segment_summary_path),
        }

    def getWindowTag(self):
        # Converte a configuração temporal em um sufixo curto e seguro para os arquivos gerados.
        return str(self.config.window).strip().replace(" ", "").replace("/", "_")

    def resolveSourcePath(self, value):
        # Resolve caminhos absolutos e nomes relativos sem incorporar diretórios específicos da máquina no backend.
        path = Path(value)
        return path if path.is_absolute() else self.config.dataDirectory / path
