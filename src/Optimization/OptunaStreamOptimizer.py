import hashlib
from pathlib import Path

from src.Optimization.DspotSearchSpace import DspotSearchSpace
from src.Optimization.OptimizationConfig import (
    DatasetProfile,
    ModelProfile,
    OptimizationConfig,
)
from src.Optimization.OptimizationExporter import OptimizationExporter
from src.Optimization.PipelineFactory import PipelineFactory
from src.Optimization.ScenarioRepository import ScenarioRepository
from src.Optimization.StreamObjective import StreamObjective


class OptunaStreamOptimizer:
    def __init__(
        self,
        config=None,
        datasetProfile=None,
        modelProfile=None,
        searchSpace=None,
    ):
        self.config = config or OptimizationConfig()
        self.datasetProfile = datasetProfile or DatasetProfile()
        self.modelProfile = modelProfile or ModelProfile()
        self.searchSpace = searchSpace or DspotSearchSpace()

    def run(self):
        optuna = self._loadOptuna()
        outputDirectory = (
            self.config.outputRoot / self.modelProfile.code.strip().upper()
        )
        outputDirectory.mkdir(parents=True, exist_ok=True)
        databasePath = (outputDirectory / "optuna.db").resolve()
        storageUrl = f"sqlite:///{databasePath.as_posix()}"

        repository = ScenarioRepository(
            dataRoot=self.config.dataRoot,
            datasetProfile=self.datasetProfile,
        )
        scenarios = repository.prepare(
            scenarios=self.config.scenarios,
            blockSize=self.config.blockSize,
        )
        pipelineFactory = PipelineFactory(
            optimizationConfig=self.config,
            modelProfile=self.modelProfile,
        )
        exporter = OptimizationExporter(
            outputDirectory=outputDirectory,
            topK=self.config.topK,
        )
        studies = {}

        for index, scenario in enumerate(scenarios):
            studyName = (
                f"{self.modelProfile.code.strip().upper()}_"
                f"{scenario.name}_block_{self.config.blockSize}_stream_v1"
            )
            study = optuna.create_study(
                study_name=studyName,
                storage=storageUrl,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.config.optunaSeed + index
                ),
                pruner=optuna.pruners.NopPruner(),
                load_if_exists=True,
            )
            self._validateStudy(study, scenario)
            objective = StreamObjective(
                scenario=scenario,
                searchSpace=self.searchSpace,
                pipelineFactory=pipelineFactory,
                modelProfile=self.modelProfile,
            )
            remainingTrials = max(
                0,
                int(self.config.nTrials) - len(study.trials),
            )
            if remainingTrials:
                study.optimize(
                    objective,
                    n_trials=remainingTrials,
                    catch=(
                        ValueError,
                        RuntimeError,
                        FloatingPointError,
                        OverflowError,
                    ),
                    gc_after_trial=True,
                    show_progress_bar=False,
                )
            studies[scenario.name] = study
            paths = exporter.export(studies)

        return {
            "databasePath": str(databasePath),
            **paths,
            "studies": studies,
        }

    def _validateStudy(self, study, scenario):
        signature = self._configurationSignature(scenario)
        storedSignature = study.user_attrs.get("configurationSignature")
        if storedSignature is None and study.trials:
            raise ValueError(
                f"O estudo existente '{study.study_name}' não possui assinatura "
                "de configuração e não pode ser retomado com segurança."
            )
        if storedSignature is not None and storedSignature != signature:
            raise ValueError(
                f"A configuração atual não corresponde ao estudo existente "
                f"'{study.study_name}'. Use outro diretório de saída ou restaure "
                "a configuração original."
            )
        if storedSignature is None:
            study.set_user_attr("configurationSignature", signature)
            study.set_user_attr("scenario", scenario.name)
            study.set_user_attr("dataset", scenario.datasetName)
            study.set_user_attr("blockSize", int(self.config.blockSize))

    def _configurationSignature(self, scenario):
        payload = {
            "scenario": scenario.name,
            "datasetHash": self._fileHash(scenario.datasetPath),
            "blockSize": int(self.config.blockSize),
            "metricsWindowSize": int(self.config.metricsWindowSize),
            "initialWarmupSize": int(self.config.initialWarmupSize),
            "seed": int(self.config.seed),
            "modelCode": self.modelProfile.code,
            "modelParameters": sorted(self.modelProfile.parameters.items()),
            "targetColumn": self.datasetProfile.targetColumn,
            "selectedFeatures": self.datasetProfile.selectedFeatures,
            "removedFeatures": self.datasetProfile.removedFeatures,
            "binaryLabel": self.datasetProfile.binaryLabel,
            "searchSpace": repr(getattr(self.searchSpace, "config", None)),
        }
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def _fileHash(self, path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _loadOptuna(self):
        try:
            import optuna
        except ImportError as error:
            raise ImportError(
                "Optuna não está instalado. Execute 'pip install optuna' "
                "antes de iniciar a otimização."
            ) from error
        return optuna
