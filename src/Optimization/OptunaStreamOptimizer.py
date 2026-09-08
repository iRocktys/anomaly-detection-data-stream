"""Coordenação da otimização conjunta, preservando a interface do notebook."""
from dataclasses import asdict, replace
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import unicodedata

import pandas as pd

from src.Anomaly.Models import ModelRegistry
from src.Anomaly.Thresholds.Incremental.DspotThreshold import DspotConfig, DspotThreshold
from src.Data.OnlineImputers import IncrementalMeanImputer, ZeroOnlineImputer
from src.Data.OnlineNormalizers import IncrementalZScoreNormalizer, NoOnlineNormalizer
from src.Optimization.AifSearchSpace import AifSearchSpace
from src.Optimization.DspotSearchSpace import DspotSearchSpace
from src.Optimization.OptimizationConfig import DatasetProfile, ModelProfile, OptimizationConfig, PreparedScenario
from src.Optimization.OptimizationExporter import OptimizationExporter
from src.Pipeline.MetricsResultManager import MetricsResultManager
from src.Pipeline.TrainingPipeline import TrainingPipeline
from src.Training.TrainAllStrategy import TrainAllStrategy


class OptunaStreamOptimizer:
    studyVersion = "joint_v2"
    scenarioAliases = {
        "adaptacao": "Adaptation", "adaptation": "Adaptation",
        "consistencia": "Consistency", "consistency": "Consistency",
        "generalizacao": "Generalization", "generalization": "Generalization",
        "recorrencia": "Recurrence", "recurrence": "Recurrence",
    }
    imputerFactories = {"zero": ZeroOnlineImputer, "incrementalMean": IncrementalMeanImputer}

    def __init__(self, config=None, datasetProfile=None, modelProfile=None,
                 searchSpace=None, modelSearchSpace=None):
        self.config = config or OptimizationConfig()
        self.datasetProfile = datasetProfile or DatasetProfile()
        self.modelProfile = modelProfile or ModelProfile()
        self.searchSpace = searchSpace or DspotSearchSpace(self.config.dspotSearchSpace)
        self.modelSearchSpace = modelSearchSpace
        if self.config.optimizeModelParameters and self.modelSearchSpace is None:
            if self.modelProfile.code == "AIF":
                self.modelSearchSpace = AifSearchSpace(self.config.aifSearchSpace)
        if not self.config.optimizeModelParameters:
            self.modelSearchSpace = None
        # Uso explícito do espaço antigo conserva a seleção de imputação dele.
        self._explicitSearchSpace = searchSpace is not None
        self._validateConfiguration()

    def _validateConfiguration(self):
        parameters = {**self.modelProfile.parameters, **self.config.fixedModelParameters}
        ModelRegistry.validateParameters(self.modelProfile.code, parameters)
        if "schema" in parameters:
            raise ValueError("O schema é obtido do dataset e não pode ser fixado nos parâmetros.")
        if self.modelProfile.code == "AIF":
            # Parâmetros em busca substituem o perfil, portanto defaults ignorados
            # não podem bloquear um espaço menor ou um warm-up personalizado.
            preview = dict(parameters)
            if isinstance(self.modelSearchSpace, AifSearchSpace):
                cfg = self.modelSearchSpace.config
                windows = [value for value in cfg.windowSizes if value <= self.config.initialWarmupSize]
                if "window_size" not in self.config.fixedModelParameters and not windows:
                    raise ValueError("Nenhuma janela do AIF cabe no warm-up.")
                preview.update({
                    "window_size": windows[0] if windows else parameters["window_size"],
                    "n_trees": cfg.nTreesMinimum, "height": cfg.heightMinimum,
                    "m_trees": cfg.mTreesMinimum, "weights": cfg.weightsMinimum,
                })
                preview.update(self.config.fixedModelParameters)
            AifSearchSpace.validateParameters(preview, self.config.initialWarmupSize)
        if len({self.scenarioAliases.get(self._normalize(v), v) for v in self.config.scenarios}) != len(self.config.scenarios):
            raise ValueError("Há cenários duplicados após normalizar os nomes.")
        cfg = getattr(self.searchSpace, "config", None)
        if cfg is not None and getattr(cfg, "calibrationWindowMaximum", 0) > self.config.initialWarmupSize:
            raise ValueError("A maior calibração do DSPOT deve caber no warm-up.")
        self._makeNormalizer()  # Falhas gerais devem aparecer antes de criar trials.

    def run(self):
        optuna = self._loadOptuna()
        outputDirectory = self.config.outputRoot / self.modelProfile.code
        outputDirectory.mkdir(parents=True, exist_ok=True)
        databasePath = (outputDirectory / "optuna.db").resolve()
        paths = self._discoverScenarios()
        studies = {}
        for index, (name, path) in enumerate(paths.items()):
            studyName = f"{self.modelProfile.code}_{name}_block_{self.config.blockSize}_{self.studyVersion}"
            study = optuna.create_study(
                study_name=studyName, storage=f"sqlite:///{databasePath.as_posix()}",
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.config.optunaSeed + index),
                pruner=optuna.pruners.NopPruner(), load_if_exists=True,
            )
            self._validateStudy(study, name, path)
            # Ao retomar durante os trials iniciais aleatórios, não reinicia a
            # mesma sequência de sugestões. A ordem depende das sessões de execução.
            samplerSeed = self.config.optunaSeed + index
            if study.trials:
                seedText = f"{samplerSeed}:{studyName}:{len(study.trials)}"
                samplerSeed = int(hashlib.sha256(seedText.encode()).hexdigest()[:8], 16)
            study.sampler = optuna.samplers.TPESampler(seed=samplerSeed)
            study.set_user_attr("lastSamplerSeed", samplerSeed)
            studies[name] = study
        exporter = OptimizationExporter(outputDirectory, topK=self.config.topK)
        exporter.preservePreviousExports(studies)
        # Todos os estudos entram no export, inclusive ao retomar uma execução.
        exportedPaths = exporter.export(studies)
        for name, path in paths.items():
            study = studies[name]
            remaining = max(0, self.config.nTrials - len(study.trials))
            print(f"{self.modelProfile.code} | {name}: {len(study.trials)} trials registrados; {remaining} restantes.")
            if not remaining:
                continue
            # Apenas um dataset fica preparado em memória por vez.
            scenario = self._prepareScenario(name, path)
            try:
                study.optimize(
                    partial(self._objective, scenario=scenario, exporter=exporter),
                    n_trials=remaining,
                    catch=(ValueError, RuntimeError, FloatingPointError, OverflowError),
                    callbacks=[lambda _study, _trial: exporter.export(studies)],
                    gc_after_trial=True, show_progress_bar=False,
                )
            finally:
                exportedPaths = exporter.export(studies)
                del scenario
        return {"databasePath": str(databasePath), **exportedPaths, "studies": studies}

    def _objective(self, trial, scenario, exporter):
        exporter.recordIdentity(trial, scenario, self.modelProfile.code)
        try:
            configuration = self._suggestConfiguration(trial)
            pipeline = self._createPipeline(scenario, configuration)
            exporter.recordConfiguration(trial, configuration, self.config)
            result = pipeline.run()
            if result.streamMetrics["evaluatedInstances"] != scenario.totalInstances - self.config.initialWarmupSize:
                raise RuntimeError("O trial não avaliou todas as instâncias após o warm-up.")
            exporter.recordResult(trial, result)
            return float(result.streamMetrics["f1"])
        except Exception as error:
            trial.set_user_attr("errorType", type(error).__name__)
            trial.set_user_attr("errorMessage", str(error))
            raise

    def _suggestConfiguration(self, trial):
        parameters = dict(self.modelProfile.parameters)
        if self.modelSearchSpace is not None:
            parameters.update(self.modelSearchSpace.suggest(
                trial, fixedParameters=self.config.fixedModelParameters,
                initialWarmupSize=self.config.initialWarmupSize,
            ))
        parameters.update(self.config.fixedModelParameters)
        definition = ModelRegistry.definitions[self.modelProfile.code]
        parameters.setdefault(definition.seedParameter, self.config.seed)
        ModelRegistry.validateParameters(self.modelProfile.code, parameters)
        if self.modelProfile.code == "AIF":
            AifSearchSpace.validateParameters(parameters, self.config.initialWarmupSize)
        if self._explicitSearchSpace:
            configuration = self.searchSpace.suggest(trial)
        else:
            imputer = trial.suggest_categorical("imputer", list(self.config.imputerNames))
            configuration = self.searchSpace.suggest(trial, imputerName=imputer)
        if configuration.calibrationWindow > self.config.initialWarmupSize:
            raise ValueError("A calibração do trial ultrapassa o warm-up.")
        return replace(configuration, modelParameters=parameters)

    def _makeNormalizer(self):
        if self.config.normalizerName == "none":
            return NoOnlineNormalizer()
        return IncrementalZScoreNormalizer(**self.config.normalizerParameters)

    def _createPipeline(self, scenario, configuration):
        if configuration.imputerName not in self.imputerFactories:
            raise ValueError(f"Imputador desconhecido: {configuration.imputerName}.")
        threshold = DspotThreshold(DspotConfig(
            driftDepth=configuration.driftDepth, calibrationSize=configuration.calibrationSize,
            initialQuantile=configuration.initialQuantile, risk=configuration.risk,
            refitEvery=configuration.refitEvery, optimizationStarts=configuration.optimizationStarts,
            tolerance=configuration.tolerance,
        ))
        return TrainingPipeline(
            stream=scenario.stream, datasetName=scenario.datasetName,
            modelCode=self.modelProfile.code, modelParameters=dict(configuration.modelParameters),
            threshold=threshold, labelNames=scenario.labelNames,
            imputer=self.imputerFactories[configuration.imputerName](), normalizer=self._makeNormalizer(),
            trainingStrategy=TrainAllStrategy(), scoreWindowSizes=configuration.scoreWindowSizes,
            metricsWindowSize=self.config.metricsWindowSize, outputPath=self.config.outputRoot,
            normalClassIndex=0, seed=self.config.seed, generatePlots=False,
            initialWarmupSize=self.config.initialWarmupSize,
            thresholdScoreSource=configuration.thresholdScoreSource,
            resultManager=MetricsResultManager(), strictTrainingErrors=True,
        )

    def _discoverScenarios(self):
        root = self.config.dataRoot
        if not root.exists():
            raise FileNotFoundError(f"Diretório de datasets não encontrado: {root.resolve()}")
        names = [self.scenarioAliases.get(self._normalize(v), v) for v in self.config.scenarios]
        expected = {name: [] for name in names}
        suffix = f"_{self.config.blockSize}"
        for path in sorted(root.rglob("*.csv")):
            stem = self._normalize(path.stem)
            if stem.endswith(suffix):
                name = self.scenarioAliases.get(stem[:-len(suffix)].rstrip("_"))
                if name in expected:
                    expected[name].append(path)
        resolved = {}
        for name, matches in expected.items():
            if not matches:
                raise FileNotFoundError(f"Dataset {name}_{self.config.blockSize} não encontrado em {root.resolve()}.")
            if len(matches) > 1:
                raise ValueError(f"Mais de um dataset para {name}: {[str(p) for p in matches]}")
            resolved[name] = matches[0]
        return resolved

    def _prepareScenario(self, name, path):
        from src.Data.Processor import DataStreamProcessor

        dataframe = pd.read_csv(path)
        processor = DataStreamProcessor(
            logging=False, selected_features=self.datasetProfile.selectedFeatures,
            removed_features=self.datasetProfile.removedFeatures,
        )
        stream, targetNames, featureNames, labelNames = processor.create_stream(
            dataframe, target_label_col=self.datasetProfile.targetColumn,
            binary_label=self.datasetProfile.binaryLabel,
        )
        if not self.datasetProfile.binaryLabel:
            if not targetNames or str(targetNames[0]).strip().upper() not in ("BENIGN", "NORMAL"):
                raise ValueError("O dataset multiclasse deve identificar BENIGN/NORMAL como classe 0.")
        total = len(labelNames)
        if total <= self.config.initialWarmupSize:
            raise ValueError(f"{name}: nenhuma instância disponível após o warm-up.")
        attacks = int((~pd.Series(labelNames).astype(str).str.upper().isin(["BENIGN", "NORMAL"])).sum())
        return PreparedScenario(
            name=name, datasetPath=path, datasetName=path.stem, stream=stream,
            targetNames=tuple(targetNames), featureNames=tuple(featureNames), labelNames=tuple(labelNames),
            totalInstances=total, attackInstances=attacks, attackRatioPercent=100.0 * attacks / total,
        )

    def _validateStudy(self, study, scenarioName, datasetPath):
        payload = self._configurationPayload(scenarioName, datasetPath)
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        stored = study.user_attrs.get("configurationSignature")
        if (stored is None and study.trials) or (stored is not None and stored != signature):
            raise ValueError(
                f"A configuração atual não corresponde ao estudo '{study.study_name}'. "
                "Restaure a configuração anterior ou use outro outputRoot. Os resultados existentes foram preservados."
            )
        if stored is None:
            study.set_user_attr("configurationSignature", signature)
            study.set_user_attr("configuration", json.loads(json.dumps(payload, default=str)))
            study.set_user_attr("scenario", scenarioName)
            study.set_user_attr("dataset", datasetPath.stem)
            study.set_user_attr("blockSize", self.config.blockSize)
            study.set_user_attr("modelCode", self.modelProfile.code)

    def _configurationPayload(self, scenarioName, datasetPath):
        configuration = asdict(self.config)
        # Orçamento/ranking podem mudar na retomada; protocolo e busca não.
        for name in ("nTrials", "topK", "dataRoot", "outputRoot", "scenarios"):
            configuration.pop(name)
        versions = {}
        for package in ("capymoa", "optuna", "numpy", "scipy"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not-installed"
        return {
            "studyVersion": self.studyVersion, "scenario": scenarioName,
            "samplerBaseSeed": self.config.optunaSeed + [
                self.scenarioAliases.get(self._normalize(v), v) for v in self.config.scenarios
            ].index(scenarioName),
            "datasetHash": self._fileHash(datasetPath), "dataset": datasetPath.stem,
            "optimization": configuration, "datasetProfile": asdict(self.datasetProfile),
            "modelProfile": asdict(self.modelProfile), "versions": versions,
            "searchSpace": self._spaceSignature(self.searchSpace),
            "explicitSearchSpace": self._explicitSearchSpace,
            "modelSearchSpace": self._spaceSignature(self.modelSearchSpace),
        }

    @staticmethod
    def _spaceSignature(space):
        if space is None:
            return None
        config = getattr(space, "config", None)
        return {"class": f"{type(space).__module__}.{type(space).__qualname__}", "config": repr(config)}

    @staticmethod
    def _fileHash(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _normalize(value):
        return (unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore")
                .decode("ascii").strip().lower().replace("-", "_").replace(" ", "_"))

    @staticmethod
    def _loadOptuna():
        try:
            import optuna
        except ImportError as error:
            raise ImportError("Optuna não está instalado. Execute 'pip install optuna'.") from error
        return optuna
