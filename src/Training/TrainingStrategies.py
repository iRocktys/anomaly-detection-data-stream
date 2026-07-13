from abc import ABC, abstractmethod


class BaseTrainingStrategy(ABC):
    """Contrato para decidir se o modelo aprende com a instância atual."""

    name = "base"
    requiresPrediction = False

    @abstractmethod
    def shouldTrain(
        self,
        prediction=None,
        thresholdReady=False,
        isWarmup=False,
        trueLabel=None,
        isAttack=None,
    ):
        raise NotImplementedError


class TrainAllStrategy(BaseTrainingStrategy):
    """Estratégia padrão: aprende incrementalmente com todas as instâncias."""

    name = "all"
    requiresPrediction = False

    def __init__(self, evaluationName=None):
        self.evaluationName = evaluationName

    def shouldTrain(
        self,
        prediction=None,
        thresholdReady=False,
        isWarmup=False,
        trueLabel=None,
        isAttack=None,
    ):
        return True


class TrainPredictedNormalStrategy(BaseTrainingStrategy):
    """Treina somente com instâncias classificadas como normais pelo threshold.

    Durante o warmup ainda não existe um threshold pronto. Por padrão, todas as
    instâncias desse período treinam o modelo para permitir sua inicialização.
    Depois do warmup, somente ``prediction == 0`` atualiza o modelo.

    Como os scores futuros passam a depender das decisões anteriores, esta
    estratégia cria um fluxo fechado entre modelo e threshold. Por isso, a
    avaliação indicada por ``evaluationName`` deve ser executada durante a
    própria geração dos scores e não apenas em replay posterior.
    """

    name = "predictedNormal"
    requiresPrediction = True

    def __init__(self, trainDuringWarmup=True, evaluationName=None):
        self.trainDuringWarmup = bool(trainDuringWarmup)
        self.evaluationName = evaluationName

    def shouldTrain(
        self,
        prediction=None,
        thresholdReady=False,
        isWarmup=False,
        trueLabel=None,
        isAttack=None,
    ):
        if bool(isWarmup) or not bool(thresholdReady):
            return self.trainDuringWarmup
        if prediction is None:
            raise ValueError(
                "TrainPredictedNormalStrategy requer a predição causal do threshold."
            )
        return int(prediction) == 0
