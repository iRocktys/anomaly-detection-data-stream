from src.Plots.ErrorPlot import ErrorPlot
from src.Plots.MetricsPlot import MetricsPlot
from src.Plots.ScorePlot import ScorePlot


class Plots:
    """Consumidor de artefatos; não executa modelos, thresholds ou métricas."""

    def __init__(self, targetNames=None):
        self.targetNames = targetNames
        self.scorePlot = ScorePlot(self.targetNames)
        self.metricsPlot = MetricsPlot()
        self.errorPlot = ErrorPlot()

    def plotScoreArtifact(self, source, outputPath=None, **options):
        return self.scorePlot.plot(source, outputPath=outputPath, **options)

    def plotWindowMetrics(self, source, outputPath=None, **options):
        return self.metricsPlot.plot(source, outputPath=outputPath, **options)

    def plotWindowErrors(self, source, outputPath=None, **options):
        return self.errorPlot.plot(source, outputPath=outputPath, **options)
