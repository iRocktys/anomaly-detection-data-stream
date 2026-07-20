from src.Plots.ErrorWindowPlot import ErrorWindowPlot
from src.Plots.MetricsWindowPlot import MetricsWindowPlot
from src.Plots.ScorePlot import ScorePlot


class Plots:
    def __init__(self):
        self.scorePlot = ScorePlot()
        self.errorWindowPlot = ErrorWindowPlot()
        self.metricsWindowPlot = MetricsWindowPlot()

    def plotScoreArtifact(self, source, outputPath=None, **options):
        return self.scorePlot.plot(source, outputPath=outputPath, **options)

    def plotWindowErrors(self, source, attackSource=None, outputPath=None, **options):
        return self.errorWindowPlot.plot(source, attackSource=attackSource, outputPath=outputPath, **options)

    def plotWindowMetrics(self, source, attackSource=None, outputPath=None, **options):
        return self.metricsWindowPlot.plot(source, attackSource=attackSource, outputPath=outputPath, **options)