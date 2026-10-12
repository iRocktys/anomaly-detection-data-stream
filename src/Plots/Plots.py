from src.Plots.ConfusionMatrixPlot import ConfusionMatrixPlot
from src.Plots.ErrorWindowPlot import ErrorWindowPlot
from src.Plots.MetricsWindowPlot import MetricsWindowPlot
from src.Plots.ScorePlot import ScorePlot


class Plots:
    def __init__(self):
        # Inicializa a fachada única para todos os gráficos persistidos pelos experimentos.
        self.scorePlot = ScorePlot()
        self.errorWindowPlot = ErrorWindowPlot()
        self.metricsWindowPlot = MetricsWindowPlot()
        self.confusionMatrixPlot = ConfusionMatrixPlot()

    def plotScoreArtifact(self, source, outputPath=None, **options):
        # Gera o gráfico de scores e threshold para detectores de anomalias.
        return self.scorePlot.plot(source, outputPath=outputPath, **options)

    def plotWindowErrors(self, source, attackSource=None, outputPath=None, **options):
        # Gera o gráfico janelado de falsos positivos e falsos negativos.
        return self.errorWindowPlot.plot(source, attackSource=attackSource, outputPath=outputPath, **options)

    def plotWindowMetrics(self, source, attackSource=None, outputPath=None, **options):
        # Gera o gráfico das métricas binárias acumuladas por janela.
        return self.metricsWindowPlot.plot(source, attackSource=attackSource, outputPath=outputPath, **options)

    def plotConfusionMatrix(self, source, outputPath=None, **options):
        # Gera a matriz de confusão binária a partir do arquivo de instâncias.
        return self.confusionMatrixPlot.plot(source, outputPath=outputPath, **options)
