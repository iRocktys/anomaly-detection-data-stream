from pathlib import Path

from src.Plots.ConfusionMatrixPlot import ConfusionMatrixPlot
from src.Plots.ErrorWindowPlot import ErrorWindowPlot
from src.Plots.MetricsWindowPlot import MetricsWindowPlot
from src.Plots.PredictionPlot import PredictionPlot
from src.Plots.ScorePlot import ScorePlot


class PlotManager:
    def generate(self, task, instanceSource, windowSource, outputDirectory, movingAverageColumns=(), windowSize=100):
        # Gera somente os gráficos compatíveis com a tarefa e com o contrato único de resultados.
        output_directory = Path(outputDirectory)
        output_directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics": MetricsWindowPlot().plot(windowSource, attackSource=instanceSource, outputPath=output_directory / "metrics_windows.png", windowSize=windowSize),
            "errors": ErrorWindowPlot().plot(windowSource, attackSource=instanceSource, outputPath=output_directory / "fp_fn_windows.png", windowSize=windowSize),
            "confusionMatrix": ConfusionMatrixPlot().plot(instanceSource, outputPath=output_directory / "confusion_matrix.png"),
        }
        if str(task).strip().lower() == "anomaly":
            paths["scores"] = ScorePlot().plot(instanceSource, outputPath=output_directory / "scores.png", movingAverageColumns=movingAverageColumns)
        elif str(task).strip().lower() == "classification":
            paths["predictions"] = PredictionPlot().plot(instanceSource, outputPath=output_directory / "predictions.png")
        return paths
