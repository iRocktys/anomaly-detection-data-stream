from collections import deque


class ScoreMovingAverages:
    def __init__(self, windowSizes=None):
        # Inicializa todas as médias móveis causais requisitadas pelo experimento.
        self.windowSizes = self.validateWindowSizes(windowSizes)
        self.reset()

    def validateWindowSizes(self, windowSizes):
        # Normaliza tamanhos de janela e rejeita valores não positivos.
        if windowSizes is None:
            return []
        validated_sizes = sorted(set(int(window_size) for window_size in windowSizes))
        if any(window_size <= 0 for window_size in validated_sizes):
            raise ValueError("Window sizes must be greater than zero.")
        return validated_sizes

    def calculate(self, score):
        # Insere o score atual e calcula cada média móvel usando somente o histórico disponível.
        score = float(score)
        self.history.append(score)
        averages = {}
        history_values = list(self.history)
        for window_size in self.windowSizes:
            values = history_values[-window_size:]
            averages[f"scoreMa{window_size}"] = sum(values) / len(values)
        return averages

    def reset(self):
        # Reinicia o histórico limitado pelo maior tamanho de média móvel configurado.
        maximum_window = max(self.windowSizes, default=1)
        self.history = deque(maxlen=maximum_window)
