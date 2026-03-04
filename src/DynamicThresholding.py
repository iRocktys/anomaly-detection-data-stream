import numpy as np
from collections import deque

class RollingZScore:
    def __init__(self, window_size=500, k=3.0):
        self.window = deque(maxlen=window_size)
        self.k = k

    def update_and_predict(self, score):
        if len(self.window) < 10:
            self.window.append(score)
            return 0
        mean, std = np.mean(self.window), max(np.std(self.window), 1e-5)
        threshold = mean + (self.k * std)
        is_anomaly = 1 if score > threshold else 0
        if not is_anomaly: self.window.append(score)
        return is_anomaly

class DynamicQuantile:
    def __init__(self, window_size=500, quantile=0.95):
        self.window = deque(maxlen=window_size)
        self.quantile = quantile

    def update_and_predict(self, score):
        self.window.append(score)
        if len(self.window) < 10: return 0
        threshold = np.quantile(self.window, self.quantile)
        return 1 if score > threshold else 0

class RollingOtsu:
    def __init__(self, window_size=500, bins=50):
        self.window = deque(maxlen=window_size)
        self.bins = bins

    def update_and_predict(self, score):
        self.window.append(score)
        if len(self.window) < 50: return 0
        hist, bin_edges = np.histogram(self.window, bins=self.bins, range=(0.0, 1.0))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total, best_variance, threshold = len(self.window), float('inf'), 0.5
        for i in range(1, self.bins):
            weight_bg, weight_fg = np.sum(hist[:i]) / total, np.sum(hist[i:]) / total
            if weight_bg == 0 or weight_fg == 0: continue
            mean_bg = np.sum(bin_centers[:i] * hist[:i]) / np.sum(hist[:i])
            mean_fg = np.sum(bin_centers[i:] * hist[i:]) / np.sum(hist[i:])
            var_bg = np.sum(((bin_centers[:i] - mean_bg) ** 2) * hist[:i]) / np.sum(hist[:i])
            var_fg = np.sum(((bin_centers[i:] - mean_fg) ** 2) * hist[i:]) / np.sum(hist[i:])
            within_class_variance = weight_bg * var_bg + weight_fg * var_fg
            if within_class_variance < best_variance:
                best_variance, threshold = within_class_variance, bin_edges[i]
        return 1 if score >= threshold else 0

class EmpiricalPOT:
    def __init__(self, window_size=500, tail_fraction=0.1, margin_multiplier=1.5):
        self.window = deque(maxlen=window_size)
        self.tail_fraction, self.margin_multiplier = tail_fraction, margin_multiplier

    def update_and_predict(self, score):
        self.window.append(score)
        if len(self.window) < 50: return 0
        base_threshold = np.quantile(self.window, 1.0 - self.tail_fraction)
        exceedances = [x - base_threshold for x in self.window if x > base_threshold]
        if not exceedances: return 1 if score > base_threshold else 0
        dynamic_threshold = base_threshold + (self.margin_multiplier * np.mean(exceedances))
        return 1 if score > dynamic_threshold else 0