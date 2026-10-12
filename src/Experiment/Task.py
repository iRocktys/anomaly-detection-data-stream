def normalizeTask(task):
    # Normaliza aliases para as duas tarefas binárias suportadas pelo projeto.
    value = str(task).strip().lower().replace("_", "").replace("-", "")
    if value in {"anomaly", "anomalydetection", "detector"}:
        return "anomaly"
    if value in {"classification", "classifier"}:
        return "classification"
    raise ValueError("task must be 'anomaly' or 'classification'.")
