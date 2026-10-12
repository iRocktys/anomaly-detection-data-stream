class ClassifierSearchSpace:
    def suggest(self, trial, modelCode, fixedParameters=None):
        # Sugere hiperparâmetros somente para o classificador binário selecionado.
        code = str(modelCode).strip().upper()
        fixed = dict(fixedParameters or {})
        if code == "ARF":
            parameters = {
                "ensemble_size": fixed.get("ensemble_size", trial.suggest_int("arf_ensemble_size", 10, 100, step=10)),
                "max_features": fixed.get("max_features", trial.suggest_float("arf_max_features", 0.2, 1.0, step=0.1)),
                "lambda_param": fixed.get("lambda_param", trial.suggest_float("arf_lambda_param", 2.0, 10.0, step=1.0)),
                "number_of_jobs": fixed.get("number_of_jobs", 1),
            }
        elif code in {"HAT", "HT"}:
            prefix = code.lower()
            parameters = {
                "grace_period": fixed.get("grace_period", trial.suggest_int(f"{prefix}_grace_period", 50, 500, step=50)),
                "confidence": fixed.get("confidence", trial.suggest_float(f"{prefix}_confidence", 1e-7, 1e-2, log=True)),
                "tie_threshold": fixed.get("tie_threshold", trial.suggest_float(f"{prefix}_tie_threshold", 0.01, 0.10)),
                "leaf_prediction": fixed.get("leaf_prediction", "NaiveBayesAdaptive"),
            }
        elif code == "NB":
            parameters = {}
        else:
            raise ValueError(f"No classifier search space is registered for: {modelCode}.")
        parameters.update(fixed)
        return parameters
