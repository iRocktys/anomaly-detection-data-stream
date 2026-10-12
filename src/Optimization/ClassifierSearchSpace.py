class ClassifierSearchSpace:
    def suggest(self, trial, modelCode, fixedParameters=None):
        # Sugere somente hiperparâmetros não fixados do classificador binário selecionado.
        code = str(modelCode).strip().upper()
        fixed = dict(fixedParameters or {})
        if code == "ARF":
            suggested = {
                "ensemble_size": int(fixed["ensemble_size"]) if "ensemble_size" in fixed else trial.suggest_int("arf_ensemble_size", 10, 100, step=10),
                "max_features": float(fixed["max_features"]) if "max_features" in fixed else trial.suggest_float("arf_max_features", 0.2, 1.0, step=0.1),
                "lambda_param": float(fixed["lambda_param"]) if "lambda_param" in fixed else trial.suggest_float("arf_lambda_param", 2.0, 10.0, step=1.0),
                "number_of_jobs": int(fixed.get("number_of_jobs", 1)),
            }
        elif code in {"HAT", "HT"}:
            prefix = code.lower()
            suggested = {
                "grace_period": int(fixed["grace_period"]) if "grace_period" in fixed else trial.suggest_int(f"{prefix}_grace_period", 50, 500, step=50),
                "confidence": float(fixed["confidence"]) if "confidence" in fixed else trial.suggest_float(f"{prefix}_confidence", 1e-7, 1e-2, log=True),
                "tie_threshold": float(fixed["tie_threshold"]) if "tie_threshold" in fixed else trial.suggest_float(f"{prefix}_tie_threshold", 0.01, 0.10),
                "leaf_prediction": fixed.get("leaf_prediction", "NaiveBayesAdaptive"),
            }
        elif code == "NB":
            suggested = {}
        else:
            raise ValueError(f"No search space is registered for classifier: {modelCode}.")
        suggested.update(fixed)
        return suggested
