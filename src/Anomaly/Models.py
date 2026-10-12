from ProjectDefaults import getAnomalyModelDefaults
from src.Models.LearnerRegistry import LearnerDefinition, LearnerRegistry


class ModelRegistry(LearnerRegistry):
    capymoaModule = "anomaly"
    learnerKind = "anomaly detector"
    definitions = {
        "AIF": LearnerDefinition(
            code="AIF",
            displayName="AdaptiveIsolationForest",
            className="AdaptiveIsolationForest",
            defaults=getAnomalyModelDefaults("AIF"),
            seedParameter="seed",
        ),
        "HST": LearnerDefinition(
            code="HST",
            displayName="HalfSpaceTrees",
            className="HalfSpaceTrees",
            defaults=getAnomalyModelDefaults("HST"),
            seedParameter="random_seed",
        ),
        "AE": LearnerDefinition(
            code="AE",
            displayName="Autoencoder",
            className="Autoencoder",
            defaults=getAnomalyModelDefaults("AE"),
            seedParameter="random_seed",
        ),
        "SRHF": LearnerDefinition(
            code="SRHF",
            displayName="StreamRHF",
            className="StreamRHF",
            defaults=getAnomalyModelDefaults("SRHF"),
            seedParameter="random_seed",
        ),
    }
