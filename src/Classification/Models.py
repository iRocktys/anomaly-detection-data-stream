from ProjectDefaults import getClassifierDefaults
from src.Models.LearnerRegistry import LearnerDefinition, LearnerRegistry


class ClassifierRegistry(LearnerRegistry):
    capymoaModule = "classifier"
    learnerKind = "classifier"
    definitions = {
        "ARF": LearnerDefinition(
            code="ARF",
            displayName="AdaptiveRandomForestClassifier",
            className="AdaptiveRandomForestClassifier",
            defaults=getClassifierDefaults("ARF"),
            seedParameter="random_seed",
        ),
        "HAT": LearnerDefinition(
            code="HAT",
            displayName="HoeffdingAdaptiveTree",
            className="HoeffdingAdaptiveTree",
            defaults=getClassifierDefaults("HAT"),
            seedParameter="random_seed",
        ),
        "HT": LearnerDefinition(
            code="HT",
            displayName="HoeffdingTree",
            className="HoeffdingTree",
            defaults=getClassifierDefaults("HT"),
            seedParameter="random_seed",
        ),
        "NB": LearnerDefinition(
            code="NB",
            displayName="NaiveBayes",
            className="NaiveBayes",
            defaults=getClassifierDefaults("NB"),
            seedParameter="random_seed",
        ),
    }
