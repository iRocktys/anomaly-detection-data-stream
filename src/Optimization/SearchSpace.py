from typing import Any, Protocol

from src.Optimization.OptimizationConfig import TrialConfiguration


class SearchSpaceProtocol(Protocol):
    def suggest(self, trial: Any) -> TrialConfiguration:
        ...
