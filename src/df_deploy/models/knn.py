from typing import Any, Mapping, Optional, Type
import platform
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .base import DeploymentModel


class DeploymentKNNEstimator(DeploymentModel):
    shortname = "knn"
    longname = "K-Neighbours Estimator"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        n_jobs = 1 if platform.system().lower() == "darwin" else -1
        self.is_classifier = False
        self.needs_calibration = False
        self.fixed_args = dict(n_jobs=n_jobs)
        self.model_cls: Type[Any] = type(None)
        self.grid = {
            "n_neighbors": [1, 5, 10, 25, 50],
            "weights": ["uniform", "distance"],
            "metric": ["cosine", "l2", "correlation"],
        }

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, {}


class DeploymentKNNClassifier(DeploymentKNNEstimator):
    shortname = "knn"
    longname = "K-Neighbours Classifier"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = True
        self.model_cls = KNeighborsClassifier


class DeploymentKNNRegressor(DeploymentKNNEstimator):
    shortname = "knn"
    longname = "K-Neighbours Regressor"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.model_cls = KNeighborsRegressor
        self.shortname = "knn"
        self.longname = "K-Neighbours Regressor"
