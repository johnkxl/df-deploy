from typing import Any, Mapping, Optional
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor

from .base import DeploymentModel


class DeploymentDummyEstimator(DeploymentModel):
    shortname = "dummy-est"
    longname = "Dummy Estimator"


class DeploymentDummyRegressor(DeploymentDummyEstimator):
    shortname = "dummy"
    longname = "Dummy Regressor"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.model_cls = SklearnDummyRegressor
        self.fixed_args = dict()
        self.grid = {
            "strategy": ["mean", "median"],
        }
    
    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args


class DeploymentDummyClassifier(DeploymentDummyEstimator):
    shortname = "dummy"
    longname = "Dummy Classifier"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = True
        self.model_cls = SklearnDummyClassifier
        self.fixed_args = dict()
        self.grid = {"strategy": ["most_frequent", "prior", "stratified", "uniform"]}
    
    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args
