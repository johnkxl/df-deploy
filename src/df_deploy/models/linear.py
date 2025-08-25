from typing import Any, Mapping, Optional
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor

from .base import DeploymentModel


class DeploymentElasticNetRegressor(DeploymentModel):
    shortname = "elastic"
    longname = "ElasticNet Regressor"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.model_cls = ElasticNet
        self.fixed_args = dict(max_iter=2000)
    
    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args


class DeploymentLRClassifier(DeploymentModel):
    shortname = "lr"
    longname = "Logistic Regression"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = True
        self.model_cls = LogisticRegression
        self.fixed_args = dict(max_iter=2000, penalty="elasticnet", solver="saga")
        self.default_args = dict(l1_ratio=0.5)
    
    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args


class DeploymentSGDClassifier(DeploymentModel):
    shortname = "sgd"
    longname = "SGD Linear Classifer"

    def __init__(self, model_args: Mapping | None = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = True
        self.model_cls = SklearnSGDClassifier
        self.default_args = dict(learning_rate="adaptive", penalty="l2", eta0=3e-4)
    
    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args


class DeploymentSGDRegressor(DeploymentModel):
    shortname = "sgd"
    longname = "SGD Linear Regressor"

    def __init__(self, model_args: Mapping | None = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.model_cls = SklearnSGDRegressor
        self.default_args = dict(eta0=3e-4)
    
    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args
