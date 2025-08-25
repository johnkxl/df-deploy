from typing import Any, Mapping, Optional, Type
from lightgbm import LGBMClassifier, LGBMRegressor

from .base import DeploymentModel


class DeploymentLightGBMEstimator(DeploymentModel):
    shortname = "lgbm"
    longname = "LightGBM Estimator"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.fixed_args = dict(verbosity=-1)
        self.model_cls: Type[Any] = type(None)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args


class DeploymentLightGBMRFEstimator(DeploymentModel):
    shortname = "rf"
    longname = "LightGBM Random Forest Estimator"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.fixed_args = dict(verbosity=-1)
        self.default_args = dict(bagging_freq=1, bagging_fraction=0.75)
        self.model_cls: Type[Any] = type(None)


class DeploymentLightGBMClassifier(DeploymentLightGBMEstimator):
    shortname = "lgbm"
    longname = "LightGBM Classifier"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = True
        self.fixed_args = dict(verbosity=-1)
        self.model_cls = LGBMClassifier


class DeploymentLightGBMRegressor(DeploymentLightGBMEstimator):
    shortname = "lgbm"
    longname = "LightGBM Regressor"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.fixed_args = dict(verbosity=-1)
        self.model_cls = LGBMRegressor


class DeploymentLightGBMRFClassifier(DeploymentLightGBMRFEstimator):
    shortname = "rf"
    longname = "LightGBM Random Forest Classifier"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = True
        self.fixed_args.update(dict(boosting_type="rf", verbosity=-1))
        self.model_cls = LGBMClassifier


class DeploymentLightGBMRFRegressor(DeploymentLightGBMRFEstimator):
    shortname = "rf"
    longname = "LightGBM Random Forest Regressor"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.fixed_args.update(dict(boosting_type="rf", verbosity=-1))
        self.model_cls = LGBMRegressor
