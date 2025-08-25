from typing import Any, Mapping, Optional
from copy import deepcopy
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

from .base import DeploymentModel


class DeploymentSVMEstimator(DeploymentModel):
    shortname = "svm"
    longname = "Support Vector Machine"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = True
        self.fixed_args = dict(cache_size=1000)
        self.default_args = dict(kernel="rbf")

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        is_rbf = full_args["kernel"]
        if self.is_classifier:
            model_cls = SVC if is_rbf else LinearSVC
        else:
            model_cls = SVR if is_rbf else LinearSVR

        args = deepcopy(full_args)
        if not is_rbf:
            args.pop("cache_size")
            args.pop("gamma")
            args.pop("kernel")

        return model_cls, args


class DeploymentSVMClassifier(DeploymentSVMEstimator):
    shortname = "svm"
    longname = "Support Vector Classifier"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = True
        self.needs_calibration = True
        self.model_cls = SVC


class DeploymentSVMRegressor(DeploymentSVMEstimator):
    shortname = "svm"
    longname = "Support Vector Regressor"

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)
        self.is_classifier = False
        self.model_cls = SVR