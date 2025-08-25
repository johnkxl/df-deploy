from typing import Literal, Type

from df_deploy.models import (
    DeploymentModel,
    DeploymentDummyClassifier, DeploymentDummyRegressor,
    DeploymentElasticNetRegressor, DeploymentKNNClassifier,
    DeploymentKNNRegressor, DeploymentLRClassifier,
    DeploymentLightGBMClassifier, DeploymentLightGBMRFClassifier,
    DeploymentLightGBMRFRegressor, DeploymentLightGBMRegressor,
    DeploymentMLPEstimator,
    DeploymentSGDClassifier, DeploymentSGDRegressor,
    DeploymentSVMClassifier, DeploymentSVMRegressor,
)


CLASSIFICATION_MODELS: dict[str, Type[DeploymentModel] | None] = {
    "knn" : DeploymentKNNClassifier,
    "lgbm" : DeploymentLightGBMClassifier,
    "rf" : DeploymentLightGBMRFClassifier,
    "lr" : DeploymentLRClassifier,
    "sgd" : DeploymentSGDClassifier,
    "mlp" : DeploymentMLPEstimator,
    "svm" : DeploymentSVMClassifier,
    "gandalf" : None,  # LOL nope
    "dummy" : DeploymentDummyClassifier,
}
REGRESSION_MODELS: dict[str, Type[DeploymentModel] | None] = {
    "knn" : DeploymentKNNRegressor,
    "lgbm" : DeploymentLightGBMRegressor,
    "rf" : DeploymentLightGBMRFRegressor,
    "elastic": DeploymentElasticNetRegressor,
    # "lr" : DeploymentLinearSVR,
    "sgd" : DeploymentSGDRegressor,
    "mlp" : DeploymentMLPEstimator,
    "svm" : DeploymentSVMRegressor,
    "gandalf" : None,  # LOL nope
    "dummy" : DeploymentDummyRegressor,
}
ClassificationModelName = Literal["knn", "lgbm", "rf", "lr", "sgd", "mlp", "dummy"]  # leave out gandalf for now
RegressionModelName = Literal["knn", "lgbm", "rf", "elastic", "sgd", "mlp", "dummy"]  # leave out gandalf for now


def get_model_cls(
        model_cls_name: ClassificationModelName | RegressionModelName,
        is_classification: bool = True,
) -> Type[DeploymentModel]:
    models = CLASSIFICATION_MODELS if is_classification else REGRESSION_MODELS

    if model_cls_name not in models:
        raise KeyError(f"Model '{model_cls_name}' not found in {'classification' if is_classification else 'regression'} models.")

    model_cls = models[model_cls_name]
    if model_cls is None:
        raise ValueError(f"Model '{model_cls_name}' is not implemented.")

    if not issubclass(model_cls, DeploymentModel):
        print(model_cls)
        raise TypeError(f"Model '{model_cls_name}' must inherit from DeploymentModel.")

    return model_cls