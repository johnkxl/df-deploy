from .base import DeploymentModel
from .dummy import DeploymentDummyClassifier, DeploymentDummyRegressor
from .knn import DeploymentKNNClassifier, DeploymentKNNRegressor
from .lgbm import DeploymentLightGBMClassifier, DeploymentLightGBMRegressor
from .lgbm import DeploymentLightGBMRFClassifier, DeploymentLightGBMRFRegressor
from .linear import DeploymentLRClassifier, DeploymentElasticNetRegressor
from .linear import DeploymentSGDClassifier, DeploymentSGDRegressor
from .mlp import DeploymentMLPEstimator
from .svm import DeploymentSVMClassifier, DeploymentSVMRegressor