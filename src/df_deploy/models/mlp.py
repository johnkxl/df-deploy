import numpy as np
from pandas import DataFrame, Series
from typing import Any, Mapping, Optional

from skorch import NeuralNetClassifier, NeuralNetRegressor
from torch.nn import CrossEntropyLoss, HuberLoss
from torch.optim import AdamW

from df_analyze.models.mlp import SkorchMLP

from .base import DeploymentModel


class DeploymentMLPEstimator(DeploymentModel):
    shortname = "mlp"
    longname = "Multilayer Perceptron"
    
    def __init__(
        self,
        num_classes: int,
        model_args: Optional[Mapping] = None,
        filtered_columns: Optional[list[str]] = None,
        selection: Optional[str] = None,
        embed_selector: Optional[str] = None,
    ) -> None:
        super().__init__(model_args, filtered_columns, selection, embed_selector)

        self.is_classifier = num_classes > 1
        self.model_cls = NeuralNetClassifier if self.is_classifier else NeuralNetRegressor
        self.model: NeuralNetClassifier | NeuralNetRegressor

        self.fixed_args = dict(
            module=SkorchMLP,
            module__num_classes=num_classes,
            criterion=CrossEntropyLoss if self.is_classifier else HuberLoss,
            optimizer=AdamW,
            max_epochs=50,
            batch_size=128,
            device="cuda",
            verbose=0,
        )

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args
    
    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        if self.model is None:
            kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
            self.model = self.model_cls_args(kwargs)[0](**kwargs)
        
        if self.filtered_columns:
            X_train = self.apply_filter(X_train)
        
        # Converting to numpy array maintains order, but loses column names; So keep track of column order.
        self.feature_order = list(X_train.columns)
        X_train = self._convert_preds(X_test)
        y_train = y_train.to_numpy().astype(np.int64 if self.is_classifier else np.float32)

        self.model.fit(X_train, y_train)

        self.is_fitted = True
    
    def predict(self, X: DataFrame) -> Any:
        if self.filtered_columns:
            X = self.apply_filter(X)
        X = self._convert_preds(X_test)
        return self.model.predict(X)
    
    def predict_proba(self, X: DataFrame):
        if self.filtered_columns:
            X = self.apply_filter(X)
        if hasattr(self.model, "predict_proba") and self.is_classifier:
            X = self._convert_preds(X_test)
            return self.model.predict_proba(X)
        raise AttributeError(f"{self.model.__class__.__name__} does not support predict_proba")
    
    def _convert_preds(self, X: DataFrame) -> np.ndarray:
        """
        Convert DataFrame into numpy array for acceptable input format for NNs.
        """
        return X.loc[:, self.feature_order].to_numpy().astype(np.float32)


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    
    from df_deploy.preprocessing import DataProcessor
    from df_deploy.predicting.utils import ModelEvaluator
    
    data_path = Path("/home/jkendall/projects/model-deployment/test_data/adult")
    deployed_path = data_path / "deploy_adult_test/Aug_01_2025_213422"
    processor_path = deployed_path / "processor/processor.pkl"
    processor = DataProcessor.from_file(processor_path)

    train_df = pd.read_csv(data_path / "adult_train_50.csv")
    test_df = pd.read_csv(data_path / "adult_valid_50.csv")

    X_train, y_train = processor.transform_df(train_df, target="class")
    X_test, y_test = processor.transform_df(test_df, target="class")

    tuned_hparams = {
        "module__width": 128,
        "module__depth": 5,
        "module__use_bn": True,
        "module__dropout": 0.3,
        "optimizer__lr": 1e-3,
        "optimizer__weight_decay": 1e-4,
    }
    model = DeploymentMLPEstimator(
        num_classes=2,
        model_args=tuned_hparams,
        # filtered_columns=filtered_feats,  # optional
    )
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator([model], processor=processor)
    evaluator.evaluate(X_test, y_test)
    evaluator.save(Path(__file__).parent / "test/mlp")

    # preds = model.predict(X_test)
    # probs = model.predict_proba(X_test)






