import pandas as pd
import re

from collections import OrderedDict
from dataclasses import dataclass
from pandas import DataFrame, Series
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from df_analyze.enumerables import ClassifierScorer, RegressorScorer

from df_deploy.models import DeploymentModel
from df_deploy.preprocessing import DataProcessor, sanitize_column_names
from df_deploy.saving import save_json


def compute_metrics(y_true, y_pred, y_prob=None, is_classifier=True) -> dict[str, float]:
    if y_true is None:
        return None

    metrics = {}

    if is_classifier:
        metrics = ClassifierScorer.get_scores(y_true, y_pred, y_prob)
    else:
        metrics = RegressorScorer.get_scores(y_true, y_pred)

    return metrics


@dataclass
class ModelPredictions(OrderedDict):
    model_name: str
    y_true: Series
    y_pred: Series
    metrics: dict = None
    y_prob: Series = None


class ModelEvaluator:
    def __init__(
        self,
        models: list[DeploymentModel],
        processor: DataProcessor = None,
    ):
        self.models: list[DeploymentModel] = models
        self.processor: DataProcessor = processor

        self.all_preds: dict[str, list] = {}
        self.all_metrics: dict[str, dict[str, float]] = {}
        self.preds_df: Optional[DataFrame] = None
        self.metrics_df: Optional[DataFrame] = None
        self.probabilities: Optional[dict[str, DataFrame]] = {}

    def predict(self, X: DataFrame) -> None:
        self._collect_predictions(X)

    def evaluate(self, X: DataFrame, y_true: Series) -> None:
        self._collect_predictions(X, y_true)

    def _collect_predictions(self, X: DataFrame, y_true: Optional[Series] = None) -> None:
        is_eval = y_true is not None

        for model in tqdm(self.models, desc="Evaluating models" if is_eval else "Predicting"):
            res: ModelPredictions = self._generate_predictions(model, X, y_true)
            model_name = res.model_name

            self.all_preds[model_name] = res.y_pred.tolist()
            if is_eval:
                if "y_true" not in self.all_preds:
                    self.all_preds["y_true"] = res.y_true.tolist()
                self.all_metrics[model_name] = res.metrics

            if res.y_prob is not None:
                class_order = model.model.classes_
                column_labels = [self.processor.labels[c] for c in class_order]
                self.probabilities[model_name] = DataFrame(res.y_prob, columns=column_labels)

        self.preds_df = DataFrame(self.all_preds)
        if is_eval:
            self.metrics_df = DataFrame(self.all_metrics).T.reset_index().rename(columns={"index": "model"})

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.preds_df is not None:
            self.preds_df.to_csv(out_dir / "predictions.csv", index=False)

        if self.metrics_df is not None:
            self.metrics_df.to_csv(out_dir / "metrics.csv", index=False)

        if self.all_metrics:
            save_json(self.all_metrics, out_dir / "metrics.json")

        if self.probabilities:
            proba_dir = out_dir / "predict_proba"
            proba_dir.mkdir(parents=True, exist_ok=True)
            for model_name, df in self.probabilities.items():
                df.to_csv(proba_dir / f"{model_name}_predict_proba.csv", index=False)

        print(f"[✓] Saved all model predictions{" and metrics" if self.all_metrics else ""} to: {out_dir}")

    def _generate_predictions(
        self,
        model: DeploymentModel,
        X: DataFrame,
        y_true: Series,
    ) -> ModelPredictions:
        y_pred = model.predict(X)
        y_prob = None

        if not model.is_classifier:
            y_pred = self.processor.remap_preds(y_pred)

        y_pred_mapped = Series(y_pred)
        y_true_mapped = y_true

        if model.is_classifier:
            y_prob = model.predict_proba(X)
            y_pred_mapped = self.processor.remap_preds(y_pred_mapped)
            y_true_mapped = self.processor.remap_preds(y_true)

        metrics = compute_metrics(y_true, y_pred, y_prob, is_classifier=model.is_classifier)
        model_name = model.fullname

        return ModelPredictions(
            model_name=model_name,
            y_true=y_true_mapped,
            y_pred=y_pred_mapped,
            metrics=metrics,
            y_prob=y_prob,
        )

def compute_performance_deltas(
    model_evaluator: ModelEvaluator,
    models_info: dict,
    ranking_method: str,
) -> dict[str, dict[str, float]]:
    """
    Compute the difference between evaluated and hypothesized performance metrics
    for each model.

    Parameters
    ----------
    model_evaluator : ModelEvaluator
        The evaluator object that has run .evaluate() and contains .all_metrics.

    models_info : dict
        Dictionary mapping model names to their info, including metrics under keys like "holdout", "5-fold", etc.

    ranking_method : str
        The evaluation method from df-analyze to compare performance (e.g., "holdout", "5-fold", or "holdout_average").

    Returns
    -------
    performance_deltas : dict
        A dictionary with model names as keys and a dict of metric deltas as values.
    """
    performance_deltas = {}

    for model_name in model_evaluator.all_metrics:
        hypothesis_metrics = models_info[model_name][ranking_method]
        evaluated_metrics = model_evaluator.all_metrics[model_name]

        delta = {
            metric: round(evaluated_metrics[metric] - hypothesis_metrics[metric], 5)
            for metric in hypothesis_metrics
            if metric in evaluated_metrics
        }

        performance_deltas[model_name] = delta

    return performance_deltas


def predict_evaluate(
    df: DataFrame,
    target: str,
    models: list[DeploymentModel],
    processor: DataProcessor,
    out: Path,
    embed_dir: Path,
) -> ModelEvaluator:
    df = sanitize_column_names(df, target)

    if embed_dir is not None:
        from df_deploy.embedding.embed import embed_df
        df = embed_df(df, embed_dir)

    X, y_true = processor.transform_df(df, target)
    print(f"Ground-truth labels {"not " if y_true is None else ""}found.")

    evaluator = ModelEvaluator(models, processor=processor)

    if y_true is not None:
        evaluator.evaluate(X, y_true)
    else:
        evaluator.predict(X)

    evaluator.save(out)

    display_df = df[[col for col in df.columns if not is_embedding_feature(col)]]
    merged_df = pd.concat([display_df.drop(columns=target, errors="ignore"), evaluator.preds_df], axis=1)
    merged_df.to_csv(out / "predictions_with_inputs.csv", index=False)

    return evaluator


def is_embedding_feature(column_name: str) -> bool:
    return bool(re.fullmatch(r"embed\d+", column_name))
