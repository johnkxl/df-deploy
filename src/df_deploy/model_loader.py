import json
import numpy as np

from dataclasses import dataclass
from pandas import DataFrame
from typing import Literal, Any

from df_analyze.enumerables import Scorer

from df_deploy.model_registry import get_model_cls
from df_deploy.models import DeploymentModel


@dataclass
class ModelDescriptor:
    model: str
    selection: str
    embed_selector: str


class ModelLoader:
    def __init__(
        self,
        performance_table: DataFrame,
        tuned_models: DataFrame,
        feature_selections: dict[str, list[str]],
        is_classification: bool = True,
    ):
        self.performance_table = performance_table
        self.tuned_models = tuned_models
        self.feature_selections = feature_selections
        self.is_classification = is_classification

    def get_best_descriptors(
        self,
        measure: str | Scorer,
        eval_method: str,
        n: int | Literal["all"] = 1,
    ) -> list[ModelDescriptor]:
        return best_model_selection_embed(
            self.performance_table, measure, eval_method, n
        )

    def load_model(
        self,
        descriptor: ModelDescriptor,
        extra_params: dict[str, Any] = None,
    ) -> DeploymentModel:
        params = get_tuned_params(self.tuned_models, descriptor)
        if extra_params:
            params.update(extra_params)

        filtered_cols = self.feature_selections.get(descriptor.selection)
        model_cls = get_model_cls(descriptor.model, self.is_classification)
        return model_cls(params, filtered_cols, descriptor.selection, descriptor.embed_selector)

    def load_top_models(
        self,
        measure: str | Scorer,
        eval_method: str,
        n: int | Literal["all"] = 1,
        extra_params: dict[str, Any] = None,
    ) -> list[DeploymentModel]:
        descriptors = self.get_best_descriptors(measure, eval_method, n)
        return [self.load_model(desc, extra_params) for desc in descriptors]


def best_model_selection_embed(
        performances: DataFrame,
        measure: str | Scorer,
        eval_method: str,
        n: int | Literal["all"] = 1,
) -> list[ModelDescriptor]:
    """
    Return a list of the top `n` models by `measure` from the `eval_method`.
    """
    # Get top model for statistic of interest
    top_rows = get_top_model_rows(performances, measure, eval_method, n)

    descriptors = []

    for row in top_rows.itertuples():
        model = row.model
        selection = row.selection
        embed_selector = row.embed_selector

        if embed_selector.lower() != 'none':
            selection, embed_selector = selection.split("_")
        
        descriptors.append(ModelDescriptor(model, selection, embed_selector))
    
    return descriptors


def get_top_model_rows(
        performance_table: DataFrame,
        measure: str | Scorer,
        eval_method: str,
        n: int | Literal["all"] = 1,
) -> DataFrame:
    """
    Return row of best performing ML+FS for the given measure and evaluation method.

    Parameters
    ----------
    performance_table: DataFrame
        Table with columns:
                              
                              metric | trainset | holdout | 5-fold | model | selection | embed_selector
                              
    measure: str or Scorer (RegressorScorer, ClassifierScorer)
        The metric of interest for determining the best model. (acc, f1, npv, ppv, auroc, ...)

    eval_method: str
        The performance evaluation ranked...
                    
                    Options: "holdout", "5-fold", "holdout_average"

    n: int or 'all'
        The number of models to train.                
    
    Returns
    -------
    top_models: DataFrame
        The rows of the top `n` performing models for the measure.
    
    """
    measure_table: DataFrame
    if isinstance(measure, Scorer):
        measure_table: DataFrame = performance_table[performance_table['metric'] == measure.value]
        measure_table: DataFrame = measure_table.sort_values(by=[eval_method], ascending=(not measure.higher_is_better()))
    else:
        # will likely delete this
        measure_table = performance_table[performance_table['metric'] == measure]
        measure_table = measure_table.sort_values(by=[eval_method], ascending=False)
    
    if n == "all":
        return measure_table

    return measure_table.head(n)


def get_tuned_params(
        tuned_models: DataFrame,
        descriptor: ModelDescriptor,
) -> dict[str, Any]:
    """
    Return the tuned parameters of the model matching the input attributes.
    """
    tuned = tuned_models[
        (tuned_models['model'] == descriptor.model) & 
        (tuned_models['selection'] == descriptor.selection) & 
        (tuned_models['embed_selector'] == descriptor.embed_selector)
    ].iloc[0]
    model_params = json.loads(tuned.params)
    return model_params


def base_performances(
        model: DeploymentModel,
        performances_table: DataFrame,
        eval_method: str = "holdout_average",
) -> dict[str, float]:
    shortname = model.shortname
    selection = model.table_selection
    embed_selector = model.embed_selector or "none"

    performances_table = performances_table.fillna(np.nan)

    tuning_metrics = performances_table[
        (performances_table["model"] == shortname) &
        (performances_table["selection"] == selection) &
        (performances_table["embed_selector"] == embed_selector)
    ]

    return {row["metric"]: row[eval_method] for _, row in tuning_metrics.iterrows()}