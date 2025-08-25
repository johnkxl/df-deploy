import joblib
import json
import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from typing import Any, Mapping

from df_analyze.enumerables import NanHandling
from df_analyze.preprocessing.cleaning import sanitize_names, unify_nans

from df_deploy.saving import save_json


class DataProcessor:
    def __init__(
            self,
            prepared_columns: list[str] = None,
            continuous_features: list[str] = None,
            imputer: SimpleImputer | IterativeImputer = None,
            cat_vals: dict[str,Any] = None,
            is_classification: bool = True,
            labels: dict[int,str] | None = None,
            scaler: RobustScaler = None,
    ):
        self.prepared_columns = prepared_columns
        self.continuous_features = continuous_features
        self.imputer = imputer
        self.cat_vals = cat_vals
        self.is_classification = is_classification

        self.labels = labels
        self.inverse_labels = {label: col for col, label in labels.items()} if labels else None
        self.scaler = scaler

        if is_classification:
            if labels is None:
                raise AttributeError("Classification tasks must have a label mapping.")
        else:
            if scaler is None:
                raise AttributeError("Regression tasks must have a target vaariable scaler/normaliser.")

    @classmethod
    def from_X_prepared_cat_cont(
        cls,
        X: DataFrame,
        X_cat: DataFrame,
        X_cont: DataFrame,
        is_classification: bool = True,
        nan_strategy: NanHandling = NanHandling.Median,
        labels: Mapping[int,Any] = None,
        scaler: RobustScaler = None,
    ) -> "DataProcessor":
        """
        Return a DataProcessor object from the full DataFrame, X, and dataframes of categorical and continuous predictors, X_cat and X_cont.

        Parameters
        ----------
        X: DataFrame
            The fully encoded training features (one-hot encoded and so on) to 
            identify the columns needed.

        X_cat: DataFrame
            The categorical variables remaining after processing (no encoding,
            for univariate metrics and the like).

        X_cont: DataFrame
            The continues variables remaining after processing (no encoding,
            for univariate metrics and the like).

        is_classification: bool, default true
            ...

        nan_strategy: NanHandling, default NanHandling.Median
            ...

        labels: Mapping, optional
            Label dict mapping int to actual labels.

        scaler: RobustScaler, optional
            Scaler pre-fit to y_true.

        Returns
        -------
        processor: DataProcesor
        """
        prepared_columns = list(X.columns)

        categorical_features = list(X_cat.columns)
        continuous_features = list(X_cont.columns)

        X_cat = unify_nans(X_cat)
        cat_vals = {c: list(X_cat[c].unique()) for c in categorical_features}

        imputer = cls._get_imputer(nan_strategy)
        imputer.fit(X_cont)

        return DataProcessor(
            prepared_columns=prepared_columns,
            continuous_features=continuous_features,
            imputer=imputer,
            cat_vals=cat_vals,
            is_classification=is_classification,
            labels=labels,
            scaler=scaler,
        )

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame | tuple[DataFrame,Series]:
        """
        Convert dataframe into the structure recognised by the trained `DeploymentModel`. 
        Preprocessing step.

        Parameters
        ----------
        X: DataFrame
            The dataset containing only feature columns.

        y: Series (Optional)
            The ground truth labels in the dataset.

        Returns
        -------
        X_encoded: DataFrame
            All encoded and processed predictors.

        y: Series (Optional)
            The regression or classification target.

        """
        X_cat = X[list(self.cat_vals.keys())].copy()#.astype(str)
        X_cont_raw = X[self.continuous_features].copy()

        if not X_cat.empty:
            X_cat = unify_nans(X_cat)
            X_cat = self._deflate_categoricals(X_cat, self.cat_vals)

            # one-hot encode categorical features
            X_cat = pd.get_dummies(X_cat, dtype=float, dummy_na=True)

        if X_cont_raw.empty:
            X_cont = X_cont_raw
        else:
            if not hasattr(self, "imputer") or self.imputer is None:
                raise AttributeError("Imputer has not been fitted. Use from_X_prepared_cat_cont() first.")
            X_cont = DataFrame(data=self.imputer.transform(X_cont_raw), columns=X_cont_raw.columns)

        # X_cont = normalize_continuous(X_cont, robust=True)

        X_encoded = pd.concat([X_cat, X_cont], axis=1)

        X_encoded = X_encoded[self.prepared_columns]

        if y is None:
            return X_encoded

        # endocde target labels
        if self.inverse_labels:
            y = y.map(self.inverse_labels)

        return X_encoded, y

    def transform_df(self, df: DataFrame, target: str) -> tuple[DataFrame, Series | None]:
        """
        Convert dataframe into the structure recognised by the trained `DeploymentModel`. 
        Preprocessing step.

        Parameters
        ----------
        df: DataFrame
            The dataset containing only feature columns.

        target: str, optional
            The name of the target column in the dataset.

        Returns
        -------
        X_encoded: DataFrame
            All encoded and processed predictors.

        y: Series or None
            The regression or classification target, if it exists.

        """
        y = df.get(target, None)
        df = df.drop(columns=target, errors="ignore")

        X_cat = df[list(self.cat_vals.keys())].copy()
        X_cont_raw = df[self.continuous_features].copy()

        if not X_cat.empty:
            X_cat = unify_nans(X_cat)
            X_cat = self._deflate_categoricals(X_cat, self.cat_vals)
            X_cat = pd.get_dummies(X_cat, dtype=float, dummy_na=True)

        if X_cont_raw.empty or X_cont_raw.isna().sum().sum() == 0:
            X_cont = X_cont_raw
        else:
            if not hasattr(self, "imputer") or self.imputer is None:
                raise AttributeError("Imputer has not been fitted. Use from_X_prepared_cat_cont() first.")

            X_cont = DataFrame(data=self.imputer.transform(X_cont_raw), columns=X_cont_raw.columns)

        # X_cont = normalize_continuous(X_cont, robust=True)

        X_encoded = pd.concat([X_cat, X_cont], axis=1)
        X_encoded = X_encoded[self.prepared_columns]

        # endocde target labels
        if y is not None and self.inverse_labels:
            y = y.map(self.inverse_labels)

        return X_encoded, y

    def remap_preds(self, y: Series | None) -> Series | None:
        """
        Convert predictions into actual value scale (regression) or labels (classification).
        """
        if y is None:
            return None

        y = y.copy()
        y_remapped: Series
        if self.is_classification:
            y_remapped = y.map(self.labels)
        else:
            y_remapped = self.scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        return y_remapped

    def save(self, filename: Path | str) -> None:
        joblib.dump(self, filename)

    def save_all(self, savedir: Path) -> None:
        savedir.mkdir(parents=True, exist_ok=True)
        self.save(savedir / "processor.pkl")
        self.save_dict(savedir / "processor_dict.json")
        joblib.dump(self.imputer, savedir / "imputer.pkl")

    def save_dict(self, filename: Path | str) -> None:
        """
        Save DataProcessor attributes as a dictionary such that the DataProcessor object can be replicated from the dictionary using the `from_dict` classmethod.
        """
        save_json(self.to_dict(), filename)

    def to_dict(self) -> dict[str, Any]:
        arg_dict = {}

        arg_dict["prepared_columns"] = self.prepared_columns
        arg_dict["continuous_features"] = self.continuous_features
        arg_dict["imputer"] = None  # not JSON serialisable
        arg_dict["cat_vals"] = self.cat_vals
        arg_dict["is_classification"] = self.is_classification
        arg_dict["labels"] = self.labels

        if self.scaler is not None:
            arg_dict["scaler"] = {
                "type": self.scaler.__class__.__name__,
                "params": self.scaler.get_params(),
                "center_": self.scaler.center_.tolist(),  # this only really applies to RobustScaler, but only this scaler seems to get applied to y column
                "scale_": self.scaler.scale_.tolist()
            }
        else:
            arg_dict["scaler"] = None

        return arg_dict

    @staticmethod
    def _deflate_categoricals(X: DataFrame, allowed_values_dict: dict[str, list]) -> DataFrame:
        """
        Replace unseen or missing values in categorical columns with "DEFLATED".

        Parameters
        ----------
        X: DataFrame
            The DataFrame of categorical feature columns.

        allowed_values_dict: dict[str,Any]
            A dict mapping column names to sets/lists of allowed values.

        Returns
        -------
            A new DataFrame with the same structure and categorical columns modified.
        """
        X = X.copy()
        X = X.fillna(np.nan)

        for col, allowed_values in allowed_values_dict.items():
            X[col] = X[col].astype(str).where(X[col].notna(), np.nan)

            # Normalize allowed_values: convert to set of strings
            allowed_values_str = set(str(val) for val in allowed_values if not pd.isna(val))
            allow_nan = any(pd.isna(val) for val in allowed_values)

            not_in_allowed = ~X[col].isin(allowed_values_str)

            if allow_nan:
                mask = not_in_allowed & X[col].notna()
            else:
                mask = not_in_allowed | X[col].isna()

            X.loc[mask, col] = "DEFLATED"

        return X

    @classmethod
    def from_file(cls, filepath: Path | str) -> "DataProcessor":
        """
        Load the fitted DataProcessor from a pickle file.
        """
        return joblib.load(filepath)

    @classmethod
    def from_dict(cls, argdict: dict) -> "DataProcessor":
        scaler_dict = argdict.get("scaler")
        if scaler_dict is not None:
            if argdict.get("scaler")["type"] != "RobustScaler":
                raise NotImplementedError("Scalers other that 'RobustScaler' for y column not implemented yet.")
            scaler = RobustScaler(**scaler_dict["params"])
            scaler.center_ = np.array(scaler_dict["center_"])
            scaler.scale_ = np.array(scaler_dict["scale_"])

            argdict["scaler"] = scaler

        return cls(**argdict)

    @classmethod
    def from_json_pkl(cls, jsonfile: Path, imputer_file: Path) -> "DataProcessor":
        argdict = json.load(open(jsonfile, 'r'))
        processor = cls.from_dict(argdict)
        imputer = joblib.load(imputer_file)
        processor.imputer = imputer
        return processor

    @staticmethod
    def _get_imputer(strategy: NanHandling) ->  SimpleImputer | IterativeImputer:
        if strategy == NanHandling.Mean:
            return SimpleImputer(strategy="mean", keep_empty_features=True)
        elif strategy == NanHandling.Median:
            return SimpleImputer(strategy="median", keep_empty_features=True)
        elif strategy == NanHandling.Impute:
            return IterativeImputer(verbose=2, keep_empty_features=True)
        else:
            raise ValueError(f"Unknown nan_strategy: {strategy}")


def sanitize_column_names(df: DataFrame, target: str) -> DataFrame:
    temp_df = df.copy()
    no_target = target not in temp_df.columns

    if no_target:
        temp_df[target] = 0

    cleaned_df, _ = sanitize_names(temp_df, target)

    if no_target:
        cleaned_df = cleaned_df.drop(columns=target)

    return cleaned_df