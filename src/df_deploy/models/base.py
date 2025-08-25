import joblib

from abc import ABC
from numpy import ndarray
from pandas import DataFrame, Series
from pathlib import Path
from typing import Any, Mapping, Optional, Type


def str_to_None(s: str) -> str | None:
    """
    Returns None if the input string equals 'none' (case-insensitive), else returns the original string.
    """
    if s is None:
        return None
    
    return None if s.lower() == "none" else s


# The base class and model class implementations are based on the model classes implemented in `df-analyze`.

class DeploymentModel(ABC):
    shortname: str = ""
    longname: str = ""

    def __init__(self, model_args: Optional[Mapping] = None, filtered_columns: list[str] = None, selection: str = None, embed_selector: str = None) -> None:
        super().__init__()  # Only necessary if I use @abstractmethod
        self.is_classifier: bool = True
        self.model_cls: type = None  # only in inheriting classes
        self.model: Optional[Any] = None
        self.fixed_args: dict[str, Any] = {}
        self.default_args: dict[str, Any] = {}
        self.model_args: Mapping = model_args or {}
        self.selection: str | None = str_to_None(selection)
        self.embed_selector: str | None = str_to_None(embed_selector)
        self.filtered_columns: list[str] = filtered_columns
        self.is_fitted: bool = False

        self.old_filter = filtered_columns
    
    @property
    def full_args(self) -> Mapping:
        return {**self.fixed_args, **self.default_args, **self.model_args}

    @property
    def feature_selection(self) -> str:
        if self.selection is None:
            return "no_select"

        return "_".join(filter(None, [self.selection, self.embed_selector]))
    
    @property
    def fullname(self) -> str:
        return f"{self.shortname}_{self.feature_selection}"

    @property
    def table_selection(self) -> str:
        if self.selection is None:
            return "none"
        if self.selection == "embed" and self.embed_selector:
            return f"embed_{self.embed_selector}"
        return self.selection

    def apply_filter(self, X: DataFrame) -> DataFrame:
        # filter-based feature selections are a bit different:
        # pred columns are original column names before one-hot-encoding
        # assoc columns are both original cols and suffixed by "__target.target_class" for each target class (classification)

        if not self.is_fitted:
            # Match filter to columns in X_train.
            # During prediction, filtered_columns will already be correct
            new_filter = set()
            X_cols = list(X.columns)
            for fcol in self.filtered_columns:
                # Extract base column name before any '__'
                base_colname = fcol.split("__")[0]

                # this captures all columns derived from base_colname, if base_colname is an original column name
                matches = set([xcol for xcol in X_cols if xcol.startswith(base_colname)])
                new_filter |= matches  # union 

                # for xcol in X_cols:
                #     if xcol.startswith(fcol):
                #         new_filter.add(xcol)

            self.filtered_columns = list(new_filter)

        return X[self.filtered_columns]

    # @abstractmethod
    def model_cls_args(
        self, full_args: dict[str, Any]
    ) -> tuple[Type[Any], dict[str, Any]]:
        """Allows for conditioning the model based on args (e.g. SVC vs. LinearSVC
        depending on kernel, and also subsequent removal or addition of necessary
        args because of this.

        Returns
        -------
        model_cls: Type[Any]
            The model class needed based on `full_args`

        clean_args: dict[str, Any]
            The args that now work for the returned `model_cls`

        """
        return self.model_cls, full_args

    def setup_model(self):
        kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
        self.model = self.model_cls_args(kwargs)[0](**kwargs)

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        if self.model is None:
            kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
            self.model = self.model_cls_args(kwargs)[0](**kwargs)
        
        if self.filtered_columns:
            X_train = self.apply_filter(X_train)

        self.model.fit(X_train, y_train)

        self.is_fitted = True

    @classmethod
    def from_file(cls, filepath: Path | str) -> "DeploymentModel":
        """
        Load a trained DeploymentModel from a pickle file.
        """
        return joblib.load(filepath)
    
    def save_to_file(self, filepath: Path | str) -> None:
        """
        Save DeploymentModel instance to disk.
        """
        joblib.dump(self, filepath)

    def predict(self, X: DataFrame) -> Any:
        if self.filtered_columns:
            X = self.apply_filter(X)
        return self.model.predict(X)

    def predict_proba(self, X: DataFrame) -> ndarray:
        if self.filtered_columns:
            X = self.apply_filter(X)
        if hasattr(self.model, "predict_proba") and self.is_classifier:
            return self.model.predict_proba(X)
        raise AttributeError(f"{self.model.__class__.__name__} does not support predict_proba")

