import json
import pandas as pd

from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentTypeError, Namespace
from datetime import datetime
from pandas import DataFrame, Series
from pathlib import Path
from typing import Any, Literal

from sklearn.preprocessing import RobustScaler

from df_analyze.enumerables import ClassifierScorer, RegressorScorer, NanHandling, Normalization

from df_deploy.feature_selection import all_feature_selections
from df_deploy.loading import load_df


class DeploymentOptions:
    def __init__(self, args: Namespace):
        self.args = args

        self.df_path: Path = args.df
        self.reports_path: Path = args.results_dir
        self.holdout_path: Path = args.holdout_df
        self.outdir: Path = args.out

        self.cls_metric: ClassifierScorer = ClassifierScorer.from_arg(args.cls_metric)
        self.reg_metric: RegressorScorer = RegressorScorer.from_arg(args.reg_metric)
        self.ranking_method = args.ranking_method
        self.top: int | Literal["all"] = args.top
        self.excluded: list[str] = args.exclude
        # self.extra_params: dict = args.extra_params

        self._validate_paths()

        options: dict[str, Any] = json.load(open(self.reports_path / "options.json"))
        self.target: str = options["target"]
        self.norm: Normalization = Normalization.from_arg(options["norm"].get("__value__", "robust"))
        self.nan_handling: NanHandling = NanHandling.from_arg(options["nan_handling"].get("__value__", "median"))

        self.results_dir: Path = self.reports_path / "results"
        final_performances_path: Path = self.results_dir / "performance_long_table.csv"
        tuned_models_path: Path = self.reports_path / "tuning/tuned_models.csv"
        selection_path: Path = self.reports_path / "selection"
        prepared_path: Path = self.reports_path / "prepared"

        info: dict[str, Any] = json.load(open(prepared_path / "info.json"))
        self.is_classification: bool = info["is_classification"]
        self.metric: ClassifierScorer | RegressorScorer = self.cls_metric if self.is_classification else self.reg_metric

        timestamp = datetime.now().strftime("%b_%d_%Y_%H%M%S")
        self.program_dir = self.outdir / timestamp
        self.models_dir = self.program_dir / "trained_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        names = ["X", "X_cat", "X_cont", "y", "labels"]
        self.prepared_data = {
            name: pd.read_parquet(prepared_path / f"{name}.parquet")
            for name in names
        }
        self.X: DataFrame = self.prepared_data["X"]
        self.X_cat: DataFrame = self.prepared_data["X_cat"]
        self.X_cont: DataFrame = self.prepared_data["X_cont"]
        self.y: Series = self.prepared_data["y"][self.target]
        self.labels: dict | None = self.prepared_data["labels"][0].to_dict()

        self.scaler: RobustScaler = None
        if not self.is_classification:
            df = load_df(self.df_path)
            y = df[self.target]
            # same scaler as in df_analyze.preprocessing.cleaning:433
            self.scaler = RobustScaler(quantile_range=(2.5, 97.5))
            self.scaler.fit_transform(y.to_numpy().reshape(-1, 1))

        final_performances: DataFrame = load_df(final_performances_path)
        final_performances['holdout_average'] = (
            (final_performances['holdout'] + final_performances['5-fold']) / 2 
        )
        final_performances = final_performances[
            ~final_performances['model'].isin(self.excluded)
        ].copy()
        self.final_performances: DataFrame = final_performances
        self.tuned_models: DataFrame = load_df(tuned_models_path)
        
        self.feature_selections: dict[str, list[str]] = all_feature_selections(selection_path)

        loading_str: str
        if isinstance(self.top, int):
            loading_str = (
                f"Loading top {str(self.top) + " " if self.top > 1 else ""}"
                f"model{"s" if self.top > 1 else ""} with "
                f"{"highest" if self.metric.higher_is_better() else "lowest"} "
                f"{self.metric.value}..."
            )
        else:
            loading_str = f"Loading all models..."
        self.loading_str: str = loading_str

        embedding_model_path: Path = self._validate_embedding_model(args.embedding_model)
        self.embedding_model_path = None
        if embedding_model_path is not None:
            import shutil
            self.embedding_model_path = Path(
                shutil.copytree(embedding_model_path, self.program_dir / "embedding_model")
            )

        if self.is_classification and self.cls_metric is None:
            raise ValueError("Classification tasks require a valid metric to be ranked.")
        if not self.is_classification and self.reg_metric is None:
            raise ValueError("Regression tasks require a valid metric to be ranked.")
        
    @staticmethod
    def _validate_embedding_model(embed_path: Path) -> Path:
        if embed_path is None:
            return None

        if not embed_path.exists():
            raise(FileNotFoundError("The embedding model directory does not exist."))
        
        embed_arg_file = embed_path / "cli_arguments.json"
        model_weights = embed_path / "model_weights.pth"
        if not all([embed_arg_file.exists(), model_weights.exists()]):
            raise(FileNotFoundError(
                "The embedding directory does not contain the required files:"
                f"\n\t{embed_arg_file.name}\n\t{model_weights.name}"
            ))
        
        from df_deploy.embedding.embed import DeepTuneOpts
        try:
            DeepTuneOpts.from_traindir(embed_path)
        except:
            raise(OSError(f"Received unexpected parameters in {embed_path / "cli_arguments.json"}"))
            # print(f"Received unexpected parameters in {embed_path / "cli_arguments.json"}")
            # print("Unable to include embedding in deployable models, but embedding can be performed prior to inference.")
            # print("Proceeding...")
            # embed_path = None
        return embed_path

    def _validate_paths(self):
        required = [
            self.df_path,
            self.reports_path / "options.json",
            self.reports_path / "prepared/info.json",
            self.reports_path / "prepared/labels.parquet",
            self.reports_path / "prepared/X_cat.parquet",
            self.reports_path / "prepared/X_cont.parquet",
            self.reports_path / "prepared/y.parquet",
            self.reports_path / "prepared/X.parquet",
            self.reports_path / "results/performance_long_table.csv",
            self.reports_path / "selection",  # check if this generates if no selection 
            self.reports_path / "tuning/tuned_models.csv",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")


def setup_program() -> DeploymentOptions:
    parser = make_parser()
    args = parser.parse_args()
    return DeploymentOptions(args)


CLASSIFICATION_MODEL_NAMES = ["knn", "lgbm", "rf", "lr", "sgd", "mlp", "dummy"]
REGRESSION_MODEL_NAMES = ["knn", "lgbm", "rf", "elastic", "sgd", "mlp", "dummy"]

ALL_MODEL_NAMES = sorted(set(CLASSIFICATION_MODEL_NAMES + REGRESSION_MODEL_NAMES))


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="df-deploy",
        # usage=USAGE_STRING,
        formatter_class=RawTextHelpFormatter,
        # epilog=USAGE_EXAMPLES,
    )
    # Required core args
    parser.add_argument(
        "--df",
        type=Path,
        required=True,
        help=DF_DEPLOY_HELP_STR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=RESULTS_DIR_HELP_STR,
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=DEPLOY_OUT_HELP_STR,
    )
    # Model selection + filtering
    parser.add_argument(
        "--cls-metric",
        choices=ClassifierScorer.choices(),
        type=ClassifierScorer.parse,
        default=ClassifierScorer.default(),
        help=CLS_METRIC_HELP_STR,
    )
    parser.add_argument(
        "--reg-metric",
        choices=RegressorScorer.choices(),
        type=RegressorScorer.parse,
        default=RegressorScorer.default(),
        help=REG_METRIC_HELP_STR,
    )
    parser.add_argument(
        "--top",
        type=parse_top_k,
        default=1,
        help=TOP_N_HELP_STR,
    )
    parser.add_argument(
        "--ranking-method",
        type=str,
        choices=["holdout", "5-fold", "holdout_average"],
        default="holdout_average",
        help=RANK_MTHD_HELP_STR,
    )
    # parser.add_argument(
    #     "--extra-params",
    #     type=json.loads,
    #     help="",
    # )
    parser.add_argument(
        "--exclude",
        nargs="*",
        choices=ALL_MODEL_NAMES,
        default=[],
        help=EXCLUDE_HELP_STR,
    )
    # Advanced / optional args
    parser.add_argument(
        "--holdout-df",
        type=Path,
        required=False,
        help=HOLDOUT_HELP_STR,
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        help=DEEPTUNE_TRAIN_HELP_STR,
    )

    return parser


def parse_top_k(value):
    if value == "all":
        return value
    try:
        k = int(value)
        if k <= 0:
            raise ArgumentTypeError("Top k must be a positive integer or 'all'")
        return k
    except ValueError:
        raise ArgumentTypeError("Top k must be a positive integer or 'all'")


DF_DEPLOY_HELP_STR = """

Dataset used in the original df-analyze run to generate model candidates.

"""

RESULTS_DIR_HELP_STR = """
Path of directory containing the output produced by df-analyze to be used to obtain trained models.

"""

CLS_METRIC_HELP_STR = f"""
Classification performance metric to determine the top model(s). 
Default is '{ClassifierScorer.default().value}' if unspecified.

"""

REG_METRIC_HELP_STR = f"""
Regression performance metric to determine the top model(s).
Default is '{RegressorScorer.default().value}' if unspecified.

"""

TOP_N_HELP_STR = """
How many of the top models to train. Integer or 'all'. Default is '1' if unspecified.

"""

RANK_MTHD_HELP_STR = """
Choose the ranking method.

  holdout           Models are sorted based on performances on the holdout set.
  
  5-fold            Models are sorted based on performances in 5-fold validation on the holdout set.
  
  holdout_average   Models are sorted based on the mean of their performances on the holdout set and 
                    5-fold validation on the holdout set (The mean of the other 2 options). Default option.

"""

# EXTRA_PARAMS_HELP_STR = """
# A dictionary of extra parameters to include in models trained. (Disabled)

# """

EXCLUDE_HELP_STR = f"""
(optional) Exclude model types by name. This could be useful to avoid models that are slow in prediction.

    Options: {', '.join(ALL_MODEL_NAMES)}

"""

DEPLOY_OUT_HELP_STR = """
Specifies location of output including data processor and model artifacts, 
model info, and model evaluation if a holdout set is provided.

"""

HOLDOUT_HELP_STR = """
Path of dataset split not seen by df-analyze to produce the results used for training the models.
The predictors and target should have the same structure as the data split input to df-analyze.

"""

DEEPTUNE_TRAIN_HELP_STR = """
Used only when data was embedded using a DeepTune model before input to df-analyze.

Path of directory produced by training a deep learner with DeepTune on the training and validation
splits of the dataset, where the test split was passed to df-analyze.

    Structure of directory:

            [deeptune_train_dir]/
            ├── cli_arguments.json
            ├── model_weights.pth
            └── ...


"""
