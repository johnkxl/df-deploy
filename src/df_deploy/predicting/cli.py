import json

from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from datetime import datetime
from pandas import DataFrame
from pathlib import Path

from df_deploy.loading import load_df
from df_deploy.preprocessing import DataProcessor


DF_HELP_STR = """Path to the file containing the super holdout dataset.

"""
MODEL_DIR_HELP_STR = """
Path of output directory containing output from df-deploy.py.

"""
MODEL_NAME_HELP_STR = """
Name of model model to use for prediction. The model must exist in the deployment directory.
If omitted, the model used is the top-performing model from `models_info.json` is used.

The structure of the model name is [model_shortname]_[feature_selection].

\033[1mEXAMPLES\033[0m
    lgbm_embed_lgbm
    rf_embed_linear
    lr_no_select

"""
OUT_HELP_STR = """
Path to the output directory where predictions and metrics will be saved.
If the directory does not exist, it will be created.
If omitted, the `predictions` subdirectory in `model-dir` will be used or created.

"""

# SHOW_PROBA_HELP_STR = """
# Produce CSV files of class probabilities for classification tasks.

# """

# SHOW_EMBED_HELP_STR = """
# Display embedded features in `predictions_with_inputs.csv`.

# """


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Evaluate a trained ML model on a super holdout dataset and save predictions + performance metrics.",
        formatter_class=RawTextHelpFormatter,
    )


    parser.add_argument(
        "--df",
        type=Path,
        required=True,
        help=DF_HELP_STR,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help=MODEL_DIR_HELP_STR,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=MODEL_NAME_HELP_STR,
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        help=OUT_HELP_STR,
    )

    # parser.add_argument(
    #     "--show-proba",
    #     action="store_true",
    #     help=SHOW_PROBA_HELP_STR,
    # )
    # parser.add_argument(
    #     "--show-emeddings",
    #     action="store_true",
    #     help=SHOW_EMBED_HELP_STR,
    # )

    return parser

# TODO: subpaths should be an enum to ensure consistency across deployment and prediction
class PredictOptions:
    def __init__(
        self,
        df: Path,
        deployment_dir: Path,
        model_name: str,
        out: Path,
    ):
        
        self.df: DataFrame = load_df(df)

        self.model_dir: Path = deployment_dir

        model_name = get_model_name(deployment_dir)
        models_dir = deployment_dir / "trained_models"
        self.models_dir: Path = models_dir
        self.models_info: dict = json.load(open(models_dir / "models_info.json", 'r'))
        self.model_path: Path = models_dir / f"{model_name}.pkl"
        # TODO: check if exists

        processor_dir: Path = deployment_dir / "processor"
        try:
            processor = DataProcessor.from_file(deployment_dir / "processor.pkl")  # try to load the old saving location
        except FileNotFoundError:
            processor = DataProcessor.from_file(processor_dir / "processor.pkl")
        except:
            processor = DataProcessor.from_json_pkl(
                processor_dir / "processor_dict.json",
                processor_dir / "imputer.pkl",
            )
        self.processor: DataProcessor = processor

        embed_dir = deployment_dir / "embedding_model"
        self.embed_dir = embed_dir if embed_dir.exists() else None

        meta_path = deployment_dir / "meta.json"
        # meta_path = models_dir / "meta.json"
        with open(meta_path) as f:
            metadata: dict = json.load(f)
        self.target = metadata.get("target", "target")

        # Program output is set to specified directory or the predictions directory in the deployment output
        outdir: Path = out or deployment_dir / "predictions"
        timestamp = datetime.now().strftime("%b_%d_%Y_%H%M%S")
        self.program_dir = outdir / timestamp
        self.program_dir.mkdir(parents=True, exist_ok=True)


def get_options() -> PredictOptions:
    parser = make_parser()
    args = parser.parse_args()
    return PredictOptions(
        args.df,
        args.model_dir,
        args.model_name,
        args.out,
    )


@dataclass
class RankedModel:
    name: str
    rank: int

def get_model_name(model_dir: Path, model_name: str = None) -> Path:
    """
    Returns `model_name` if not None, else the best ranked model.
    """
    if model_name is not None:
        return model_name

    trained_models = model_dir / "trained_models"
    model_info: dict[str, dict] = json.load(open(trained_models / "models_info.json", 'r'))

    best_model: RankedModel = None

    for name, info in model_info.items():
        model = RankedModel(name, info["rank"])

        if best_model is None or model.rank < best_model.rank:
            best_model = model

    if best_model is None:
        raise KeyError("Unable to find a best model.")
    
    return best_model.name