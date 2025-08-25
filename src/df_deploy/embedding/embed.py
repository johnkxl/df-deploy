import pandas as pd

from pandas import DataFrame
from pathlib import Path
from typing import Any

from deeptune_beta.embed.vision.embed import embed_vision_dataset
from deeptune_beta.embed.nlp.embed import embed_nlp_dataset
from deeptune_beta.utils import UseCase, set_seed, get_model_architecture


class DeepTuneOpts:
    def __init__(self, cli_dict: dict[str, Any]):
        self.data_type = cli_dict["data_type"]

        model_version: str = cli_dict.get("model_version")
        model_architecture: str = get_model_architecture(model_version)

        self.model_version: str = model_version
        self.model_architecture: str = cli_dict.get("model_architecture", model_architecture)

        self.use_case: UseCase = UseCase.from_string("finetuned" if not cli_dict.get("use-peft", False) else "peft")
        self.num_classes: int = cli_dict.get("num_classes")
        self.added_layers: int = cli_dict.get("added_layers")
        self.embed_size: int = cli_dict.get("embed_size")
        self.freeze_backbone: bool = cli_dict.get("freeze_backbone", False)  # not needed outside training
        self.mode: str = cli_dict.get("mode")  # should be present
        
        self.model_weights: Path = cli_dict.get("model_weights")
        
        self.model_name: str = cli_dict.get("model", f"{self.use_case.value}-{model_version}")

        self.batch_size: int = cli_dict.get("batch_size", 16)
    
    def __setattr__(self, name, value):
        if name == "mode":
            existing = self.__dict__.get("mode")
            if existing is not None and existing != value:
                raise ValueError("Trying to set mode to a different value than indicated in the training CLI arguments.")
        super().__setattr__(name, value)
    
    @classmethod
    def from_traindir(cls, dirpath: Path) -> "DeepTuneOpts":
        """
        Load DeepTuneOpts from training output directory.
        """
        import json
        model_weights = dirpath / "model_weights.pth"
        cli_arguments_json = dirpath / "cli_arguments.json"
        assert all([model_weights.exists(), cli_arguments_json.exists()])
        
        with open(cli_arguments_json, 'r') as f:
            cli_dict: dict = json.load(f)
        cli_dict.update(model_weights=model_weights)
        return cls(cli_dict)


def embed_df(df: DataFrame, model_path: Path) -> DataFrame:
    opts = DeepTuneOpts.from_traindir(model_path)

    if opts.data_type == "image":
        if "images" not in df.columns:
            raise KeyError("The DataFrame column containing the image bytes must be called `images`.")
        data_col = "images"
        # Ensure df to embd only contains data, not target, else embedded with also have target
        data_df = DataFrame(df[data_col])

        set_seed(use_fixed_seed=True)
        embedded_df = embed_vision_dataset(
            df=data_df,
            mode=opts.mode,
            num_classes=opts.num_classes,
            model_version=opts.model_version,
            model_architecture=opts.model_architecture,
            model_weights=opts.model_weights,
            use_case=opts.use_case,
            added_layers=opts.added_layers,
            embed_size=opts.embed_size,
            batch_size=opts.batch_size or 16,
        )
    
    elif opts.data_type == "text":
        if "text" not in df.columns:
            raise KeyError("The DataFrame column containing the text to be embedded must be called `text`.")
        data_col = "text"
        data_df = DataFrame(df[data_col])

        set_seed(use_fixed_seed=True)
        embedded_df = embed_nlp_dataset(
            df=data_df,
            mode=opts.mode,
            num_classes=opts.num_classes,
            model_version=opts.model_version,
            model_architecture=opts.model_architecture,
            model_path=model_path,
            use_case=opts.use_case,
            added_layers=opts.added_layers,
            embed_size=opts.embed_size,
            batch_size=opts.batch_size or 16,
        )
    else:
        raise NotImplementedError("Only embedding for image data and text data is currently supported.")
    
    # Since embedded_df doesn't contain target column, it won't be duplicated
    concat_df = pd.concat([embedded_df, df.drop(columns=data_col)], axis=1)
    
    return concat_df
