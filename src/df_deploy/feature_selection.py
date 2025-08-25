import ast
import json
from pathlib import Path
from typing import Literal


FeatureSelectionName = Literal["wrap", "pred", "assoc", "linear", "embed", "none"]

from enum import Enum
class FeatureSelections(Enum):
    WRAP = "wrap"
    PRED = "pred"
    ASSOC = "assoc"
    LINEAR = "linear"
    EMBED = "embed"
    NONE = "none"

    def get_path(self) -> str:
        return {
            FeatureSelections.WRAP : "wrapper/wrapper_selection_data.json",
            FeatureSelections.PRED : "filter/prediction_selection_data.json",
            FeatureSelections.ASSOC : "filter/association_selection_data.json",
            FeatureSelections.LINEAR : "embed/linear_embed_selection_data.json",
            FeatureSelections.EMBED : "embed/lgbm_embed_selection_data.json",
            FeatureSelections.NONE : None,
        }.get(self)
    
    @classmethod
    def choices(cls) -> list[str]:
        return [x.value for x in cls]
    
    @classmethod
    def from_string(cls, s: str) -> "FeatureSelections":
        return cls(s)


SELECTIONS = {
    'wrap' : "wrapper/wrapper_selection_data.json",
    "pred" : "filter/prediction_selection_data.json",
    "assoc": "filter/association_selection_data.json",
    "linear": "embed/linear_embed_selection_data.json",
    "embed": "embed/lgbm_embed_selection_data.json",
}


def get_feature_selection(selection_path: Path | str, selection: FeatureSelectionName | None) -> list[str] | None:
    """
    Return list of columns in feature selection.

    Parameters
    ----------
    selection_path : Path or str
        Directory containing the selection files.

    selection : str or None
        Name of feature selection.

    Returns
    -------
    selected_features : list[str] | None
        List of selected features, if applicable.
    """
    selection_path = Path(selection_path)
    feature_file: str = SELECTIONS.get(selection, None)
    filtered_cols = None

    # Try loading from JSON if available
    if feature_file:
        json_path = selection_path / feature_file
        if json_path.exists():
            with open(json_path, 'r') as f:
                selection_dict: dict = json.load(f)
                filtered_cols = selection_dict.get("selected", None)

    # Fallback: Try parsing Markdown if JSON is missing or invalid
    if filtered_cols is None:
        # Try to find a .md file with the same base name
        report_path = (selection_path / feature_file.replace("data", "report")).with_suffix('.md') if feature_file else None
        if report_path and report_path.exists():
            with open(report_path, 'r') as f:
                content = f.read()

            # Look for the line following the "## Selected Features" heading
            try:
                split_on_heading = content.split("## Selected Features", 1)[1]
                feature_line = split_on_heading.strip().splitlines()[0]
                # Safely parse the Python list using ast.literal_eval
                filtered_cols = ast.literal_eval(feature_line)
            except (IndexError, ValueError, SyntaxError):
                filtered_cols = None

    return filtered_cols


def all_feature_selections(selection_path: Path) -> dict[str, list[str]]:
    selection_names = SELECTIONS.keys()

    feature_selections: dict[str, list[str]] = {}

    for name in selection_names:
        feature_list = get_feature_selection(selection_path, name)
        if feature_list is None:
            continue
        feature_selections[name] = feature_list

    return feature_selections

