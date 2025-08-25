import math
import json

from pathlib import Path


def clean_json_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_json_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_nan(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


def save_json(to_save, filepath: Path | str) -> None:
    cleaned = clean_json_nan(to_save)
    with open(filepath, "w") as f:
        json.dump(cleaned, f, indent=4)