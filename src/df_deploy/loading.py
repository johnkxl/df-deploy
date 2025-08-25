from pandas import DataFrame, read_csv, read_excel, read_parquet
from pathlib import Path


def load_df(df_path: Path | str) -> DataFrame:
    """
    Identify filetype and return DataFrame.
    """
    ext = Path(df_path).suffix
    df: DataFrame
    match ext:
        case ".csv":
            df = read_csv(df_path)
        case ".xlsx":
            df = read_excel(df_path)
        case ".parquet":
            df = read_parquet(df_path)
        case _:
            raise TypeError(f"Cannot load file of type {ext} as DataFrame.")
    return df