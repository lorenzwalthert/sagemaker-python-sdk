import ast

from io import StringIO

import pandas as pd


def resolve_json_path(path: str, size: int) -> list[int]:
    """
    Resolves a purly integer based JSON Path like `$[1, 2]` or `$[-1, 3:4]`
    to it's indexes.
    """
    try:
        center = path[2:-1]
        elements = center.split(",")
        idx = []
        existing_col_idx = range(size)
        for element in elements:
            if ":" in element:
                if element[-1] == ":":
                    to_append = existing_col_idx[ast.literal_eval(element[:-1]) :]
                elif element[0] == ":":
                    to_append = existing_col_idx[: ast.literal_eval(element[1:])]
                else:  # a:b
                    start, stop = element.split(":")
                    to_append = existing_col_idx[ast.literal_eval(start) : ast.literal_eval(stop)]
                idx.extend(to_append)
            else:
                idx.append(ast.literal_eval(element))
        return list(set(idx))
    except Exception:
        raise ValueError(f"Invalid JSON path: {path}")


def apply_json_path(
    df: pd.DataFrame,
    path: str | None,
) -> pd.DataFrame:
    # sagemaker only supports numeric positive and negative JSON paths and :n, n:, but not x:y
    if path is None:
        return df
    idx = resolve_json_path(path, df.shape[1])
    return df.iloc[:, idx]


def df_to_str(df: pd.DataFrame) -> str:
    with StringIO() as buffer:
        df.to_csv(buffer, header=False, index=False)
        return buffer.getvalue()


def str_to_df(x: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(x), header=None)
