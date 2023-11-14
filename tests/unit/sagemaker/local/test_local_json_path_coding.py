import numpy as np
import pandas as pd
import pytest
import sagemaker.local.json_coding as json_coding

@pytest.fixture()
def df():
    return pd.DataFrame(
        {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]},
    )


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("$[1]", [1]),
        ("$[:3]", [0, 1, 2]),
        ("$[1:3]", [1, 2]),
        ("$[1:3, -1]", [1, 2, -1]),
        ("$[1:]", [1, 2, 3, 4]),
        ("$[3:3]", []),
        ("$[1:2, -1]", [1, -1]),
        ("$[1, 1, 1]", [1]),
    ],
)
def test_json_path(expr, expected):
    assert json_coding.resolve_json_path(expr, 5) == expected

@pytest.mark.parametrize(
        ("path", "idx"), 
        [("$[1]", [1]), ("$[-1]", [-1]), (None, range(3))]
)
def test_apply_json_path(df, path, idx):
    assert all(json_coding.apply_json_path(df, path) == df.iloc[:, idx])


def test_str_to_df(df):
    assert (json_coding.str_to_df(json_coding.df_to_str(df)).values == df.values).all()
