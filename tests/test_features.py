import numpy as np
import pandas as pd
import pytest

from mlops_phishing.features import normalize_urls


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.google.com", "google.com"),
        ("http://www.example.com/", "example.com"),
        ("https://github.com/mlflow/mlflow/", "github.com/mlflow/mlflow"),
        ("HTTPS://WWW.Example.com", "Example.com"),
        ("example.com/login?id=1", "example.com/login?id=1"),
        ("https://www2.example.com", "www2.example.com"),
        ("https://example.com/www.page", "example.com/www.page"),
        ("", ""),
    ],
)
def test_normalize_urls(url, expected):
    assert normalize_urls([url])[0] == expected


def test_normalize_urls_accepts_series_and_returns_string_array():
    result = normalize_urls(pd.Series(["https://www.a.com/", "http://b.org"]))

    assert isinstance(result, np.ndarray)
    assert result.dtype.kind == "U"
    assert result.tolist() == ["a.com", "b.org"]


def test_normalize_urls_empty_input():
    assert normalize_urls([]).shape == (0,)
