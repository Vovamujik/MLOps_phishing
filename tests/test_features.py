import numpy as np
import pandas as pd
import pytest

from mlops_phishing.features import URL_FEATURES, extract_hosts, normalize_urls, url_features


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


def features_of(url: str) -> dict[str, float]:
    return url_features(pd.Series([url])).iloc[0].to_dict()


def test_url_features_columns_are_fixed_and_numeric():
    result = url_features(pd.DataFrame({"URL": ["https://www.a.com", "http://b.org/x"]}))

    assert tuple(result.columns) == URL_FEATURES
    assert len(result) == 2
    assert all(dtype == np.float64 for dtype in result.dtypes)


def test_url_features_ignore_scheme_and_www():
    assert features_of("https://www.example.com/") == features_of("example.com")


def test_url_features_lengths_and_ratios():
    features = features_of("https://ab1-2.com/x?y=1")

    # Нормализованный URL: ab1-2.com/x?y=1 — 15 символов, 3 цифры, 5 спецсимволов
    assert features["length"] == 15
    assert features["host_length"] == 9
    assert features["digit_ratio"] == pytest.approx(3 / 15)
    assert features["special_ratio"] == pytest.approx(5 / 15)
    assert features["host_hyphens"] == 1


@pytest.mark.parametrize(
    ("url", "subdomains"),
    [
        ("https://example.com", 0),
        ("https://a.b.example.com/c.d", 2),
        ("http://user@login.example.com:8080/", 1),
        ("http://192.168.0.1/admin", 0),
    ],
)
def test_url_features_subdomains(url, subdomains):
    assert features_of(url)["subdomains"] == subdomains


def test_url_features_flags():
    phishing = features_of("http://192.168.0.1/login/verify-account")
    legit = features_of("https://www.google.com")

    for flag in ("is_ip", "has_login", "has_verify", "has_account"):
        assert phishing[flag] == 1
        assert legit[flag] == 0
    assert features_of("https://xn--80ak6aa92e.com")["has_punycode"] == 1
    for host in ("evil.firebaseapp.com", "evil.web.app", "cloudflare-ipfs.com/ipfs/qm"):
        assert features_of(host)["free_hosting"] == 1


def test_url_features_empty_url_has_zero_ratios():
    features = features_of("")

    assert features["length"] == 0
    assert features["digit_ratio"] == 0


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.google.com", "google.com"),
        ("https://github.com/mlflow/mlflow", "github.com"),
        ("http://example.com?id=1", "example.com"),
        ("example.com#top", "example.com"),
        ("https://docs.python.org/3/library/os.html", "docs.python.org"),
        ("http://192.168.0.1:8080/admin", "192.168.0.1:8080"),
        ("", ""),
    ],
)
def test_extract_hosts(url, expected):
    assert extract_hosts([url])[0] == expected


def test_extract_hosts_accepts_series_and_returns_string_array():
    result = extract_hosts(pd.Series(["https://www.a.com/x", "b.org"]))

    assert isinstance(result, np.ndarray)
    assert result.dtype.kind == "U"
    assert result.tolist() == ["a.com", "b.org"]


def test_extract_hosts_empty_input():
    assert extract_hosts([]).shape == (0,)
