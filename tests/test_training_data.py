import hashlib

import pandas as pd
import pytest

from training import data


def make_frame(n_domains: int = 200, rows_per_domain: int = 3) -> pd.DataFrame:
    rows = [
        {"URL": f"https://d{d}.com/{i}", "Domain": f"d{d}.com", data.TARGET: d % 2}
        for d in range(n_domains)
        for i in range(rows_per_domain)
    ]
    return pd.DataFrame(rows)


def test_split_keeps_each_domain_in_one_part():
    splits = data.split_by_domain(make_frame())
    train, val, test = (set(part["Domain"]) for _, part in splits.named())

    assert not train & val
    assert not train & test
    assert not val & test


def test_split_loses_no_rows_and_is_about_70_15_15():
    df = make_frame()
    sizes = {name: len(part) for name, part in data.split_by_domain(df).named()}

    assert sum(sizes.values()) == len(df)
    assert sizes["val"] / len(df) == pytest.approx(0.15, abs=0.02)
    assert sizes["test"] / len(df) == pytest.approx(0.15, abs=0.02)


def test_split_is_reproducible():
    df = make_frame()

    assert data.split_by_domain(df).test.index.equals(data.split_by_domain(df).test.index)


@pytest.fixture
def raw_csv(tmp_path, monkeypatch):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "FILENAME": ["1.txt", "op1.txt", "op2.txt"],
            "URL": ["https://www.a.com", "http://b.net/login", "http://b.net/login"],
            "Domain": ["www.a.com", "b.net", "b.net"],
            "label": [1, 0, 0],
        }
    ).to_csv(csv, index=False)
    monkeypatch.setattr(data, "CSV_PATH", csv)
    monkeypatch.setattr(data, "CSV_SHA256", hashlib.sha256(csv.read_bytes()).hexdigest())
    return csv


def test_load_dataset_drops_leaky_column_and_duplicate_urls(raw_csv):
    df = data.load_dataset()

    assert "FILENAME" not in df.columns
    assert df["URL"].tolist() == ["https://www.a.com", "http://b.net/login"]


def test_load_dataset_makes_phishing_the_positive_class(raw_csv):
    df = data.load_dataset()

    assert "label" not in df.columns
    assert df[data.TARGET].tolist() == [0, 1]


def test_load_dataset_rejects_modified_file(raw_csv):
    raw_csv.write_text("URL\nhttps://tampered.example\n")

    with pytest.raises(SystemExit):
        data.load_dataset()
