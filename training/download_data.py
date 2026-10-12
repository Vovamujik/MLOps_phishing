"""Скачивает датасет PhiUSIIL с UCI в data/raw/ и проверяет sha256."""

import hashlib
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import certifi

DATASET_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
CSV_SHA256 = "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CSV_PATH = RAW_DIR / CSV_NAME


def sha256_of(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download_csv(target: Path) -> None:
    # Сертификаты из certifi Python с python.org на macOS не видит системные
    context = ssl.create_default_context(cafile=certifi.where())
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dataset.zip"
        with (
            urllib.request.urlopen(DATASET_URL, timeout=60, context=context) as response,
            archive.open("wb") as file,
        ):
            shutil.copyfileobj(response, file)
        with zipfile.ZipFile(archive) as zf, zf.open(CSV_NAME) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def main() -> None:
    if CSV_PATH.exists() and sha256_of(CSV_PATH) == CSV_SHA256:
        print(f"Уже скачан, sha256 совпадает: {CSV_PATH}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    partial = CSV_PATH.with_suffix(".part")
    print(f"Скачиваю {DATASET_URL}")
    download_csv(partial)

    actual = sha256_of(partial)
    if actual != CSV_SHA256:
        partial.unlink()
        raise SystemExit(f"sha256 не совпал: ожидали {CSV_SHA256}, получили {actual}")

    partial.replace(CSV_PATH)
    print(f"Готово: {CSV_PATH} ({CSV_PATH.stat().st_size / 2**20:.1f} MiB), sha256 совпадает")


if __name__ == "__main__":
    main()
