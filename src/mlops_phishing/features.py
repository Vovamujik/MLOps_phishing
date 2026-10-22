"""Предобработка URL, общая для обучения и воркера.

Живёт в src/, а не в training/: обученная модель хранит ссылку на эти функции,
и воркер при загрузке модели должен их импортировать.
"""

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

# Схема и www. в датасете выдают класс: все легитимные URL начинаются с https://www.
_PREFIX = re.compile(r"^(https?://)?(www\.)?", re.IGNORECASE)


def normalize_urls(urls: Iterable[str]) -> np.ndarray:
    """Убирает из каждого URL схему http(s)://, www. и / в конце."""
    # object, а не str: строковый массив numpy дополняет каждую строку до самой
    # длинной, и на train из-за одного URL в 6 тыс. символов вышло бы 3.8 GB
    return np.array([_PREFIX.sub("", url).rstrip("/") for url in urls], dtype=object)


def extract_hosts(urls: Iterable[str]) -> np.ndarray:
    """Нормализует URL, как normalize_urls, и оставляет только хост: всё до /, ? или #.

    У легитимных URL датасета пути нет никогда, и модель по пути учит класс.
    """
    hosts = [re.split(r"[/?#]", url, maxsplit=1)[0] for url in normalize_urls(urls)]
    return np.array(hosts, dtype=object)


# Порядок колонок фиксирован: модель обучается и предсказывает по позициям
URL_FEATURES = (
    "length",
    "host_length",
    "digit_ratio",
    "special_ratio",
    "subdomains",
    "host_hyphens",
    "is_ip",
    "has_punycode",
    "has_login",
    "has_verify",
    "has_account",
    "free_hosting",
)
_IPV4 = r"^\d{1,3}(\.\d{1,3}){3}$"
# Бесплатные хостинги, на которых легко поднять фишинговую страницу
_FREE_HOSTING = r"firebaseapp|web\.app|ipfs"


def url_features(urls: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Числовые признаки URL: по колонке URL таблицы или по Series строк.

    Считаются по нормализованному URL: схема и www. в датасете выдают класс,
    и по длине с ними модель снова выучила бы формат сбора.
    """
    if isinstance(urls, pd.DataFrame):
        urls = urls["URL"]
    url = pd.Series(normalize_urls(urls), index=urls.index, dtype=str).str.lower()
    # Хост: до пути, query или фрагмента, без логина и порта
    host = url.str.split(r"[/?#]", n=1, regex=True).str[0]
    host = host.str.rsplit("@", n=1).str[-1].str.replace(r":\d+$", "", regex=True)
    is_ip = host.str.fullmatch(_IPV4)
    length = url.str.len()
    # max(1): у пустой строки доли нулевые, а не деление на ноль
    denominator = length.clip(lower=1)
    features = pd.DataFrame(
        {
            "length": length,
            "host_length": host.str.len(),
            "digit_ratio": url.str.count(r"\d") / denominator,
            "special_ratio": url.str.count(r"[^a-z0-9]") / denominator,
            "subdomains": (host.str.count(r"\.") - 1).clip(lower=0).where(~is_ip, 0),
            "host_hyphens": host.str.count("-"),
            "is_ip": is_ip,
            "has_punycode": host.str.contains("xn--", regex=False),
            "has_login": url.str.contains("login|signin|logon", regex=True),
            "has_verify": url.str.contains("verif", regex=False),
            "has_account": url.str.contains("account", regex=False),
            "free_hosting": host.str.contains(_FREE_HOSTING, regex=True),
        },
        index=urls.index,
    )
    return features[list(URL_FEATURES)].astype(np.float64)
