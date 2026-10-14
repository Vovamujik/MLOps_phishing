"""Предобработка URL, общая для обучения и приложения.

Живёт в src/, а не в training/: обученная модель хранит ссылку на эти функции,
и приложение при загрузке модели должно их импортировать.
"""

import re
from collections.abc import Iterable

import numpy as np

# Схема и www. в датасете выдают класс: все легитимные URL начинаются с https://www.
_PREFIX = re.compile(r"^(https?://)?(www\.)?", re.IGNORECASE)


def normalize_urls(urls: Iterable[str]) -> np.ndarray:
    """Убирает из каждого URL схему http(s)://, www. и / в конце."""
    return np.array([_PREFIX.sub("", url).rstrip("/") for url in urls], dtype=str)
