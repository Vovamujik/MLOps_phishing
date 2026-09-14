"""Тесты работы с пулом соединений."""

from mlops_phishing.config import Settings
from mlops_phishing.db import close_pool, create_pool, mask_dsn

from .conftest import DEAD_DATABASE_URL


def test_mask_dsn_hides_credentials():
    """Логин и пароль не должны попадать в логи."""
    masked = mask_dsn("postgresql://admin:s3cret@db.internal:5432/app")

    assert masked == "db.internal:5432/app"
    assert "s3cret" not in masked
    assert "admin" not in masked


def test_mask_dsn_keeps_dsn_without_credentials():
    """Строка без учётных данных остаётся как есть."""
    assert mask_dsn("postgresql://localhost:5432/app") == "postgresql://localhost:5432/app"


async def test_pool_does_not_connect_at_startup():
    """Приложение обязано подниматься при недоступной базе.

    Ключевое свойство: пул создаётся с min_size=0 и не открывает ни одного
    соединения при старте. Иначе авария Postgres уводила бы контейнер
    приложения в рестарт-луп, вместо того чтобы отразиться в /api/v1/health.
    DSN указывает на закрытый порт: если бы пул подключался, тест упал бы.
    """
    pool = await create_pool(Settings(database_url=DEAD_DATABASE_URL))
    try:
        assert pool.get_min_size() == 0
        assert pool.get_size() == 0
    finally:
        await close_pool(pool)
