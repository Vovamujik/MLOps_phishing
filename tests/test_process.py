"""Тесты POST /api/v1/process: контракт, инференс и загрузка модели на старте."""

import logging
import threading

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from mlops_phishing import ml
from mlops_phishing.app import app
from mlops_phishing.config import Settings, get_settings

from .conftest import DEAD_DATABASE_URL, FakeClassifier, make_model

URL = "http://paypal-secure-login.verify-account.xyz/signin"


async def test_process_returns_verdict_and_model_version(client, install_model):
    """По ответу видно, какая версия модели его дала: это спросят на защите."""
    install_model(make_model(FakeClassifier(phishing_probability=0.93), version="3"))

    response = await client.post("/api/v1/process", json={"url": URL})

    assert response.status_code == 200
    assert response.json() == {
        "url": URL,
        "is_phishing": True,
        "phishing_probability": 0.93,
        "threshold": 0.5,
        "model_name": "phishing-url-clf",
        "model_version": "3",
        "model_alias": "champion",
    }


async def test_process_gives_model_the_url_column_it_was_trained_on(client, install_model):
    """Модель обучена на колонке URL, приложение отдаёт ей ровно её."""
    classifier = FakeClassifier()
    install_model(make_model(classifier))

    await client.post("/api/v1/process", json={"url": URL})

    assert list(classifier.frames[0].columns) == ["URL"]
    assert classifier.frames[0]["URL"].tolist() == [URL]


async def test_low_probability_is_not_phishing(client, install_model):
    install_model(make_model(FakeClassifier(phishing_probability=0.2)))

    body = (await client.post("/api/v1/process", json={"url": URL})).json()

    assert body["is_phishing"] is False


async def test_threshold_comes_from_settings(client, install_model, monkeypatch):
    """Порог — настройка, а не число в коде."""
    install_model(make_model(FakeClassifier(phishing_probability=0.8)))
    monkeypatch.setattr(app.state, "settings", Settings(phishing_threshold=0.9))

    body = (await client.post("/api/v1/process", json={"url": URL})).json()

    assert body["threshold"] == 0.9
    assert body["is_phishing"] is False


async def test_inference_runs_outside_the_event_loop_thread(client, install_model):
    """predict_proba блокирует, поэтому работает в отдельном потоке, а не в event loop."""
    classifier = FakeClassifier()
    install_model(make_model(classifier))

    await client.post("/api/v1/process", json={"url": URL})

    assert classifier.thread_ids
    assert classifier.thread_ids[0] != threading.get_ident()


async def test_process_returns_503_without_model(client):
    """MLflow в тестах недоступен: модель не загрузилась, но приложение живо."""
    response = await client.post("/api/v1/process", json={"url": URL})

    assert response.status_code == 503
    assert (await client.get("/healthz")).status_code == 200


@pytest.mark.parametrize(
    "url",
    ["", "   ", "paypal.com/login", "ftp://files.example.com", "https://", "https://" + "a" * 2050],
)
async def test_process_rejects_invalid_url(client, install_model, url):
    install_model(make_model())

    response = await client.post("/api/v1/process", json={"url": url})

    assert response.status_code == 422


async def test_process_trims_whitespace_around_url(client, install_model):
    install_model(make_model())

    body = (await client.post("/api/v1/process", json={"url": f"  {URL}\n"})).json()

    assert body["url"] == URL


async def test_process_does_not_log_the_url(client, install_model, caplog):
    """URL — данные пользователя: в логе только вердикт и версия модели."""
    caplog.set_level(logging.DEBUG, logger="mlops_phishing")
    install_model(make_model(version="3"))

    await client.post("/api/v1/process", json={"url": URL})

    assert "v3" in caplog.text
    assert URL not in caplog.text


async def test_model_is_loaded_once_at_startup_not_per_request(monkeypatch):
    """Модель грузится один раз в lifespan, а запросы работают с ней в памяти."""
    calls = []

    def fake_load(settings):
        calls.append(settings.model_alias)
        return make_model()

    monkeypatch.setattr(ml, "load_model", fake_load)
    monkeypatch.setenv("DATABASE_URL", DEAD_DATABASE_URL)
    get_settings.cache_clear()

    async with (
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        # Загружена ещё до первого запроса, а не лениво по первому /process
        assert calls == ["champion"]
        assert app.state.model is not None

        for _ in range(3):
            assert (await http.post("/api/v1/process", json={"url": URL})).status_code == 200

    assert calls == ["champion"]
