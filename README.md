# MLOps Phishing Detector

Асинхронный FastAPI-сервис обнаружения фишинговых сайтов по URL
(датасет PhiUSIIL, UCI). Текущий этап: инфраструктурный каркас.

## Быстрый старт

```bash
uv sync
uv run pre-commit install
cp .env.example .env

docker compose up -d --build        # приложение + Postgres
curl localhost:8000/api/v1/health
```

Без Docker: `uv run uvicorn mlops_phishing.app:app --reload`
(Postgres из compose доступен на `localhost:5433`).

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/healthz` | liveness-проба: процесс жив. Не версионируется и не ходит в БД |
| GET | `/api/v1/version` | версия из метаданных пакета, git-коммит и время сборки образа |
| GET | `/api/v1/health` | версии и `latency_ms` зависимостей; 200 или 503 |

## Проверки

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest                 # тесты + покрытие, порог 80%
uv run pre-commit run --all-files
```

## Docker

Двухстадийный образ: в builder `uv` ставит сначала зависимости (слой
кэшируется), потом сам пакет (non-editable, с `METADATA`, откуда читается
версия). В runtime попадает только `.venv`, без `uv` и исходников.
Процесс работает под непривилегированным пользователем, `HEALTHCHECK` бьёт
в `/healthz`. `GIT_COMMIT` и `BUILD_TIME` передаются build-аргументами и
отдаются в `/api/v1/version`.

## CI/CD

### CI — `.github/workflows/ci.yml`

Запускается на push в любую ветку и на pull request в `main`.

| Job | Что делает |
|---|---|
| `lint` | `uv lock --check`, `ruff check`, `ruff format --check`, все хуки pre-commit |
| `test` | pytest с покрытием (падает ниже 80%), `coverage.xml` сохраняется артефактом |
| `smoke` | `docker compose up --wait`, запросы ко всем трём эндпоинтам на живом Postgres |

### CD — `.github/workflows/cd.yml`

Собирает образ и публикует в GHCR: `ghcr.io/vovamujik/mlops_phishing`.

**Условие запуска:** push git-тега `vX.Y.Z`, и только после того, как для
этого коммита заново прошёл весь CI (CI подключён как reusable workflow,
`needs: ci`).

Обоснование:

- **Релиз это осознанное решение.** Не каждый коммит в `main` должен
  становиться образом. Тег ставится, когда версия готова, реестр не
  засоряется промежуточными сборками.
- **В реестр не попадает непроверенный код.** Публикация зависит от
  линтеров, тестов и smoke-теста с БД.
- **Одна версия везде.** CD падает, если тег не совпадает с `version` в
  `pyproject.toml`. Эта же версия приходит в `/api/v1/version` из метаданных
  пакета. В итоге git-тег, тег образа и ответ API всегда совпадают.
- **Неизменяемость и откат.** Семвер-тег не перезаписывается, откат это
  деплой предыдущего тега.

**Версионирование образа** для тега `v1.2.3`:

| Тег образа | Смысл |
|---|---|
| `1.2.3` | точная неизменяемая версия |
| `1.2` | последний патч минорной ветки |
| `sha-<7 символов>` | трассировка до коммита |
| `latest` | последний релиз |

### Выпуск релиза

```bash
uv version --bump patch          # или вручную поправить version в pyproject.toml
git commit -am "chore: release $(uv version --short)"
git tag "v$(uv version --short)"
git push && git push --tags
```

## Логирование

Формат: `время | уровень | request_id | логгер | сообщение`. Middleware
присваивает каждому запросу 8-символьный id, возвращает его в заголовке
`X-Request-ID` и подмешивает во все записи лога этого запроса.

```bash
docker compose logs -f app
```
