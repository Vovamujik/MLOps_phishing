# MLOps Phishing Detector

Сервис обнаружения фишинговых сайтов по URL (датасет PhiUSIIL, UCI).
API на FastAPI принимает задачи и кладёт их в ограниченную очередь RabbitMQ,
воркер на FastStream считает их моделью из MLflow Model Registry
(`phishing-url-clf@champion`, загружается один раз при старте воркера) и пишет
результат в Postgres. Подробнее — в разделах [Конвейер задач](#конвейер-задач)
и [MLflow](#mlflow).

## Быстрый старт

```bash
uv sync
uv run pre-commit install
cp .env.example .env

docker compose up -d --build --wait   # API, воркер, RabbitMQ, Postgres, MLflow (mlflow-db, s3)
curl localhost:8000/api/v1/health
```

Воркеру нужна `phishing-url-clf@champion` в реестре: без неё он падает при
старте, и `--wait` его не дождётся. Сначала модель (раздел [MLflow](#mlflow)),
или стек без воркера: `docker compose up -d --build --wait app` — API принимает
задачи, они ждут в очереди.

Без Docker: `docker compose up -d postgres rabbitmq rabbitmq-init`, затем
`uv run uvicorn mlops_phishing.app:app --reload` и воркер
`uv run faststream run mlops_phishing.worker:app` (адреса Postgres `:5433`,
RabbitMQ `:5672` и MLflow `:5050` — в `.env.example`).

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/healthz` | liveness-проба: процесс жив. Не версионируется и не ходит в БД |
| GET | `/api/v1/version` | версия из метаданных пакета, git-коммит и время сборки образа |
| GET | `/api/v1/health` | версии и `latency_ms` Postgres и RabbitMQ; 200 или 503 |
| POST | `/api/v1/tasks` | `{"url"}` → 202, `task_id` и ссылка на результат в `Location`; 503 и `Retry-After`, если очередь полна; 503 без него, если брокер не ответил за `PUBLISH_TIMEOUT` (2 с) |
| GET | `/api/v1/tasks/{task_id}` | статус `PENDING` или `DONE` и результат: вероятность фишинга, версия модели; 404, если задачи нет или она ещё не записана |

## Конвейер задач

```
источник CSV ──────┐
                   ├─ publish + confirm ─▶ [tasks] quorum-очередь: лимит 100, reject-publish
POST /api/v1/tasks ┘     nack, если полна       │ prefetch 4 = 4 потока inference
                                                ▼
       Postgres ◀── upsert по task_id ── worker (FastStream, модель @champion)
                                                │ 3 неудачные доставки
                                                ▼
                                       tasks.dlx → [tasks.dlq]
```

| Сервис | Что делает |
|---|---|
| `rabbitmq` | брокер; Management UI http://localhost:15672 (`app` / `app`) |
| `rabbitmq-init` | при каждом `up` задаёт очередь `tasks`, policy с лимитом и повторами, DLQ |
| `worker` | модель по alias, inference в пуле потоков, результат в базу; второй — `docker compose up -d --scale worker=2` |
| `source` | пакетный скоринг CSV, запускается вручную: `docker compose run --rm source --limit 2000` |

Когда очередь полна, брокер отвечает издателю nack, и дальше по-разному:

- **источник** внутри контура: ждёт с растущей паузой и шлёт ту же задачу
  снова. Буфера нет, скорость сама равна пропускной способности воркеров —
  это противодавление;
- **внешний клиент** вне контура, замедлить его нельзя: API сразу отвечает 503
  с `Retry-After` и ничего не пишет в базу — это сброс нагрузки (load shedding).
  `Retry-After` — время, за которое один воркер разберёт 20% лимита: после
  отказа quorum-очередь снова принимает с 80% лимита. При настройках по
  умолчанию 20 задач / 20 задач/с = 1 с.

Статуса `FAILED` нет: задача, ушедшая в DLQ, остаётся `PENDING`, её разбирает
оператор. Если база была недоступна в момент приёма, задача всё равно в
очереди, а строку создаст воркер — до этого GET отвечает 404.

```bash
curl -si -X POST localhost:8000/api/v1/tasks -H 'Content-Type: application/json' \
  -d '{"url": "https://secure-login-paypal.example.com/verify/account"}'
# HTTP/1.1 202 Accepted
# location: http://localhost:8000/api/v1/tasks/c0aec273-9ba5-422f-8302-e14d12d4bfb1

curl -s localhost:8000/api/v1/tasks/c0aec273-9ba5-422f-8302-e14d12d4bfb1
```

```json
{"task_id":"c0aec273-9ba5-422f-8302-e14d12d4bfb1",
 "url":"https://secure-login-paypal.example.com/verify/account","source":"api",
 "status":"DONE","is_phishing":true,"phishing_probability":0.9977993369102478,
 "model_version":"1","enqueued_at":"2026-10-20T16:35:55.193645Z",
 "completed_at":"2026-10-20T16:35:55.429423Z"}
```

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
| `test` | pytest с покрытием (падает ниже 80%) |
| `smoke` | поднимает MLflow, регистрирует игрушечную seed-модель как `@champion`, `docker compose up --wait` (API, RabbitMQ, воркер), запросы к эндпоинтам и задача через `POST /api/v1/tasks`, которая доходит до `DONE` с версией модели 1 |

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

Каждый запрос пишется одной строкой с методом, путём, статусом, временем
ответа и IP клиента. Уровень зависит от ответа: 5xx — ERROR (кроме 503 с
`Retry-After`: очередь заполнена, это штатный сброс нагрузки — WARNING),
4xx — WARNING, остальное — INFO. Запрос, упавший с необработанным исключением, тоже пишется:
на ERROR, с `exception=<тип>`, а traceback следом печатает uvicorn.
Успешные запросы к `/healthz` пишутся на DEBUG — их каждые
10 секунд делает Docker HEALTHCHECK. Access-лог uvicorn выключен: он дублировал
запись middleware, но без `request_id`.

```bash
docker compose logs -f app
```

## MLflow

### Сервисы и порты

```
Комп: training/*.py ── http://localhost:5050 ──┐
                                               ▼
worker ──────── http://mlflow:5000 ──▶ mlflow (сервер + UI)
                                        ├─▶ mlflow-db (Postgres): эксперименты, раны, метрики, реестр
                                        └─▶ s3 (RustFS, бакет mlflow): графики и модели
```

| Сервис | Адрес с компа | Назначение |
|---|---|---|
| `mlflow` | http://localhost:5050 | Tracking Server и UI. 5000 на macOS занят AirPlay |
| `mlflow-db` | — | Postgres, Backend Store: эксперименты, раны, метрики, реестр |
| `s3` | http://localhost:9001 | RustFS, Artifact Store; веб-консоль `mlflow` / `mlflow-secret` |
| `s3-init` | — | создаёт бакет `mlflow` и завершается с кодом 0 |
| `worker` | — | FastStream: загружает модель при старте, считает задачи из очереди |
| `app` | http://localhost:8000 | FastAPI: `POST /api/v1/tasks`, `GET /api/v1/tasks/{task_id}` |
| `postgres` | `localhost:5433` | Postgres приложения |

Клиенты знают только адрес MLflow: файлы проксирует сервер, ключи от S3
только у него.

### От запуска до предсказания

```bash
uv sync --all-groups                          # обычный `uv sync` удалит группу ml
docker compose up -d --wait mlflow            # только MLflow (+ mlflow-db, s3): воркеру пока нечего загружать
uv run python -m training.download_data       # CSV в data/raw/ + проверка sha256

uv run python -m training.eda                 # ран в эксперименте phiusiil-eda
uv run python -m training.train baseline normalized handcrafted host-only

# run_id берётся из UI или из вывода train
uv run python -m training.registry register <run_id baseline>    # v1
uv run python -m training.registry register <run_id normalized>  # v2
uv run python -m training.registry alias champion 2
uv run python -m training.registry show

docker compose up -d --build --wait           # воркер стартует, когда @champion уже есть
curl -s -X POST localhost:8000/api/v1/tasks -H 'Content-Type: application/json' \
  -d '{"url": "https://secure-login-paypal.example.com/verify/account"}'
curl -s localhost:8000/api/v1/tasks/<task_id из ответа>
```

Воркер при старте один раз разрешает `phishing-url-clf@champion` в номер версии
и загружает её. Если он уже запущен (например, после быстрого старта), после
`alias` нужен `docker compose restart worker`: без рестарта модель не
перечитывается. Результат задачи содержит `model_version` — версию, которая его
посчитала (пример ответа — в разделе [Конвейер задач](#конвейер-задач)).

Переключение версии без правки кода: `registry alias champion 1` и
`docker compose restart worker` (без пересборки) — в логе старта
`Воркер готов: модель phishing-url-clf v1 (@champion)`, новые задачи получают
`"model_version":"1"`. `registry alias champion 2` и ещё один рестарт
возвращают v2. Версия меняется только при старте воркера.

Модели `normalized`, `handcrafted` и `host-only` ссылаются на функции из
`src/mlops_phishing/features.py`. Со старым образом без этого файла загрузка
упадёт с `ModuleNotFoundError`: воркер не стартует, причина в логе, задачи
ждут в очереди.

### EDA

Эксперимент `phiusiil-eda`, ран `eda`: 8 графиков,
таблица ROC-AUC каждого числового признака, выводы `conclusions.md`.
Датасет: 235 795 URL, 134 850 легитимных и 100 945 фишинга (42.8%),
константная модель даёт accuracy 0.572. В датасете много подсказок,
выдающих метку:

| Подсказка | Что видно |
|---|---|
| `URLSimilarityIndex` | `== 100` у 100% легитимных и 0.8% фишинга; правило «≠ 100 → фишинг» даёт accuracy 99.67% |
| Префикс `https://www.` | у 100% легитимных и 2.4% фишинга |
| `/` после хоста | у 0% легитимных и 60% фишинга: легитимные URL — только главные страницы |
| `FILENAME` | имя файла выдаёт источник, а значит и метку: accuracy 100% |
| HTML-признаки | 9 из топ-10 по одиночному AUC (`LineOfCode` 0.990, `NoOfExternalRef` 0.988): медиана у фишинга около 0, страницу просто не удалось скачать |
| Дубли | 425 повторов URL; в обучении убираются, сплит по доменам |

Поэтому точность около 1.0 здесь — сигнал утечки, а не качества. Сервис
принимает только сырой URL: HTML-признаки и `URLSimilarityIndex` в момент
запроса недоступны, `FILENAME` — артефакт сборки датасета.

### Варианты

Эксперимент `phishing-url-clf`, один вариант — один ран. Сплит по доменам
70/15/15, основная метрика `val_pr_auc` (PR-AUC по фишингу), test — финальная
проверка.

| Вариант | Модель |
|---|---|
| `baseline` | TF-IDF по символьным 3–5-граммам сырого URL → LogisticRegression |
| `normalized` | то же, но из URL убраны схема, `www.` и `/` в конце |
| `handcrafted` | 12 своих признаков нормализованного URL (длина, доли символов, поддомены, IP, `xn--`, login/verify/account, бесплатные хостинги) → HistGradientBoosting |
| `host-only` | как `normalized`, но TF-IDF только по хосту, без пути и query |

Раны в UI называются по варианту. Метрики детерминированы: сплит и модели
с фиксированным seed, повторный запуск даёт те же числа. Время зависит от машины.

| Вариант | val PR-AUC | test PR-AUC | val recall | test recall | val FPR | test FPR | train, с | predict, мс/1000 URL |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.9989 | 0.9992 | 0.9902 | 0.9923 | 0.0000 | 0.0000 | 5.1 | 20.9 |
| `normalized` | **0.9543** | 0.9472 | 0.8128 | 0.7946 | 0.0260 | 0.0239 | 7.3 | 16.7 |
| `handcrafted` | 0.8946 | 0.8790 | 0.6830 | 0.6529 | 0.0269 | 0.0287 | 4.3 | 6.4 |
| `host-only` | 0.9371 | 0.9279 | 0.7697 | 0.7457 | 0.0393 | 0.0363 | 4.3 | 15.0 |

Проверка на отдельных URL, вероятность фишинга:

| URL | baseline | normalized | handcrafted | host-only |
|---|---|---|---|---|
| `https://www.google.com` | 0.001 | 0.713 | 0.230 | 0.911 |
| `https://github.com/mlflow/mlflow` | 0.990 | 0.963 | 1.000 | 0.747 |
| `https://github.com` | 0.696 | 0.588 | 0.230 | 0.747 |
| `https://www.wikipedia.org` | 0.002 | 0.029 | 0.208 | 0.030 |
| `https://www.microsoft.com` | 0.009 | 0.576 | 0.208 | 0.570 |
| `https://docs.python.org/3/library/os.html` | 0.976 | 0.924 | 1.000 | 0.364 |
| `http://paypal-verify-account.web.app` | 1.000 | 0.998 | 1.000 | 0.999 |
| `https://secure-login-paypal.example.com/verify/account` | 0.998 | 0.998 | 1.000 | 0.928 |

### Выводы

- **v2 = `normalized`.** Критерий заявлен заранее: лучший `val_pr_auc` среди
  вариантов без прямой подсказки схемы и `www.`. У `normalized` 0.954 против
  0.937 у `host-only` и 0.895 у `handcrafted`.
- **Но перевес над `host-only` целиком даёт путь.** В val нет ни одного
  легитимного URL с путём, и правило «есть путь → фишинг» даёт precision 1.000
  при recall 0.334. На URL без пути (все 20 260 легитимных и 10 733 фишинговых
  в val) варианты равны: PR-AUC 0.909 у обоих, на test тоже. `normalized`
  оставили v2: на URL без пути он не хуже, а путь в реальном трафике несёт
  настоящие признаки фишинга (`/login`, `/verify`). В этом датасете путь —
  ещё и утечка, см. ограничения.
- **`baseline` в отбор не идёт** и остаётся v1 для истории. Его 0.999 держится
  на префиксе `https://www.`, найденном в EDA: google он считает легитимным
  только потому, что URL начинается с `https://www.`.
- **Переобучения нет.** У всех вариантов val и test расходятся не больше чем
  на 0.016 по PR-AUC и 0.031 по recall; сплит по доменам не даёт модели узнать
  в test сайт из train.

### Ограничения

- **Ни один вариант не годится для реального использования.** Легитимный класс
  в датасете — только главные страницы вида `https://www.<домен>` без путей и
  почти без крупных брендов. Любая страница с путём выглядит для модели как
  фишинг, а хост с брендом — тоже: `google` встречается в 1 388 фишинговых
  URL и в 12 легитимных, `github` — в 402 и в 1.
  `normalized` (v2) называет `https://www.google.com` фишингом с p=0.71.
- **Каждый шаг убирает одну подсказку датасета и обнажает следующую.**
  `normalized` снимает схему и `www.`, но остаётся путь. `host-only` снимает
  путь, но модель начинает считать фишингом бренды. Метрики падают на каждом
  шаге: это не регрессия, а честная оценка того, что URL сам по себе знает о
  фишинге в этом датасете.
- **Горячее обновление по alias.** Переключить `@champion` и
  `docker compose restart worker`: пока воркер стартует, обработка стоит, а API
  продолжает принимать задачи — они ждут в очереди (до лимита, дальше 503 с
  `Retry-After`); воркеры, перезапущенные в разное время, пишут разные
  `model_version`; новая версия может требовать новый код (`features.py`), то
  есть модель и образ связаны.
