"""Конфиги наблюдаемости против критериев лабы: проверка файлов без запуска стека.

Синтаксис каждого конфига проверяют сами инструменты (job observability в CI:
promtool, alloy validate, otelcol validate, loki и tempo -verify). Здесь — правила,
которых инструменты не знают: единицы на панелях, rate у счётчиков, `for` у алертов,
закреплённые образы, лимиты памяти, лейблы Loki низкой кардинальности.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "observability"
GRAFANA = OBS / "grafana"
DASHBOARD = GRAFANA / "dashboards" / "pipeline.json"
DATASOURCES = GRAFANA / "provisioning" / "datasources" / "datasources.yaml"
ALERTS = GRAFANA / "provisioning" / "alerting" / "alerts.yaml"

# Служебные «источники» Grafana: выражения алертов и встроенный
BUILTIN_UIDS = {"__expr__", "-- Grafana --", "grafana"}
# Панели без числовой оси: единица им не нужна
UNITLESS_PANELS = {"row", "logs", "table", "text"}
COUNTER = re.compile(r"\b([a-z_:]+_total)\b")
BUCKET = re.compile(r"\b([a-z_:]+_bucket)\b")
RATE_OVER = r"\b(?:rate|irate|increase|deriv)\(\s*{name}\b"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def datasource_uids() -> set[str]:
    return {ds["uid"] for ds in load_yaml(DATASOURCES)["datasources"]}


def dashboard_panels() -> list[dict]:
    return json.loads(DASHBOARD.read_text())["panels"]


def panel_exprs() -> list[tuple[str, str]]:
    return [
        (panel["title"], target["expr"])
        for panel in dashboard_panels()
        for target in panel.get("targets", [])
        if target.get("expr")
    ]


def alert_rules() -> list[dict]:
    return [rule for group in load_yaml(ALERTS)["groups"] for rule in group["rules"]]


def alert_exprs() -> list[tuple[str, str]]:
    return [
        (rule["title"], query["model"]["expr"])
        for rule in alert_rules()
        for query in rule["data"]
        if "expr" in query["model"]
    ]


def test_datasources_have_fixed_uids_and_correlation():
    sources = {ds["uid"]: ds for ds in load_yaml(DATASOURCES)["datasources"]}
    assert {"prometheus", "loki", "tempo"} <= set(sources)
    # Лог → трейс, трейс → логи, метрика → трейс: переходы в один клик (критерий 4.3)
    loki_links = sources["loki"]["jsonData"]["derivedFields"]
    assert any(field["datasourceUid"] == "tempo" for field in loki_links)
    assert sources["tempo"]["jsonData"]["tracesToLogsV2"]["datasourceUid"] == "loki"
    exemplars = sources["prometheus"]["jsonData"]["exemplarTraceIdDestinations"]
    assert exemplars[0]["datasourceUid"] == "tempo"


def test_dashboard_panels_reference_known_datasources():
    known = datasource_uids() | BUILTIN_UIDS
    for panel in dashboard_panels():
        for item in [panel, *panel.get("targets", [])]:
            uid = (item.get("datasource") or {}).get("uid")
            assert uid is None or uid in known, f"{panel['title']}: неизвестный datasource {uid}"


def test_dashboard_panels_have_units():
    for panel in dashboard_panels():
        if panel["type"] in UNITLESS_PANELS:
            continue
        unit = panel.get("fieldConfig", {}).get("defaults", {}).get("unit")
        assert unit, f"{panel['title']}: нет единицы измерения (штраф критерия 4.2)"


@pytest.mark.parametrize("source", ["dashboard", "alerts"])
def test_counters_and_buckets_are_always_rated(source):
    exprs = panel_exprs() if source == "dashboard" else alert_exprs()
    for title, expr in exprs:
        for pattern in (COUNTER, BUCKET):
            for name in set(pattern.findall(expr)):
                assert re.search(RATE_OVER.format(name=name), expr), (
                    f"{title}: {name} без rate/increase — суммируется сырой счётчик"
                )


def test_alerts_have_for_and_known_datasources():
    known = datasource_uids() | BUILTIN_UIDS
    rules = alert_rules()
    assert {"tasks-queue-full", "tasks-dlq-growth"} <= {rule["uid"] for rule in rules}
    for rule in rules:
        # Без for алерт дребезжит на каждом колебании (штраф критерия 4.4)
        assert rule.get("for") not in (None, "", "0", "0s"), f"{rule['title']}: нет for"
        assert rule.get("noDataState"), f"{rule['title']}: не задано поведение без данных"
        for query in rule["data"]:
            assert query["datasourceUid"] in known


def compose_services() -> dict[str, dict]:
    return load_yaml(OBS / "compose.yaml")["services"]


def test_observability_services_are_pinned_limited_and_profiled():
    for name, service in compose_services().items():
        image = service["image"]
        # ${LOKI_VERSION:-3.7.9}: версия по умолчанию — после «:-»
        default = re.search(r":-([^}]+)\}", image)
        tag = default.group(1) if default else image.rsplit(":", 1)[-1]
        assert tag and tag != "latest" and ":" in image, f"{name}: образ без версии"
        assert service.get("profiles"), f"{name}: без профиля поднимется с приложением"
        assert service.get("mem_limit") or service.get("deploy", {}).get("resources"), (
            f"{name}: нет лимита памяти"
        )


def test_storages_and_grafana_have_healthchecks_and_volumes():
    services = compose_services()
    for name in ("loki", "tempo", "prometheus", "grafana"):
        assert services[name].get("healthcheck"), f"{name}: нет healthcheck"
        assert any(":" in str(v) for v in services[name].get("volumes", [])), f"{name}: нет тома"


def test_env_example_pins_versions():
    versions = re.findall(r"^([A-Z_]+_VERSION)=(.*)$", (ROOT / ".env.example").read_text(), re.M)
    assert versions
    for name, value in versions:
        assert value and value != "latest", f"{name} не закреплена"


def test_app_does_not_depend_on_observability():
    app = load_yaml(ROOT / "docker-compose.yml")
    observability = set(compose_services())
    for name, service in app["services"].items():
        deps = service.get("depends_on") or {}
        assert not observability & set(deps), f"{name} зависит от стека наблюдаемости"


def test_loki_labels_are_low_cardinality():
    alloy = (OBS / "alloy" / "config.alloy").read_text()
    keep = re.search(r"stage\.label_keep\s*\{\s*values\s*=\s*\[([^\]]*)\]", alloy)
    assert keep, "в Alloy нет stage.label_keep"
    labels = set(re.findall(r'"([^"]+)"', keep.group(1)))
    # task_id и trace_id — поля записи, лейблом стали бы потоком на каждую задачу
    assert labels == {"service", "level", "env"}
