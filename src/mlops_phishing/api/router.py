"""Сборка всех маршрутов приложения.

Версионируемая часть API собирается под префиксом `/api/v1`, системные
эндпоинты подключаются в корень. Префикс задаётся в одном месте: при
появлении `/api/v2` добавится второй роутер, а первый останется рабочим.
"""

from fastapi import APIRouter

from mlops_phishing.api import system
from mlops_phishing.api.v1 import health, version

API_V1_PREFIX = "/api/v1"

v1_router = APIRouter(prefix=API_V1_PREFIX)
v1_router.include_router(version.router)
v1_router.include_router(health.router)

root_router = APIRouter()
root_router.include_router(system.router)
root_router.include_router(v1_router)
