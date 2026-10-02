"""Системные (операционные) эндпоинты.

Живут вне `/api/v1` намеренно. Это не часть публичного API-контракта, а
интерфейс для оркестратора: kubernetes liveness-probe, балансировщик,
мониторинг.
"""

import logging

from fastapi import APIRouter

from mlops_phishing.schemas import LivenessResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["system"])

LIVENESS_PATH = "/healthz"


@router.get(
    LIVENESS_PATH,
    summary="Liveness-проверка",
    description="Процесс жив и event loop не заблокирован. Без обращений к зависимостям.",
)
async def liveness() -> LivenessResponse:
    """Вернуть признак живости процесса."""
    return LivenessResponse()
