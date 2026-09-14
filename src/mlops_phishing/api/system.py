"""Системные (операционные) эндпоинты.

Живут вне `/api/v1` намеренно. Это не часть публичного API-контракта, а
интерфейс для оркестратора: kubernetes liveness-probe, балансировщик,
мониторинг.

1. Эндпоинт не версионируется. Он должен оставаться по одному адресу вечно:
   при переходе на `/api/v2` никто не станет править манифесты k8s.
2. Эндпоинт не ходит во внешние системы. Если проверять здесь базу, то при
   её аварии оркестратор начнёт перезапускать полностью здоровые поды.

Диагностика зависимостей живёт отдельно, в `/api/v1/health`.
"""

import logging

from fastapi import APIRouter

from mlops_phishing.schemas import LivenessResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get(
    "/healthz",
    summary="Liveness-проверка",
    description="Процесс жив и event loop не заблокирован. Без обращений к зависимостям.",
)
async def liveness() -> LivenessResponse:
    """Вернуть признак живости процесса."""
    return LivenessResponse()
