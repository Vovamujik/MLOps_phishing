"""End-to-end health-эндпоинт."""

import logging

from fastapi import APIRouter, Request, Response, status

from mlops_phishing.schemas import HealthReport, ReportStatus
from mlops_phishing.services import health as health_service

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="End-to-end health-check",
    description="Версии third-party компонентов и время ответа каждого из них.",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Зависимость недоступна"}},
)
async def read_health(request: Request, response: Response) -> HealthReport:
    """Собрать и вернуть отчёт о состоянии зависимостей.

    При деградации возвращается 503: мониторинг и балансировщик смотрят на
    статус-код, а не разбирают тело ответа.
    """
    report = await health_service.build_report(
        request.app.state.db_pool,
        request.app.state.settings,
    )
    if report.status is not ReportStatus.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("Health-отчёт: %s", report.status.value)
    return report
