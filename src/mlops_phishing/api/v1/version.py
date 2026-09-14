"""Эндпоинт версии приложения."""

import logging

from fastapi import APIRouter

from mlops_phishing.config import get_settings
from mlops_phishing.metadata import get_build_time, get_git_commit, get_version
from mlops_phishing.schemas import VersionResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["meta"])


@router.get(
    "/version",
    summary="Версия приложения",
    description="Версия из pyproject.toml плюс коммит и время сборки образа.",
)
async def read_version() -> VersionResponse:
    """Вернуть версию приложения и метаданные сборки.

    Откуда берётся версия — см. `metadata.get_version`.
    """
    settings = get_settings()
    return VersionResponse(
        name=settings.app_name,
        version=get_version(),
        git_commit=get_git_commit(),
        build_time=get_build_time(),
    )
