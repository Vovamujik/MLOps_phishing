"""Инференс: проверка URL на фишинг моделью, загруженной при старте."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, status

from mlops_phishing.ml import LoadedModel
from mlops_phishing.schemas import ProcessRequest, ProcessResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])


@router.post(
    "/process",
    summary="Проверить URL на фишинг",
    description="Инференс моделью из MLflow Model Registry, загруженной при старте приложения.",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Модель не загружена"}},
)
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    """Вернуть вероятность фишинга и версию модели, которая её посчитала."""
    model: LoadedModel | None = request.app.state.model
    if model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена, причина — в логе старта приложения",
        )

    # predict_proba — синхронная CPU-работа: в потоке, чтобы не блокировать event loop
    probability = await asyncio.to_thread(model.phishing_probability, payload.url)
    threshold = request.app.state.settings.phishing_threshold
    is_phishing = probability >= threshold
    log.info("Модель v%s: is_phishing=%s, p=%.3f", model.version, is_phishing, probability)

    return ProcessResponse(
        url=payload.url,
        is_phishing=is_phishing,
        phishing_probability=round(probability, 4),
        threshold=threshold,
        model_name=model.name,
        model_version=model.version,
        model_alias=model.alias,
    )
