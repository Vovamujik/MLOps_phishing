"""Публикация задач из API: соединение с брокером, подтверждения, таймаут.

Соединение открывается при первой публикации или проверке здоровья, как пул БД
с min_size=0: API стартует и без RabbitMQ, а /api/v1/health покажет его
недоступным. После обрыва aio-pika переподключается сам.
"""

import asyncio
from typing import Protocol

from aio_pika import RobustConnection
from aio_pika.exceptions import ChannelInvalidStateError
from faststream.rabbit import RabbitBroker

from mlops_phishing.config import Settings
from mlops_phishing.messaging import PUBLISH_CHANNEL, TaskMessage, publish_task

# Брокер недоступен или не ответил вовремя. Ошибки соединения aio-pika — подклассы OSError
BROKER_ERRORS = (OSError, TimeoutError, ChannelInvalidStateError)


class TaskPublisher(Protocol):
    """Что API нужно от издателя. Протокол, чтобы в тестах подставлять заглушку."""

    async def publish(self, task: TaskMessage) -> None:
        """Отправить задачу и дождаться ответа брокера."""
        ...

    async def server_version(self) -> str:
        """Версия брокера: для health-отчёта."""
        ...

    async def close(self) -> None:
        """Закрыть соединение."""
        ...


class RabbitTaskPublisher:
    def __init__(self, settings: Settings) -> None:
        self.broker = RabbitBroker(settings.rabbitmq_url, default_channel=PUBLISH_CHANNEL)
        self.timeout = settings.publish_timeout
        self._connecting: asyncio.Future[RobustConnection] | None = None

    async def connect(self) -> RobustConnection:
        """Соединение с брокером: одна попытка на все запросы, новая — только после неудачи.

        Попытку не отменяем (shield), а лишь перестаём её ждать по таймауту вызывающего:
        отменённый connect_robust оставил бы соединение, которое переподключается само
        и не закрывается в close(). Запросы не выстраиваются в очередь за чужими попытками.
        """
        attempt = self._connecting
        failed = (
            attempt is not None
            and attempt.done()
            and (attempt.cancelled() or attempt.exception() is not None)
        )
        if attempt is None or failed:
            attempt = self._connecting = asyncio.ensure_future(self.broker.connect())
        return await asyncio.shield(attempt)

    async def publish(self, task: TaskMessage) -> None:
        """Отправить задачу и дождаться ответа брокера; соединение и ответ — в пределах таймаута.

        DeliveryError — очередь полна (nack), PublishError — очереди нет,
        BROKER_ERRORS — брокер недоступен или не ответил вовремя.
        """
        async with asyncio.timeout(self.timeout):
            await self.connect()
            await publish_task(self.broker, task)

    async def server_version(self) -> str:
        """Версия брокера после запроса к нему.

        Версия лежит в уже открытом соединении, но зависший брокер (memory alarm, пауза)
        соединение не рвёт ещё до минуты. Открыть и закрыть канал — запрос к брокеру:
        не ответил за таймаут — health покажет его недоступным.
        """
        async with asyncio.timeout(self.timeout):
            connection = await self.connect()
            transport = connection.transport
            # Пока aio-pika переподключается, транспорта нет или он уже закрыт: канал не открыть
            if transport is None or not connection.connected.is_set():
                raise ConnectionError("соединение с RabbitMQ восстанавливается")
            channel = await connection.channel()
            await channel.close()
        return str(transport.connection.server_properties["version"])

    async def close(self) -> None:
        if self._connecting is not None and not self._connecting.done():
            # При остановке ждать соединения незачем
            self._connecting.cancel()
        await self.broker.stop()
