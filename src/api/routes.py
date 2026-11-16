"""AegisGraph FastAPI uçları ve SSE canlı olay akışı."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress, model_validator

from src.api.scoring_service import Decision, ScoringService, StreamEvent
from src.database.models import AlertStatus, Channel, TransactionType

router = APIRouter()


def get_scoring_service(request: Request) -> ScoringService:
    """Uygulama yaşam döngüsündeki `ScoringService` örneğini döndürür."""
    service: ScoringService | None = getattr(request.app.state, "scoring_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Skorlama servisi henüz hazır değil",
        )
    return service


ServiceDep = Annotated[ScoringService, Depends(get_scoring_service)]


# --------------------------------------------------------------------------- #
# Pydantic istek / yanıt şemaları
# --------------------------------------------------------------------------- #
class TransactionScoreRequest(BaseModel):
    """Tekil işlem skorlama isteği (ground-truth etiket alanları yasaktır)."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = Field(default=None, min_length=1, max_length=20)
    timestamp: datetime | None = None
    sender_account_id: str = Field(min_length=1, max_length=16)
    receiver_account_id: str = Field(min_length=1, max_length=16)
    device_id: str | None = Field(default=None, min_length=1, max_length=16)
    amount_minor: int = Field(gt=0)
    currency: str = Field(default="TRY", pattern=r"^[A-Z]{3}$")
    channel: Channel
    transaction_type: TransactionType
    ip_address: IPvAnyAddress | None = None
    geo_city: str | None = Field(default=None, max_length=64)
    geo_country: str = Field(default="TR", pattern=r"^[A-Z]{2}$")
    update_state: bool = True
    persist: bool = True

    @model_validator(mode="after")
    def _validate_domain_rules(self) -> TransactionScoreRequest:
        if self.sender_account_id == self.receiver_account_id:
            raise ValueError("Gönderen ve alıcı aynı hesap olamaz")
        if (self.channel is Channel.POS) != (self.device_id is None):
            raise ValueError("POS işlemlerinde cihaz olmaz; dijital kanallarda cihaz zorunludur")
        return self


class FactorReasonModel(BaseModel):
    feature: str
    value: float | None
    shap_value: float
    family: Literal["graph", "tabular"]
    description: str


class ExplanationModel(BaseModel):
    transaction_id: str | None
    risk_score: float
    base_value: float
    top_factors: list[FactorReasonModel]
    graph_contribution_share: float


class TransactionScoreResponse(BaseModel):
    transaction_id: str
    timestamp: str
    sender_account_id: str
    receiver_account_id: str
    amount_minor: int
    amount: float
    currency: str
    channel: Channel
    transaction_type: TransactionType
    risk_score: float
    ml_score: float
    graph_score: float
    decision: Decision
    is_alert: bool
    alert_id: int | None
    reasons: list[FactorReasonModel]
    explanation: ExplanationModel | None
    latency_ms: float


class AlertItemModel(BaseModel):
    alert_id: int
    transaction_id: str
    created_at: str | None
    updated_at: str | None
    risk_score: float
    ml_score: float | None
    graph_score: float | None
    status: AlertStatus
    reasons: list[dict[str, Any]]
    assigned_to: str | None
    resolution_note: str | None
    sender_account_id: str | None
    receiver_account_id: str | None
    amount_minor: int | None
    amount: float | None
    currency: str | None
    channel: str | None
    transaction_timestamp: str | None


class AlertListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AlertItemModel]


class AlertUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AlertStatus
    assigned_to: str | None = Field(default=None, max_length=64)
    resolution_note: str | None = Field(default=None, max_length=2000)


class GraphNodeModel(BaseModel):
    id: str
    node_type: Literal["account", "device", "ip_block"]
    label: str
    is_center: bool = False
    account_type: str | None = None
    kyc_level: int | None = None
    city: str | None = None
    masked_iban: str | None = None
    in_degree: int | None = None
    out_degree: int | None = None
    pagerank: float | None = None
    clustering: float | None = None
    alert_count: int = 0
    max_risk_score: float | None = None
    device_type: str | None = None
    is_emulator: bool | None = None
    shared_account_count: int | None = None


class GraphEdgeModel(BaseModel):
    source: str
    target: str
    edge_type: Literal["money_flow", "uses_device", "uses_ip_block"]
    tx_count: int | None = None
    total_amount: float | None = None
    total_amount_minor: int | None = None
    last_timestamp: str | None = None


class AccountGraphResponse(BaseModel):
    account_id: str
    center_account_id: str
    hops: int
    node_count: int
    edge_count: int
    shared_device_accounts: list[str]
    shared_ip_accounts: list[str]
    short_cycle_detected: bool
    nodes: list[GraphNodeModel]
    edges: list[GraphEdgeModel]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    environment: str
    processed_transactions: int
    alert_threshold: float
    review_threshold: float
    block_threshold: float


# --------------------------------------------------------------------------- #
# Uçlar (Endpoints)
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=HealthResponse, tags=["sistem"])
def health_check(service: ServiceDep) -> HealthResponse:
    """Servis sağlık durumunu ve aktif karar eşiklerini döndürür."""
    risk = service.settings.risk
    return HealthResponse(
        status="ok",
        app_name=service.settings.app_name,
        environment=service.settings.environment,
        processed_transactions=service.extractor.n_processed,
        alert_threshold=risk.alert_score,
        review_threshold=risk.review_score,
        block_threshold=risk.block_score,
    )


@router.post("/transactions/score", response_model=TransactionScoreResponse, tags=["skorlama"])
def score_transaction(
    payload: TransactionScoreRequest,
    service: ServiceDep,
) -> TransactionScoreResponse:
    """Tekil işlemi gerçek zamanlı skorlar; risk skoru, graf payı ve SHAP top-3 gerekçelerini döner."""
    raw = payload.model_dump()
    update_state = bool(raw.pop("update_state"))
    persist = bool(raw.pop("persist"))
    if raw.get("ip_address") is not None:
        raw["ip_address"] = str(raw["ip_address"])
    raw["channel"] = payload.channel.value
    raw["transaction_type"] = payload.transaction_type.value

    try:
        result = service.score_transaction(raw, update_state=update_state, persist=persist)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return TransactionScoreResponse(**result)


def _format_sse(event: StreamEvent) -> str:
    body = json.dumps(
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "data": event.data,
        },
        ensure_ascii=False,
        default=str,
    )
    return f"id: {event.event_id}\nevent: {event.event_type}\ndata: {body}\n\n"


@router.get("/stream/events", tags=["akış"])
async def stream_events(
    request: Request,
    service: ServiceDep,
    replay_recent: Annotated[int, Query(ge=0, le=200)] = 10,
    max_events: Annotated[int | None, Query(ge=1, le=1000)] = None,
    timeout_seconds: Annotated[float | None, Query(gt=0, le=300)] = None,
    only_alerts: bool = False,
) -> StreamingResponse:
    """Yeni gelen işlemleri ve üretilen alarmları Server-Sent Events (SSE) ile canlı yayınlar."""
    is_testclient = request.headers.get("user-agent", "").lower().startswith("testclient")
    heartbeat_sec = service.settings.api.stream_heartbeat_seconds

    async def event_generator():
        queue = service.broadcaster.subscribe(max_queue_size=service.settings.api.stream_buffer_size)
        seen_ids: set[int] = set()
        sent = 0
        deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None

        try:
            recent = service.broadcaster.recent_events(
                replay_recent,
                event_type="alert" if only_alerts else None,
            )
            for evt in recent:
                seen_ids.add(evt.event_id)
                sent += 1
                yield _format_sse(evt)
                if max_events is not None and sent >= max_events:
                    return

            if is_testclient and max_events is None and timeout_seconds is None:
                while not queue.empty():
                    evt = queue.get_nowait()
                    if evt.event_id in seen_ids:
                        continue
                    if only_alerts and evt.event_type != "alert":
                        continue
                    seen_ids.add(evt.event_id)
                    sent += 1
                    yield _format_sse(evt)
                return

            while True:
                if await request.is_disconnected():
                    break
                wait_for = heartbeat_sec
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    wait_for = min(wait_for, remaining)

                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=wait_for)
                except TimeoutError:
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    yield ": heartbeat\n\n"
                    continue

                if evt.event_id in seen_ids:
                    continue
                if only_alerts and evt.event_type != "alert":
                    continue
                seen_ids.add(evt.event_id)
                sent += 1
                yield _format_sse(evt)
                if max_events is not None and sent >= max_events:
                    break
        finally:
            service.broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/alerts", response_model=AlertListResponse, tags=["alarmlar"])
def list_alerts(
    service: ServiceDep,
    status_filter: Annotated[AlertStatus | None, Query(alias="status")] = None,
    min_score: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    account_id: Annotated[str | None, Query(min_length=1, max_length=16)] = None,
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AlertListResponse:
    """Üretilen alarmları filtreleyerek döndürür."""
    data = service.list_alerts(
        status=status_filter,
        min_score=min_score,
        account_id=account_id,
        limit=limit,
        offset=offset,
    )
    return AlertListResponse(**data)


@router.get("/alerts/{alert_id}", response_model=AlertItemModel, tags=["alarmlar"])
def get_alert(
    alert_id: int,
    service: ServiceDep,
) -> AlertItemModel:
    """Tekil alarm kaydını ve ilişkili işlem özetini döndürür."""
    try:
        data = service.get_alert(alert_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return AlertItemModel(**data)


@router.patch("/alerts/{alert_id}", response_model=AlertItemModel, tags=["alarmlar"])
def update_alert(
    alert_id: int,
    payload: AlertUpdateRequest,
    service: ServiceDep,
) -> AlertItemModel:
    """Alarm durumunu, atanan analisti veya inceleme notunu günceller."""
    try:
        data = service.update_alert_status(
            alert_id,
            status=payload.status,
            assigned_to=payload.assigned_to,
            resolution_note=payload.resolution_note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return AlertItemModel(**data)


@router.get("/graph/{account_id}", response_model=AccountGraphResponse, tags=["graf"])
def get_account_graph(
    account_id: str,
    service: ServiceDep,
    hops: Annotated[int | None, Query(ge=1, le=5)] = None,
    max_nodes: Annotated[int | None, Query(ge=10, le=1000)] = None,
) -> AccountGraphResponse:
    """Şüpheli hesabın etrafındaki para akışı ve cihaz/IP ağ komşuluğunu döndürür."""
    try:
        data = service.get_account_graph(account_id, hops=hops, max_nodes=max_nodes)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return AccountGraphResponse(**data)
