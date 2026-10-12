"""Optional Textual AML contract adapter, mounted through HttpExtension.

Protocol translation stays here; persistence, authorization, extraction and
retrieval remain in the ordinary engine. No external-id mapping table is needed.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints, field_validator

from hindsight_api.api.admission import AdmissionRejected
from hindsight_api.api.passthrough_headers import collect_passthrough_headers
from hindsight_api.config import get_config
from hindsight_api.engine.memory_engine import Budget, RetainOperationConflictError
from hindsight_api.extensions.http import HttpExtension
from hindsight_api.extensions.operation_validator import OperationValidationError
from hindsight_api.extensions.tenant import AuthenticationError
from hindsight_api.models import RequestContext

if TYPE_CHECKING:
    from hindsight_api import MemoryEngine

NonEmptyText = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
ExternalId = Annotated[NonEmptyText, Field(max_length=2048)]
_IDENTITY_NAMESPACE = uuid.UUID("ec2aa6b6-467c-4b95-9637-9130e7b63d10")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AmlMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: NonEmptyText
    timestamp: Annotated[int, Field(strict=True)] | None = None

    @field_validator("timestamp")
    @classmethod
    def valid_milliseconds(cls, value: int | None) -> int | None:
        if value is not None:
            try:
                _EPOCH + timedelta(milliseconds=value)
            except OverflowError as exc:
                raise ValueError("timestamp is outside the supported datetime range") from exc
        return value


class AmlAddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: ExternalId
    user_id: ExternalId
    session_id: ExternalId
    messages: Annotated[list[AmlMessage], Field(min_length=1)]


class AmlAddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class AmlSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: NonEmptyText
    user_id: ExternalId
    top_k: Annotated[int, Field(strict=True, ge=1, le=100)]
    options: list[NonEmptyText] | None = None


class AmlMemory(BaseModel):
    id: NonEmptyText
    content: NonEmptyText
    score: Annotated[float, Field(allow_inf_nan=False)] | None = None
    created_at: str | None = None


class AmlSearchResponse(BaseModel):
    data: list[AmlMemory]


class AmlSettings(BaseModel):
    # These are extension settings, loaded by the existing HTTP_* env loader.
    # Credentials are mandatory even with the no-auth single-tenant backend.
    aml_api_key: SecretStr
    aml_namespace: NonEmptyText = "aml-textual-v1"
    aml_add_timeout: float = Field(default=1770, gt=0, le=1800, allow_inf_nan=False)
    aml_poll_interval: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    aml_search_budget: Budget = Budget.HIGH
    aml_search_max_tokens: int = Field(default=8192, gt=0)

    @field_validator("aml_api_key")
    @classmethod
    def nonempty_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("aml_api_key is required")
        return value


class _RetainedMessage(BaseModel):
    role: str
    content: str
    timestamp: str | None


class _RetainItem(BaseModel):
    content: str
    event_date: datetime | None
    document_id: str
    context: str
    metadata: dict[str, str]
    observation_scopes: Literal["shared"] = "shared"


class _Submission(BaseModel):
    operation_id: str
    items_count: int


class _ChildStatus(BaseModel):
    operation_id: str
    status: str


class _OperationMetadata(BaseModel):
    extraction_errors_count: int = 0


class _OperationStatus(BaseModel):
    status: Literal["pending", "processing", "completed", "failed", "cancelled", "not_found"]
    child_operations: list[_ChildStatus] = Field(default_factory=list)
    result_metadata: _OperationMetadata = Field(default_factory=_OperationMetadata)


def aml_bank_id(namespace: str, user_id: str) -> str:
    """Unambiguous encoding preserves the exact user scope without a lookup table."""
    identity = json.dumps([namespace, user_id], ensure_ascii=True, separators=(",", ":"))
    return "aml-" + hashlib.sha256(identity.encode()).hexdigest()


class AmlHttpExtension(HttpExtension):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self.settings = AmlSettings.model_validate(config)

    def get_router(self, memory: MemoryEngine) -> APIRouter:
        router = APIRouter(prefix="/aml", tags=["AML Textual"])
        settings = self.settings

        def authenticate(request: Request) -> RequestContext:
            authorization = request.headers.get("authorization", "")
            scheme, _, credential = authorization.partition(" ")
            key = credential.strip() if scheme.lower() in ("token", "bearer") else ""
            key = key or request.headers.get("x-api-key", "")
            if not hmac.compare_digest(key.encode(), settings.aml_api_key.get_secret_value().encode()):
                raise HTTPException(status_code=401, detail="Invalid memory system key")
            # Thread the validated key through normal tenant auth/authorization;
            # never set internal=True or substitute the default tenant schema.
            return RequestContext(
                api_key=key,
                extra_headers=collect_passthrough_headers(
                    request.headers.raw, get_config().extension_passthrough_headers
                ),
            )

        @asynccontextmanager
        async def engine_call(request: Request, lane: str) -> AsyncIterator[None]:
            try:
                # Share the native route's process budget rather than opening a
                # second unbounded lane through the extension endpoints.
                async with request.app.state.admission.admit(lane):
                    yield
            except AdmissionRejected as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Memory service is at capacity",
                    headers={"Retry-After": str(exc.retry_after_seconds)},
                ) from exc
            except RetainOperationConflictError as exc:
                raise HTTPException(status_code=409, detail="request_id was used with different content") from exc
            except OperationValidationError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.reason) from exc
            except AuthenticationError as exc:
                raise HTTPException(status_code=401, detail="Memory service authentication failed") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid memory operation") from exc

        @router.post("/add", response_model=AmlAddResponse, operation_id="aml_add")
        async def add(
            body: AmlAddRequest, request: Request, context: RequestContext = Depends(authenticate)
        ) -> AmlAddResponse:
            bank_id = aml_bank_id(settings.aml_namespace, body.user_id)
            logical_id = json.dumps([settings.aml_namespace, body.user_id, body.request_id], ensure_ascii=True)
            operation_id = str(uuid.uuid5(_IDENTITY_NAMESPACE, logical_id))
            fingerprint = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
            messages = [
                _RetainedMessage(
                    role=message.role,
                    content=message.content,
                    timestamp=(_EPOCH + timedelta(milliseconds=message.timestamp)).isoformat()
                    if message.timestamp is not None
                    else None,
                )
                for message in body.messages
            ]
            # One document per logical Add: session_id alone would overwrite
            # earlier AML chunks. Keep ordered messages together for attribution.
            dated = [message.timestamp for message in body.messages if message.timestamp is not None]
            item = _RetainItem(
                content="[" + ",".join(message.model_dump_json() for message in messages) + "]",
                event_date=_EPOCH + timedelta(milliseconds=dated[-1]) if dated else None,
                document_id="aml-" + operation_id,
                context="AML session " + body.session_id,
                metadata={"request_id": body.request_id, "session_id": body.session_id},
            )
            async with engine_call(request, "retain"):
                try:
                    async with asyncio.timeout(settings.aml_add_timeout):
                        submission = _Submission.model_validate(
                            await memory.submit_async_retain(
                                bank_id,
                                [item.model_dump()],
                                request_context=context,
                                operation_id=operation_id,
                                request_fingerprint=fingerprint,
                            )
                        )
                        while True:
                            status = _OperationStatus.model_validate(
                                await memory.get_operation_status(
                                    bank_id, submission.operation_id, request_context=context
                                )
                            )
                            if status.status in ("failed", "cancelled", "not_found") or any(
                                child.status in ("failed", "cancelled") for child in status.child_operations
                            ):
                                raise HTTPException(status_code=503, detail="Memory ingestion did not complete")
                            if status.status == "completed":
                                # Native retain can finish with recorded partial
                                # extraction errors. Never acknowledge lost facts,
                                # even if the deployment's strict-error flag is off.
                                for child in status.child_operations:
                                    child_status = _OperationStatus.model_validate(
                                        await memory.get_operation_status(
                                            bank_id, child.operation_id, request_context=context
                                        )
                                    )
                                    if (
                                        child_status.status != "completed"
                                        or child_status.result_metadata.extraction_errors_count > 0
                                    ):
                                        raise HTTPException(status_code=503, detail="Memory ingestion was incomplete")
                                if status.result_metadata.extraction_errors_count > 0:
                                    raise HTTPException(status_code=503, detail="Memory ingestion was incomplete")
                                break
                            await asyncio.sleep(settings.aml_poll_interval)
                except TimeoutError as exc:
                    # The persisted job survives the waiting HTTP request. A
                    # retry joins that job, without submitting extraction twice.
                    raise HTTPException(status_code=504, detail="Memory ingestion is still pending") from exc
            return AmlAddResponse(request_id=body.request_id, user_id=body.user_id, session_id=body.session_id)

        @router.post("/search", response_model=AmlSearchResponse, operation_id="aml_search")
        async def search(
            body: AmlSearchRequest, request: Request, context: RequestContext = Depends(authenticate)
        ) -> AmlSearchResponse:
            async with engine_call(request, "recall"):
                try:
                    result = await memory.recall_async(
                        bank_id=aml_bank_id(settings.aml_namespace, body.user_id),
                        query=body.query,
                        budget=settings.aml_search_budget,
                        max_tokens=settings.aml_search_max_tokens,
                        request_context=context,
                    )
                except OperationValidationError as exc:
                    # A user with no Add has no Bank. Preserve the engine's
                    # auth/read checks, then translate an absent scope to AML's
                    # empty result (without creating a Bank during a read).
                    if exc.status_code == 404:
                        return AmlSearchResponse(data=[])
                    raise
            # Options are accepted but do not become an answer-generation prompt.
            # Use native ranking, not a guessed score, and bound the final list.
            return AmlSearchResponse(
                data=[
                    AmlMemory(
                        id=fact.id,
                        content=fact.text,
                        score=fact.scores.final if fact.scores else None,
                        created_at=fact.mentioned_at,
                    )
                    for fact in result.results[: body.top_k]
                ]
            )

        return router
