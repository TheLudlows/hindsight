"""AML wire contract through real FastAPI routes, with a controlled engine."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from hindsight_api.api.admission import AdmissionController, LaneConfig
from hindsight_api.engine.memory_engine import RetainOperationConflictError
from hindsight_api.extensions.builtin.aml import AmlHttpExtension, aml_bank_id
from hindsight_api.extensions.operation_validator import OperationValidationError


def add_body(**changes):
    return {
        "request_id": "req:1",
        "user_id": "user:1",
        "session_id": "session:1",
        "messages": [
            {"role": "user", "content": "Alice moved to Berlin", "timestamp": 1704067200123},
            {"role": "assistant", "content": "Noted"},
        ],
        **changes,
    }


def make_app(memory, **settings):
    app = FastAPI()
    app.state.admission = AdmissionController({})
    extension = AmlHttpExtension({"aml_api_key": "secret", "aml_poll_interval": "0.001", **settings})
    app.include_router(extension.get_router(memory), prefix="/ext")
    return app


@pytest.fixture
def engine():
    return SimpleNamespace(
        submit_async_retain=AsyncMock(return_value={"operation_id": "op", "items_count": 1}),
        get_operation_status=AsyncMock(return_value={"status": "completed"}),
        recall_async=AsyncMock(return_value=SimpleNamespace(results=[])),
    )


@pytest.mark.asyncio
async def test_add_echoes_ids_preserves_message_order_and_source_time(engine):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post("/ext/aml/add", json=add_body(), headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json() == {"success": True, "request_id": "req:1", "user_id": "user:1", "session_id": "session:1"}
    call = engine.submit_async_retain.call_args
    assert call.args[0] == aml_bank_id("aml-textual-v1", "user:1")
    item = call.args[1][0]
    import json

    assert json.loads(item["content"]) == [
        {"role": "user", "content": "Alice moved to Berlin", "timestamp": "2024-01-01T00:00:00.123000+00:00"},
        {"role": "assistant", "content": "Noted", "timestamp": None},
    ]
    assert item["event_date"].isoformat() == "2024-01-01T00:00:00.123000+00:00"
    assert item["metadata"] == {"request_id": "req:1", "session_id": "session:1"}
    assert call.kwargs["request_context"].api_key == "secret"
    assert call.kwargs["request_context"].internal is False


@pytest.mark.asyncio
async def test_add_waits_until_completed_and_does_not_ack_queueing(engine):
    ready = asyncio.Event()
    polled = asyncio.Event()

    async def status(*args, **kwargs):
        polled.set()
        return {"status": "completed" if ready.is_set() else "pending"}

    engine.get_operation_status.side_effect = status
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        task = asyncio.create_task(client.post("/ext/aml/add", json=add_body(), headers={"X-Api-Key": "secret"}))
        await asyncio.wait_for(polled.wait(), timeout=5)
        assert not task.done()
        ready.set()
        response = await asyncio.wait_for(task, timeout=5)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_replay_identity_is_stable_but_each_chunk_gets_its_own_document(engine):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        for body in [add_body(), add_body(), add_body(request_id="req:2"), add_body(user_id="user:2")]:
            assert (await client.post("/ext/aml/add", json=body, headers={"X-Api-Key": "secret"})).status_code == 200
    a, retry, chunk, other = engine.submit_async_retain.call_args_list
    assert a.kwargs["operation_id"] == retry.kwargs["operation_id"]
    assert a.kwargs["request_fingerprint"] == retry.kwargs["request_fingerprint"]
    assert a.args[0] == chunk.args[0]
    assert a.args[1][0]["document_id"] != chunk.args[1][0]["document_id"]
    assert a.args[0] != other.args[0]
    assert a.kwargs["operation_id"] != other.kwargs["operation_id"]


@pytest.mark.asyncio
async def test_changed_body_is_bound_to_same_identity_and_conflict_is_reported(engine):
    engine.submit_async_retain.side_effect = RetainOperationConflictError("private details")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post(
            "/ext/aml/add", json=add_body(session_id="changed"), headers={"X-Api-Key": "secret"}
        )
    assert response.status_code == 409
    assert "private details" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "not_found"])
async def test_terminal_unsuccessful_job_never_returns_success(engine, status):
    engine.get_operation_status.return_value = {"status": status}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post("/ext/aml/add", json=add_body(), headers={"X-Api-Key": "secret"})
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_timeout_returns_retryable_failure_without_cancelling_persisted_job(engine):
    engine.get_operation_status.return_value = {"status": "pending"}
    app = make_app(engine, aml_add_timeout="0.01")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ext/aml/add", json=add_body(), headers={"X-Api-Key": "secret"})
    assert response.status_code == 504
    assert engine.submit_async_retain.await_count == 1


@pytest.mark.asyncio
async def test_search_returns_sorted_evidence_and_applies_top_k(engine):
    engine.recall_async.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(id=str(i), text=f"fact {i}", scores=SimpleNamespace(final=1 - i / 10), mentioned_at=None)
            for i in range(4)
        ]
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post(
            "/ext/aml/search",
            json={"user_id": "user:1", "query": "Where?", "options": ["A. Berlin", "B. Paris"], "top_k": 2},
            headers={"Authorization": "Token secret"},
        )
    assert response.status_code == 200
    assert [(item["id"], item["content"], item["score"]) for item in response.json()["data"]] == [
        ("0", "fact 0", 1),
        ("1", "fact 1", 0.9),
    ]
    assert engine.recall_async.call_args.kwargs["query"] == "Where?"
    assert engine.recall_async.call_args.kwargs["bank_id"] == aml_bank_id("aml-textual-v1", "user:1")


@pytest.mark.asyncio
async def test_empty_search_is_a_data_object(engine):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post(
            "/ext/aml/search",
            json={"user_id": "unknown", "query": "Where?", "top_k": 100},
            headers={"X-Api-Key": "secret"},
        )
    assert response.json() == {"data": []}


@pytest.mark.asyncio
async def test_missing_bank_is_an_empty_aml_search(engine):
    engine.recall_async.side_effect = OperationValidationError("Bank not found", status_code=404)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post(
            "/ext/aml/search",
            json={"user_id": "unknown", "query": "q", "top_k": 100},
            headers={"X-Api-Key": "secret"},
        )
    assert response.status_code == 200
    assert response.json() == {"data": []}
    engine.submit_async_retain.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"X-Api-Key": "wrong"}])
async def test_authentication_precedes_engine_access(engine, headers):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post("/ext/aml/add", json=add_body(), headers=headers)
    assert response.status_code == 401
    engine.submit_async_retain.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"messages": []},
        {"user_id": " "},
        {"messages": [{"role": "system", "content": "x"}]},
        {"messages": [{"role": "user", "content": " "}]},
        {"messages": [{"role": "user", "content": "x", "timestamp": True}]},
        {"messages": [{"role": "user", "content": "x", "timestamp": 10**100}]},
    ],
)
async def test_invalid_add_contract_is_rejected(engine, change):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post("/ext/aml/add", json=add_body(**change), headers={"X-Api-Key": "secret"})
    assert response.status_code == 422
    engine.submit_async_retain.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [0, -1, True, "100", 101])
async def test_invalid_top_k_is_rejected(engine, top_k):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post(
            "/ext/aml/search", json={"query": "q", "user_id": "u", "top_k": top_k}, headers={"X-Api-Key": "secret"}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_engine_authorization_is_not_bypassed(engine):
    engine.recall_async.side_effect = OperationValidationError("bank denied", status_code=403)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post(
            "/ext/aml/search", json={"query": "q", "user_id": "u", "top_k": 1}, headers={"X-Api-Key": "secret"}
        )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_extension_shares_native_admission_budget(engine):
    app = make_app(engine)
    app.state.admission = AdmissionController({"recall": LaneConfig(max_in_flight=1, max_wait_seconds=0)})
    async with app.state.admission.admit("recall"):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/ext/aml/search", json={"query": "q", "user_id": "u", "top_k": 1}, headers={"X-Api-Key": "secret"}
            )
    assert response.status_code == 503
    engine.recall_async.assert_not_called()


def test_namespace_and_user_encoding_are_unambiguous():
    assert aml_bank_id("a:b", "c") != aml_bank_id("a", "b:c")
    assert aml_bank_id("v1", " user ") != aml_bank_id("v1", "user")
    assert aml_bank_id("v1", "用户") == aml_bank_id("v1", "用户")


def test_missing_key_fails_at_extension_startup():
    with pytest.raises(ValidationError):
        AmlHttpExtension({})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored,requested,conflict", [("a", "a", False), ("a", "b", True), (None, "a", True), ("a", None, False)]
)
async def test_engine_fingerprint_replay_guard(monkeypatch, stored, requested, conflict):
    from contextlib import asynccontextmanager
    import uuid
    from hindsight_api.engine import memory_engine

    connection = SimpleNamespace(
        fetchrow=AsyncMock(
            return_value={
                "bank_id": "bank",
                "operation_type": "batch_retain",
                "result_metadata": {"items_count": 1, "request_fingerprint": stored},
            }
        )
    )

    @asynccontextmanager
    async def acquire(backend):
        yield connection

    monkeypatch.setattr(memory_engine, "acquire_with_retry", acquire)
    engine = SimpleNamespace(_get_backend=AsyncMock(return_value=object()))
    if conflict:
        with pytest.raises(RetainOperationConflictError):
            await memory_engine.MemoryEngine._resolve_retain_replay(engine, uuid.uuid4(), "bank", requested)
    else:
        result = await memory_engine.MemoryEngine._resolve_retain_replay(engine, uuid.uuid4(), "bank", requested)
        assert result["items_count"] == 1


@pytest.mark.asyncio
async def test_partial_extraction_failure_is_not_acknowledged(engine):
    engine.get_operation_status.side_effect = [
        {"status": "completed", "child_operations": [{"operation_id": "child", "status": "completed"}]},
        {"status": "completed", "result_metadata": {"extraction_errors_count": 1}},
    ]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post("/ext/aml/add", json=add_body(), headers={"X-Api-Key": "secret"})
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_cancelled_child_does_not_wait_for_parent_aggregation(engine):
    engine.get_operation_status.return_value = {
        "status": "pending",
        "child_operations": [{"operation_id": "child", "status": "cancelled"}],
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(engine)), base_url="http://test") as client:
        response = await client.post("/ext/aml/add", json=add_body(), headers={"X-Api-Key": "secret"})
    assert response.status_code == 503
