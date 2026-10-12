"""AML against a real API, worker and database, including a process restart.

Optional extension routes do not belong to the base generated SDK. Exercise the
external AML contract through the published low-level ApiClient, then verify
durable state and native recall through Hindsight. No engine imports or SQL.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid

import httpx
import pytest
from hindsight_client import Hindsight
from hindsight_client_api import ApiClient, Configuration

from hindsight_system_tests import start_hindsight_server, wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


class _AmlClient(ApiClient):
    """Use the SDK's serialization/transport for optional extension routes."""

    def __init__(self, base_url: str, header_name: str, header_value: str) -> None:
        super().__init__(Configuration(host=base_url))
        self.set_default_header(header_name, header_value)

    async def post(self, path: str, *, json: dict) -> httpx.Response:
        serialized = self.param_serialize("POST", path, body=json, header_params={"Content-Type": "application/json"})
        response = await self.call_api(*serialized, _request_timeout=60)
        # httpx only supplies a response parser here, not a second transport.
        return httpx.Response(response.status, content=await response.read())


async def test_aml_chunks_replay_scope_and_restart(stub_server, llm, tmp_path):
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cellist").returns(
        extracted(fact("Alice is a cellist", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("extract_facts", contains="Paris").returns(
        extracted(fact("Bob lives in Paris", who="Bob", entities=["Bob", "Paris"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    namespace = "systest-aml-" + uuid.uuid4().hex
    users = ["alice", "bob"]
    # This is the documented public mapping, not an engine/internal helper.
    banks = [
        "aml-"
        + hashlib.sha256(json.dumps([namespace, user], ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
        for user in users
    ]
    env = {
        "HINDSIGHT_API_HTTP_EXTENSION": "hindsight_api.extensions.builtin.aml:AmlHttpExtension",
        "HINDSIGHT_API_HTTP_AML_API_KEY": "aml-story-key",
        "HINDSIGHT_API_HTTP_AML_NAMESPACE": namespace,
        "HINDSIGHT_API_HTTP_AML_POLL_INTERVAL": "0.05",
    }
    server = start_hindsight_server(stub_url=stub_server.url, log_path=tmp_path / "first.log", extra_env=env)
    native = Hindsight(base_url=server.url)
    alice_body = {
        "request_id": "chunk-1",
        "session_id": "session-1",
        "user_id": users[0],
        "messages": [{"role": "user", "content": "Alice lives in Berlin", "timestamp": 1704067200000}],
    }
    try:
        async with _AmlClient(server.url, "Authorization", "Bearer aml-story-key") as aml:
            first, replay = await asyncio.gather(
                aml.post("/ext/aml/add", json=alice_body), aml.post("/ext/aml/add", json=alice_body)
            )
            expected_ack = {"success": True, "request_id": "chunk-1", "session_id": "session-1", "user_id": users[0]}
            assert first.status_code == replay.status_code == 200
            assert first.json() == replay.json() == expected_ack
            # Immediate Search, before settled(), proves the Add acknowledgement
            # means retrievable facts rather than acceptance into a worker queue.
            search = await aml.post(
                "/ext/aml/search", json={"user_id": users[0], "query": "Where does Alice live?", "top_k": 100}
            )
            assert search.status_code == 200
            assert [item["content"] for item in search.json()["data"]] == ["Alice lives in Berlin | Involving: Alice"]
            await wait_until_settled(native, banks[0])
            parents = await native.operations.list_operations(banks[0], limit=100)
            assert sum(op.task_type == "batch_retain" for op in parents.operations) == 1
            changed = {**alice_body, "messages": [{"role": "user", "content": "different"}]}
            assert (await aml.post("/ext/aml/add", json=changed)).status_code == 409

            chunk2 = {
                **alice_body,
                "request_id": "chunk-2",
                "messages": [{"role": "user", "content": "Alice is a cellist"}],
            }
            assert (await aml.post("/ext/aml/add", json=chunk2)).status_code == 200
            bob = {**alice_body, "user_id": users[1], "messages": [{"role": "user", "content": "Bob lives in Paris"}]}
            assert (await aml.post("/ext/aml/add", json=bob)).status_code == 200
            for bank in banks:
                await wait_until_settled(native, bank)
            native_recall = await native.arecall(banks[0], "Alice", max_tokens=8192, budget="high")
            expected_contents = [item.text for item in native_recall.results]
            assert set(expected_contents) == {
                "Alice lives in Berlin | Involving: Alice",
                "Alice is a cellist | Involving: Alice",
            }
            assert len({item.document_id for item in native_recall.results}) == 2
            assert all(item.metadata["session_id"] == "session-1" for item in native_recall.results)

        await native.aclose()
        server.stop()
        server = start_hindsight_server(stub_url=stub_server.url, log_path=tmp_path / "restart.log", extra_env=env)
        native = Hindsight(base_url=server.url)
        async with _AmlClient(server.url, "X-Api-Key", "aml-story-key") as aml:
            assert (await aml.post("/ext/aml/add", json=alice_body)).json() == expected_ack
            parents = await native.operations.list_operations(banks[0], limit=100)
            assert sum(op.task_type == "batch_retain" for op in parents.operations) == 2
            response = await aml.post("/ext/aml/search", json={"query": "Alice", "user_id": users[0], "top_k": 100})
            assert [item["content"] for item in response.json()["data"]] == expected_contents
            bob_response = await aml.post("/ext/aml/search", json={"query": "Bob", "user_id": users[1], "top_k": 100})
            assert [item["content"] for item in bob_response.json()["data"]] == ["Bob lives in Paris | Involving: Bob"]
            empty = await aml.post("/ext/aml/search", json={"query": "Alice", "user_id": "unknown", "top_k": 100})
            assert empty.status_code == 200
            assert empty.json() == {"data": []}
    finally:
        for bank in banks:
            await native.banks.delete_bank(bank)
        await native.aclose()
        server.stop()
