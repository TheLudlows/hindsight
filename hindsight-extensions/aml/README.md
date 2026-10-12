# AML Textual adapter

An optional HTTP extension for Agent Memory Leaderboard (AML). It exposes
`POST /ext/aml/add` and `POST /ext/aml/search` on the existing Hindsight API.
Submit `/health` as the unauthenticated health URL. AML receives memory evidence;
its platform owns answer generation and scoring. This is independent of the AMB
quality harness in `hindsight-system-evals/`.

For the MemoHarbor commercial Textual submission, see the
[Chinese submission preparation checklist](SUBMISSION_PREPARATION.zh-CN.md)
for required materials, deployment readiness, and evaluation steps.

## Enable

Use a build containing this extension, including the API and a running worker.
For a source checkout with the default local reranker:

```bash
uv sync --package hindsight-api-slim --extra local-ml --extra embedded-db
```

Copy `aml.env.example` into your deployment configuration and replace credentials
and the embedding endpoint. Start the API using the ordinary `hindsight-api`
entrypoint. Environment variables must reach both the API and worker; a separate
worker uses the same database and model configuration. Do not enable provider
Batch APIs: their deferred processing window is unsuitable for synchronous AML
Add. A released upstream image does not contain this change until released;
build an image from the submitted commit rather than using `latest`.

Keep `HINDSIGHT_API_FAIL_ON_EXTRACTION_ERRORS=true` as in the example so an
extraction failure cannot be acknowledged as a completed Add with missing facts.

The Memory System Key protects the AML routes. The example also uses the same
key in `ApiKeyTenantExtension` so ordinary Bank APIs are authenticated. With a
custom tenant extension, the AML key must resolve to a valid tenant there too;
the adapter passes it to every engine call and never marks calls internal.

| Extension environment variable | Default | Purpose |
| --- | --- | --- |
| `HINDSIGHT_API_HTTP_EXTENSION` | unset | Set to `hindsight_api.extensions.builtin.aml:AmlHttpExtension` |
| `HINDSIGHT_API_HTTP_AML_API_KEY` | required | Dedicated evaluation credential; Token, Bearer and X-Api-Key accepted |
| `HINDSIGHT_API_HTTP_AML_NAMESPACE` | `aml-textual-v1` | Stable deployment/version namespace; keep across restarts and resume |
| `HINDSIGHT_API_HTTP_AML_ADD_TIMEOUT` | `1770` | Maximum seconds waiting for submission and completion, at most 1800 |
| `HINDSIGHT_API_HTTP_AML_POLL_INTERVAL` | `0.5` | Seconds between operation-status reads |
| `HINDSIGHT_API_HTTP_AML_SEARCH_BUDGET` | `high` | Native recall effort, independent of result count |
| `HINDSIGHT_API_HTTP_AML_SEARCH_MAX_TOKENS` | `8192` | Native recall fact-token budget |

These settings use the existing HTTP-extension configuration loader. No schema
migration, external ID mapping table, or changes to the default API are required.
The extension is opt-in and occupies the HTTP extension slot; a deployment with
another HTTP extension must compose their routers explicitly.

## Contract and mapping

Textual requests only. Image content arrays are rejected rather than discarded.
Ordered user/assistant messages are retained together as a JSON transcript;
provided Unix-millisecond timestamps are rendered as UTC ISO timestamps. Missing
timestamps stay unknown instead of becoming the date of the evaluation run.
`session_id` is preserved as provenance and never filters Search.

```json
{
  "request_id": "run:sample:chunk-1",
  "user_id": "run:sample",
  "session_id": "session-1",
  "messages": [{"role": "user", "content": "Alice lives in Berlin", "timestamp": 1704067200000}]
}
```

Add returns HTTP 200 with exactly the same three identifiers and
`"success": true` after its durable async retain operation completes. It does not
return 202 or acknowledge a merely queued operation. Derived observation
consolidation remains ordinary background work; new raw facts are searchable at
acknowledgement, and Search considers raw facts and available observations.

The Bank ID is `aml-` plus SHA-256 of the UTF-8 representation of the compact
JSON array `[namespace, user_id]` (`ensure_ascii=True`). The exact user ID is
preserved in the identity input; no parsing, trimming or delimiter concatenation.
The operation UUID is UUIDv5 with namespace
`ec2aa6b6-467c-4b95-9637-9130e7b63d10` and the default JSON encoding of
`[namespace, user_id, request_id]` (`ensure_ascii=True`). Each Add has its own
`aml-<operation UUID>` document, so later chunks do not replace earlier ones.

The engine's optional `request_fingerprint` binds the operation identity to the
validated Add body. Identical retries (including concurrent retries and retries
after process restart) join the original operation. A different session, message,
role or timestamp under the same identity returns 409. The existing operation
primary key arbitrates first-submission races, and the fingerprint is in the
existing parent metadata. No in-memory replay cache is involved.

Keep `HINDSIGHT_API_OPERATION_RETENTION_DAYS=0` (the existing default) while a run
can be resumed. Pruning or manually deleting an operation removes that replay
identity; after evaluation, delete the evaluation Banks and associated data
according to AML's retention rules.

Search accepts `query`, `user_id`, `top_k` (integer 1–100) and optional `options`.
Options are accepted but not added to the query in this baseline. It calls native
`recall_async`, preserves native ranking and returns at most `top_k` results:

```json
{"data": [{"id": "fact-id", "content": "memory evidence", "score": 0.87}]}
```

Empty results use `{"data": []}`. `results[].text` becomes `content` and the
native final ranking score becomes `score`. No reflect/answer call is made.
The base SDK/OpenAPI remains unchanged when the extension is off; the enabled
server's OpenAPI includes these extension routes automatically.

## Failure and capacity behavior

- Missing/incorrect key: 401; invalid fields: 422; operation authorization is
  still enforced by Hindsight's tenant and operation-validator extensions.
- Admission uses the same retain/recall lanes as ordinary API requests.
- Failed, cancelled or missing retain jobs: 503. Add timeout: 504; the durable
  task is not cancelled and an identical retry rejoins it.
- Configure load-balancer/proxy timeouts to allow the selected Add deadline;
  avoid a shorter upstream timeout that repeatedly abandons an otherwise valid
  synchronous Add. Provision model quotas, worker throughput and database
  connections for the concurrency declared to AML.
- Freeze commit, model settings, prompts and retrieval settings before Full.
  Never preload held-out questions, answers or other users' memories.

## Validation

```bash
uv run --project hindsight-api-slim pytest tests/test_aml_http_extension.py
uv run --project hindsight-api-slim pytest tests/test_async_retain_operation_id.py
uv run --project hindsight-system-tests pytest tests/test_75_aml_adapter.py
```

The system story uses a real API process, embedded PostgreSQL, worker and
scripted model transports. It checks concurrent replay, immediate search,
multi-chunk preservation, scope isolation and process restart; durable state is
read through the published Hindsight client. Optional extension calls use the
published low-level ApiClient because they are outside the base generated routes.
Real-model quality, quota sufficiency, public reachability and AML's official
Smoke/Full remain deployment checks; deterministic tests are not a leaderboard
score.

## Sources

- [AML API guide](https://agentmemoryleaderboard.ai/api-guide)
- [AML competition and model requirements](https://agentmemoryleaderboard.ai/competition/)
- Original system: [Vectorize Hindsight](https://github.com/vectorize-io/hindsight).
  Disclose the original authors/work and this adapter when submitting the fork.
