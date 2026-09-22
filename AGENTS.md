# AGENTS.md

FastAPI labeling backend for the ZLabel desktop client: OpenList (file storage +
identity source), SQLite, ONNXRuntime (SAM family). **v2 rewrite in progress** —
v1 (`app/`) was deleted; its last state is the `onnx` branch (`fc04ef4`) and the
design lives in `docs/architecture-v2.md` (read it before structural changes).

## Commands

- Setup: `uv sync` (Python 3.13 via `.python-version`)
- Migrations: `uv run alembic upgrade head` · `uv run alembic revision --autogenerate -m "msg"`
  (URL comes from `ZLV2_DATABASE_URL`, `-x db_url=...` overrides; never put it in `alembic.ini`)
- Dev server: `uv run fastapi run v2/main.py` (entrypoint `v2/main.py`; OpenAPI at `/docs`).
  On Windows run `chcp 65001` or set `PYTHONIOENCODING=utf-8` — FastAPI's rich banner
  crashes on a GBK console with `UnicodeEncodeError`.
- Tests: `uv run pytest` (pyproject defaults to `-m 'not slow'`) ·
  `uv run pytest -m slow` (real ONNX models, minutes) ·
  `uv run pytest -m gpu` (CPU/CUDA parity; needs a working CUDA/cuBLAS stack)
- Lint: `uv run ruff check .` and `uv run ruff format .` (line-length 110)
- Docker: `docker compose up` (server + openlist; the inference worker service is
  commented out until M3)

## Milestones (see `docs/architecture-v2.md` §12)

| | |
|---|---|
| M0 | v1 frozen baseline (`fc04ef4`) — done |
| M1 | v2 skeleton: app factory, config/errors/logging, models + alembic, `/api/v2/health` — done |
| M1b | inference assets → `inference/`, vendored SDK → `v2/vendor/`, v1 deleted, infra/docs — done |
| M2 | services + v2 endpoints (auth/projects/tasks/annotations/images/labels/progress) — **next** |
| M3 | inference worker process + `InferenceClient` (embedding cache by image sha256, health, metrics) |
| M4 | desktop client switches to `/api/v2` (`docs/client-migration-v2.md`) |
| M5 | acceptance (DoD §14) |

## Contracts that must not break

- **anno_id**: `md5("<project>/<project-relative posix path>")` (`v2/contracts/ids.py`),
  identical to the desktop client's `zlabel.utils.project.anno_id_for`. Local mirrors
  and OpenList annotation files stay interchangeable only while this holds.
- **Annotation files stay in OpenList** at `{ZLV2_OPLIST_PROJ_DIR}/{project}/zlabel/<anno_id>.zlabel`;
  the DB keeps metadata/versions only.
- **Error model**: `{code, message, detail}` with machine-readable codes
  (`unauthorized`/`session_stale`/`forbidden`/`not_found`/`conflict`/`lease_conflict`/
  `validation_error`/`upstream_error`/`inference_unavailable`). A missing annotation
  is `404 not_found` = "not annotated yet" (safe for the client to create).
- **Capabilities** in `GET /api/v2/health` (`v2/api/v2/health.py: CAPABILITIES`):
  the desktop gates claim/review/version UI on them — add a capability whenever a
  new client-visible feature appears.
- **Auth**: the client holds a server session token; the user's OpenList token is
  stored inside `sessions.oplist_token` and used for FS calls. Never store passwords.

## Architecture

- `v2/app.py` — `create_app(settings, database)`; **all state on `app.state`**
  (`settings`, `db`). Tests build an isolated app with an in-memory DB instead of
  monkeypatching module globals (the v1 pattern that v2 deliberately drops).
- `v2/core/` — `config.py` (`ZLV2_*` settings, `get_settings()` cached),
  `errors.py` (ApiError hierarchy + handlers), `logging.py` (request-id aware).
- `v2/db/` — `base.py` (`Database.session_scope`, `get_session` dependency),
  `models.py` (users, sessions, projects, labels, tasks, annotations,
  annotation_versions, audit_log, link tables), `migrations/` (alembic).
  State machine: `draft → submitted → approved|rejected`; claim trio
  `claimed_by/claimed_at/lease_expires_at`.
- `v2/api/v2/` — routers only (thin); `v2/services/` — business logic;
  `v2/adapters/` — OpenList + inference clients; `v2/vendor/openlist_api/` —
  vendored third-party SDK (do not edit; wrap it in an adapter).
- `inference/` — `sam_ort/` (Predictor/SamRunner/Sam2Runner/Sam3Runner),
  `worker.py` (`ZSamWorker`: prompt → mask → contour post-processing),
  `ztypes.py` (wire types + `AutoMode`/`ReturnType`), `config.py`
  (`InferenceSettings`, same `ZLV2_` prefix), `logging.py` (`ZLogger`).
  The API process must **not** hold model state (M3 moves it to its own service).

## Gotchas

- **`alembic.ini` must stay ASCII-only**: `configparser` reads it with the console
  locale encoding and a GBK Windows console dies on non-ASCII (e.g. an em dash).
- **ONNX models are gitignored** (`assets/onnx/*.onnx`; only `vocab.json`/`merges.txt`
  are tracked) and `tests/conftest.py` asserts `tests/data/zidane.jpg` exists
  (also gitignored) — provide both or collection/tests fail.
- **SAM3 CUDA quirks (respect the code comments)**: the vision encoder's CUDA arena
  holds ~7GB and never shrinks, so `Sam3Runner` builds the vision session per encode
  and releases it; the text encoder is deliberately pinned to CPU.
- **404 semantics depend on the vendored SDK**: OpenList reports "object not found"
  as HTTP 200 + `{"code": 500, ...}` (sometimes a plain 500); `BaseClient` maps any
  error whose message contains "not found" to `NotFoundError(404)`. Never turn that
  into a blanket 500.
- `tests/conftest.py` is shared (image fixtures + `FakePredictor`); v2 API tests use
  `tests/v2/conftest.py` (in-memory DB, no network).
- `.env*` is gitignored except `.env.example`; the real config is `.env.v2`.

## Decisions (already agreed with the user — do not relitigate)

1. **OpenList is the identity source**: login proxy + server-issued session token;
   local `users` rows hold role/stats; the user's OpenList token is kept in the
   session for per-user file ACLs.
2. **Only `/api/v2`**; v1 is deleted; server and desktop ship together (no compat
   layer, no gradual rollout) — hence `GET /api/v2/health` version/capability checks.
3. **Inference runs in a separate process**; embeddings cached by image sha256
   (this fixes v1's "predict ran on whichever frame was loaded last" bug).
4. **Claim + lease, then submit/review** (`force` and role checks for reviewers).
5. **Fresh database, no v1 data migration** (old projects are not preserved).
