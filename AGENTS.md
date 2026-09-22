# AGENTS.md

FastAPI labeling backend for the ZLabel desktop client: OpenList (file storage +
identity source), SQLite, ONNXRuntime (SAM family). **v2 rewrite in progress** —
v1 (`app/`) was deleted; its last state is the `onnx` branch (`fc04ef4`) and the
design lives in `docs/architecture-v2.md` (read it before structural changes).

## Commands

- Setup: `uv sync` (Python 3.13 via `.python-version`)
- Migrations: `uv run alembic upgrade head` · `uv run alembic revision --autogenerate -m "msg"`
  (URL comes from `ZLSERVER_DATABASE_URL`, `-x db_url=...` overrides; never put it in `alembic.ini`)
- Dev server: `uv run fastapi run v2/main.py` (entrypoint `v2/main.py`; OpenAPI at `/docs`).
  On Windows run `chcp 65001` or set `PYTHONIOENCODING=utf-8` — FastAPI's rich banner
  crashes on a GBK console with `UnicodeEncodeError`.
- Cross-repo contract test: `tests/v2/test_client_contract.py` loads the desktop's
  `zlabel/utils/api_helper.py` (standalone, without PySide6) and routes its
  `requests` calls into the in-process app — every URL/params/body the desktop
  produces is checked against the server. It skips unless the desktop checkout is
  present next door (``zlabel_server/`` normally lives inside it).
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
| M2 | services + v2 endpoints (auth/projects/tasks/annotations/images/labels/progress/predict) — done |
| M3 | inference worker process (`/infer`, embedding snapshots by image sha256, health, metrics) — done |
| M4 | desktop client switches to `/api/v2` (`docs/client-migration-v2.md`) — **next** |
| M5 | acceptance (DoD §14) |

## Contracts that must not break

- **anno_id**: `md5("<project>/<project-relative posix path>")` (`v2/contracts/ids.py`),
  identical to the desktop client's `zlabel.utils.project.anno_id_for`. Local mirrors
  and OpenList annotation files stay interchangeable only while this holds.
- **Annotation files stay in OpenList** at `{ZLSERVER_OPLIST_PROJ_DIR}/{project}/zlabel/<anno_id>.zlabel`;
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
- **Whose OpenList token?** Split by direction, on purpose:
  * **reads** (frames, annotation documents, history) use the **session user's**
    token, so a user only sees what their own OpenList ACLs allow;
  * **writes** (annotation + history files, project directories/marker) use the
    **service account** (`ZLSERVER_OPLIST_TOKEN`, else `ZLSERVER_OPLIST_USERNAME` /
    `PASSWORD`), because annotator accounts are read-only by design. The service
    account is also the background scanner's identity.
  A wrong token shows up as OpenList's 403 `permission denied` (the message names
  the refused operation); the *attribution* of a save is unaffected — it comes from
  the session (`annotations.author_id`, audit rows).

## Architecture

- `v2/app.py` — `create_app(settings, database)`; **all state on `app.state`**
  (`settings`, `db`). Tests build an isolated app with an in-memory DB instead of
  monkeypatching module globals (the v1 pattern that v2 deliberately drops).
- `v2/core/` — `config.py` (`ZLSERVER_*` settings, `get_settings()` cached),
  `errors.py` (ApiError hierarchy + handlers), `logging.py` (request-id aware).
- `v2/db/` — `base.py` (`Database.session_scope`, `get_session` dependency),
  `models.py` (users, sessions, projects, labels, tasks, annotations,
  annotation_versions, audit_log, link tables), `migrations/` (alembic).
  State machine: `draft → submitted → approved|rejected`; claim trio
  `claimed_by/claimed_at/lease_expires_at`.
- `v2/api/v2/` — routers only (thin): `auth`, `projects`, `labels`, `tasks`,
  `annotations`, `images`, `predict`, `health`; shared dependencies live in
  `v2/api/deps.py` (`get_services`, `get_auth`, `require_roles`).
- `v2/services/` — business logic: `auth_service` (sessions/roles),
  `project_service` (discovery/sync, labels, progress), `task_service`
  (listing, claim+lease, submit/review), `annotation_service` (versioned save,
  history), `image_store` (content-addressed uploads), `grouping`, `audit`,
  `container` (the `Services` dataclass built by `create_app`).
- `v2/adapters/` — `openlist.py` (the only OpenList boundary, token-explicit: one
  client per call) and `inference.py` (`InferenceClient` → the worker's `/infer`).
- `v2/vendor/openlist_api/` — vendored third-party SDK (do not edit; wrap it).
- `v2/inference_worker/` — the model's own process: `main.py`
  (`create_worker_app`: `/infer` + `/health` + `/metrics`), `engine.py`
  (`InferenceEngine`: admission queue, embedding snapshots, crop handling,
  metrics), `schemas.py` (job/response models). Run it with
  `uv run fastapi run v2/inference_worker/main.py --port 8001`.
- `inference/` — `sam_ort/` (Predictor/SamRunner/Sam2Runner/Sam3Runner),
  `worker.py` (`ZSamWorker`: prompt → mask → contour post-processing),
  `ztypes.py` (wire types + `AutoMode`/`ReturnType`), `config.py`
  (`InferenceSettings`, same `ZLSERVER_` prefix), `logging.py` (`ZLogger`).
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
  `tests/v2/conftest.py` (in-memory DB, fake OpenList, `auth_headers` helper) and
  `tests/v2/fakes.py` (`FakeOpenList`, `FakeInference`).
- The API suite disables scanners (`scan_on_startup=False`, `project_scan_interval=0`)
  so tests stay deterministic; `tests/v2/test_startup.py` covers the scan itself and
  must use a **file-backed** DB — the in-memory StaticPool connection cannot be
  shared with the scan thread.
- **Serving cache = image state snapshots.** `runner.export_image_state()/import_image_state()`
  (and the `Predictor` wrappers) move the encoded image around; the worker caches
  them per `sha256(+crop)` and restores instead of re-encoding. If you add state to
  a runner (a new cached tensor), add it to that runner's `_state_fields` or the
  restored frame silently produces wrong masks.
- `mode` quirks (inherited from the desktop): 1 SAM, 2 CV, 3 SAM|CV, 0 = "SAM & CV"
  (the value the client sends when both toggles are on). Only the rect path
  implements 0/3; the worker validates up front so clients get 422, not a 500.
- Listing parameters the desktop relies on: `state` accepts a comma separated list
  (`draft,rejected`), `order` is `sequence|id|recent|random`, and
  `POST /projects/{p}/scan` works for a project that does not exist yet (finding new
  OpenList directories is the point of a scan).
- Claim state: `draft → submitted → approved|rejected` (`reopen` pulls back to draft).
  A claim carries `lease_expires_at`; an expired lease is claimable by anyone, a live
  one answers 409 `lease_conflict` with the holder + expiry. Saves renew the lease and
  a save without `base_version` is only accepted while the task has no stored version.
- `TaskRow` resolves the holder/reviewer names with one extra query: reading them via
  `task.claimer` goes stale the moment `claimed_by` is mutated in the same session.
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
