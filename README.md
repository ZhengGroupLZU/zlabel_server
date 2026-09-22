# ZLabel Server

Labeling backend for the ZLabel desktop client. FastAPI + SQLite + ONNXRuntime
(SAM family) on top of OpenList (file storage and identity source).

**v2**: multi-user (sessions, roles), task claim/lease, submit/review workflow,
versioned annotations, stateless inference in a separate process.

## Layout

```console
v2/            # the API (only /api/v2 exists: v1 was deleted, see docs/)
  app.py       #   create_app() factory; state lives on app.state
  core/        #   settings, errors, logging
  db/          #   models + alembic migrations
  api/v2/      #   routers (auth, projects, tasks, annotations, images, predict, health)
  services/    #   business logic (auth/project/task/annotation/stats)
  adapters/    #   OpenList + inference clients
  vendor/      #   vendored OpenList SDK
inference/     # model assets: sam_ort/ (runners), worker.py, ztypes.py, config.py
docs/          # architecture-v2.md (design), client-migration-v2.md (desktop checklist)
tests/         # v2 API/service tests, inference regression tests, vendor tests
```

## Usage

```console
uv sync
cp .env.example .env.v2        # adjust OpenList host/credentials, model
uv run alembic upgrade head    # create the v2 schema (fresh database)
uv run fastapi run v2/main.py  # http://127.0.0.1:8000  (OpenAPI at /docs)
```

The v2 database (`ZLV2_DATABASE_URL`, default `./data/zlabel_server_v2.db`) is
independent from the old `zlabel_server.db`: v1 data is **not** migrated.

Quality gates:

```console
uv run pytest                  # fast suite (-m 'not slow' is the default)
uv run pytest -m slow          # real-model smoke tests (minutes)
uv run pytest -m gpu           # CPU/CUDA parity (needs a working CUDA stack)
uv run ruff check . && uv run ruff format .
```

## Notes

- `GET /api/v2/health` reports capabilities; the client uses it to gate claim/
  review UI and to refuse a version mismatch.
- Annotation files stay in OpenList (`<project>/zlabel/<anno_id>.zlabel`), so
  historical annotations remain readable; `anno_id` is
  `md5("<project>/<project-relative posix path>")` on both sides.
- The inference worker is a separate process: the API never holds model state
  (this removes v1's "predict on whatever frame was loaded last" bug).
- v1 is gone; its last state is the `onnx` branch (`fc04ef4`).
