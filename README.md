# ZLabel Server

Labeling backend for the ZLabel desktop client. FastAPI + SQLite + ONNXRuntime
(SAM family), **self-hosted**: the server owns its storage tree and its accounts.

**v2**: multi-user (sessions, roles, project membership), task claim/lease,
submit/review workflow, versioned annotations, stateless inference in a separate
process, web administration UI at `/admin`.

## Layout

```console
app/            # the API (only /api/v2 exists: v1 was deleted, see docs/)
  app.py       #   create_app() factory; state lives on app.state
  core/        #   settings, errors, logging
  db/          #   models + alembic migrations
  api/v2/      #   routers (auth, projects, tasks, annotations, images, predict, health, admin)
  services/    #   business logic (auth/project/task/annotation/storage)
  adapters/    #   the edges: local disk storage + inference client + identity provider
  admin/       #   starlette-admin UI (cookie sessions, admin role only)
inference/     # model assets: sam_ort/ (runners), worker.py, ztypes.py, config.py
docs/          # architecture-v2.md (design), plan-selfhosted-storage.md (runbook)
tests/         # v2 API/service tests, inference regression tests
```

## Usage

```console
uv sync
cp .env.example .env.v2        # set ZLSERVER_STORAGE_ROOT (and the model settings)
uv run alembic upgrade head    # create the v2 schema (fresh database)

# create the first admin, then start the API
uv run python -m app.cli user add <name> --role admin
uv run fastapi run app/main.py  # http://127.0.0.1:8000, OpenAPI at /docs

# inference worker (own process, own GPU; needs ZLSERVER_MODEL_* + ZLSERVER_INFERENCE_TOKEN)
uv run fastapi run app/inference_worker/main.py --port 8001
```

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
- The server owns the datasets: `<ZLSERVER_STORAGE_ROOT>/<project>/…` with
  annotations in `<project>/.zlabel/annos/<anno_id>.zlabel` — the same layout the
  desktop uses for a local dataset, so one directory works on both sides.
  `anno_id` is `sha256("<project id>/<project-relative posix path>")`, where the project id is
  the dataset's `.zlabel/project.json` `"id"` (the desktop's `Project.id`) — renaming a
  project or its directory does not invalidate annotations.
- Accounts are local (scrypt hashes in the `users` table); `/admin` manages them
  (Dashboard / Users / Projects / Files / Audit log), with project-scoped files,
  members, labels and tasks on the project detail page. The Dashboard also shows
  live **Server status** / **Inference worker** cards (polled via `/admin/status`).
- The inference worker is a separate process: the API never holds model state
  (this removes v1's "predict on whatever task was loaded last" bug).
- v1 is gone; its last state is the `onnx` branch (`fc04ef4`).
