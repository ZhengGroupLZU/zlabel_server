# AGENTS.md

FastAPI labeling backend for ZLabel. Runs SAM-family (EdgeSAM/SlimSAM/SAM/SAM2/SAM3) segmentation via ONNXRuntime, proxies an external OpenList file/dataset service, stores task state in SQLite.

## Commands

- Setup: `uv sync` (uv, Python 3.13 per `.python-version`)
- Dev server: `uv run fastapi run app/app.py` — the entrypoint is `app/app.py`. The `main.py` in README.md does not exist; README is stale.
- Tests: `uv run pytest` (pyproject `addopts = "-m 'not slow'"` runs fast suite by default)
  - `uv run pytest -m slow` — real-model smoke tests (takes minutes)
  - `uv run pytest -m gpu` — CPU/CUDA parity tests (auto-skipped without CUDAExecutionProvider)
- Lint: `uv run ruff check .` (line-length 110, configured in pyproject)

## Gotchas

- **Missing objects answer 404**: OpenList reports "object not found" as HTTP 200 with `{"code": 500, ...}` (occasionally a plain HTTP 500); `BaseClient` maps any error whose message contains "not found" to `NotFoundError(404)`, so `get_zlabel`/`get_image`/`fs.get_file_bytes` return 404 and the desktop can distinguish "not annotated yet" from a real failure. Never reintroduce a blanket 500 for those.
- **Canonical ids**: `db.anno_id_for(project, filename, base)` = `md5("<project>/<project-relative POSIX path>")`, the same formula as the ZLabel desktop's `anno_id_for()`. `Path.relative_to(...).as_posix()` is deliberate: Windows would otherwise emit backslashes and the two sides would disagree. Task rows keep the absolute OpenList path in `filename`.
- **Uploads (B1/B5)**: `/api/v1/save_zlabel` requires a valid token (`require_user`), parses the JSON first (invalid -> 422), refuses to overwrite a newer stored annotation (409 + `{updated_at, updated_by}`, bypass with `force=true`), and only writes the DB/link tables **after** a successful `stream_upload` (failure -> 502, task stays unfinished). `/api/v1/set_image` and `/api/v1/predict` are token-gated too (verification cached for 60 s).
- **Labels / progress / soft delete (B4/B6)**: a successful upload links the annotation's label names (`link_task_label`, unknown names create `Label` rows, users are lower-cased, repeats never duplicate links). `GET /api/v1/labels?project=` lists a project's labels and `GET /api/v1/how-many-finished?project=` returns `{finished, total}` (it used to be a bare int). `Task.missing` marks files that vanished from OpenList: they stay in the DB for history but are hidden from `get_tasks`/`get_labels`/`get_progress`. `_set_model_image` now awaits the embedding, so a predict right after a set-image cannot race it.
- **Windows startup**: the SQLite file is created under `ZLSERVER_DATABASE_URL` (see `.env.onnx`); `app/db.py` now creates the parent directory automatically, so a fresh checkout no longer dies with `unable to open database file`. The FastAPI CLI prints emoji through rich, which crashes on a GBK console (`UnicodeEncodeError`): run `chcp 65001` (or set `PYTHONIOENCODING=utf-8`) before `uv run fastapi run app/app.py`.
- **Startup costs**: the first background project scan runs at startup and `get_tasks` awaits a throttled scan (2 s), so with a slow/remote OpenList the first client requests can take a few seconds while the scan holds the SQLite write lock.
- **ONNX models are gitignored.** Only `assets/onnx/{vocab.json,merges.txt}` are tracked. The `.onnx` files must be placed in `assets/onnx/` using the exact names in `app/sam_ort/predictor.py:20` (`MODEL_FILES`) and `runner.py`. Loose `.onnx`/`.pt` files in the `assets/` root are stale leftovers, not the models tests use. Without them, the server can't load and most model tests fail.
- **Fast tests need a test image**: `tests/conftest.py` asserts `tests/data/zidane.jpg` (720x1280) exists — without it a fresh checkout fails collection. The `.jpg` is not tracked (`.gitignore` has `*.jpg`), so it must be provided.
- **`tests/test_predictor_policy.py` was stale (now fixed)**: it used to import `CUDA_FAST_MODELS` / `_effective_backend` from `app.sam_ort.predictor`, which no longer exist. The per-model CUDA policy was removed; `backends.build_providers` now uses CUDA for any model when `ZLSERVER_ORT_BACKEND=CUDA`.
- **`app/app.py` imports create state at module load**: `SETTINGS`, `oplist_client`, `SAM_MODEL`, `IMAGE_CACHE` are module-level. Tests monkeypatch `SAM_MODEL` and `oplist_client.fs` to stay hermetic; keep API tests that way.
- Docker CMD runs `app/app.py`; `docker-compose.yml` pairs the server with an `openlist` container at `172.20.0.3:5244`.

## Config (`app/config.py`)

pydantic-settings; all env vars prefixed `ZLSERVER_`, optional `.env.onnx` file. Key vars: `ZLSERVER_MODEL_NAME`, `ZLSERVER_MODEL_DIR` (default `assets/onnx`), `ZLSERVER_ORT_BACKEND` (CPU/CUDA; CUDA falls back to CPU), `ZLSERVER_OPLIST_HOST`, `ZLSERVER_OPLIST_PROJ_DIR`, `ZLSERVER_IMAGE_CACHE_SIZE`, `ZLSERVER_MIN_CONTOUR_AREA_RATIO`, `ZLSERVER_PROJECT_MARKER` (default `.zlabel-server-project-root`), `ZLSERVER_PROJECT_SCAN_INTERVAL` (seconds, default 300).

## Architecture

- `app/app.py` — all FastAPI routes. Most endpoints proxy the OpenList client (`app/openlist_api/`); token goes in the `Authorization` header. Module-level `IMAGE_CACHE` (OrderedDict) caches image bytes by name. Note: `_set_model_image` fires `SAM_MODEL.set_image` into a threadpool without awaiting, so a predict that follows a set-image races the embedding. `POST /api/v1/save_zlabel` (form field `project`) and `GET /api/v1/get_zlabel` (query param `project`) route zlabel read/writes to `{oplist_proj_dir}/{project}/zlabel`; when `project` is omitted they fall back to the configured `ZLSERVER_OPLIST_PROJ_NAME`.
- `app/sam_ort/` — ONNXRuntime layer. `Predictor` facade (`predictor.py`, thread-safe via a lock; sessions created lazily by `setup_model()`) → `SamRunner`/`Sam2Runner`/`Sam3Runner` (`runner.py`). Preprocessing mirrors ZLabel: SAM/EdgeSAM stretch to 1024², SlimSAM/SAM2 letterbox, SAM3 stretches for PCS and letterboxes for PVS (dual-cached vision features); `Predictor` routes text→PCS (`segment_text`), points/boxes→interactive paths. `backends.py` `OrtSession` retries session creation: drops flaky `SimplifiedLayerNormFusion` optimizer → disables graph optimizations → CPU fallback.
- `app/project_scan.py` — OpenList project auto-discovery. A **top-level** dir under `ZLSERVER_OPLIST_PROJ_DIR` is a project only if it contains the hidden marker file `ZLSERVER_PROJECT_MARKER` (probed via `fs.get`, not `ls`, because OpenList may hide dotfiles). Images are collected recursively inside the project. `ScanResult` records present dirs + confirmed-missing dirs so the DB only deactivates dirs we are *sure* no longer qualify; uninspectable dirs are left untouched. Scan is triggered on app startup, periodically (`ZLSERVER_PROJECT_SCAN_INTERVAL`), and on `GET /api/v1/get_projects`; `db.sync_projects_from_scan` upserts projects as active and deactivates (not deletes) removed/marker-less ones.
- `app/worker.py` — `ZSamWorker`: prompt → SAM/CV segmentation → contour post-processing → Rect/Polygon/RLE. `AutoMode` is a flag enum (`SAM | CV` combos). `app/ztypes.py` holds pydantic result types.
- `app/db.py` — SQLite via SQLAlchemy; tables created via `Base.metadata.create_all(engine)` on import. Task `anno_id` = md5 of `{oplist_proj_dir}/{project}/{filename}`.
- `assets/mnn/` is dead — the MNN backend was replaced by ONNXRuntime.

## CUDA/ORT memory quirks (respect the code comments)

- SAM3 vision encoder's CUDA arena holds ~7GB and never shrinks; `Sam3Runner` creates the vision session per-encode and releases it (`runner.py:219`) to avoid OOM on 12GB GPUs.
- SAM3 text encoder (1.4GB fp32, trivial compute) is deliberately pinned to CPU (`runner.py:205`).
