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

- **ONNX models are gitignored.** Only `assets/onnx/{vocab.json,merges.txt}` are tracked. The `.onnx` files must be placed in `assets/onnx/` using the exact names in `app/sam_ort/predictor.py:20` (`MODEL_FILES`) and `runner.py`. Loose `.onnx`/`.pt` files in the `assets/` root are stale leftovers, not the models tests use. Without them, the server can't load and most model tests fail.
- **Fast tests need a test image**: `tests/conftest.py` asserts `tests/data/zidane.jpg` (720x1280) exists — without it a fresh checkout fails collection. The `.jpg` is not tracked (`.gitignore` has `*.jpg`), so it must be provided.
- **`tests/test_predictor_policy.py` was stale (now fixed)**: it used to import `CUDA_FAST_MODELS` / `_effective_backend` from `app.sam_ort.predictor`, which no longer exist. The per-model CUDA policy was removed; `backends.build_providers` now uses CUDA for any model when `ZLSERVER_ORT_BACKEND=CUDA`.
- **`app/app.py` imports create state at module load**: `SETTINGS`, `oplist_client`, `SAM_MODEL`, `IMAGE_CACHE` are module-level. Tests monkeypatch `SAM_MODEL` and `oplist_client.fs` to stay hermetic; keep API tests that way.
- Docker CMD runs `app/app.py`; `docker-compose.yml` pairs the server with an `openlist` container at `172.20.0.3:5244`.

## Config (`app/config.py`)

pydantic-settings; all env vars prefixed `ZLSERVER_`, optional `.env.onnx` file. Key vars: `ZLSERVER_MODEL_NAME`, `ZLSERVER_MODEL_DIR` (default `assets/onnx`), `ZLSERVER_ORT_BACKEND` (CPU/CUDA; CUDA falls back to CPU), `ZLSERVER_OPLIST_HOST`, `ZLSERVER_OPLIST_PROJ_DIR`, `ZLSERVER_IMAGE_CACHE_SIZE`, `ZLSERVER_MIN_CONTOUR_AREA_RATIO`.

## Architecture

- `app/app.py` — all FastAPI routes. Most endpoints proxy the OpenList client (`app/openlist_api/`); token goes in the `Authorization` header. Module-level `IMAGE_CACHE` (OrderedDict) caches image bytes by name. Note: `_set_model_image` fires `SAM_MODEL.set_image` into a threadpool without awaiting, so a predict that follows a set-image races the embedding.
- `app/sam_ort/` — ONNXRuntime layer. `Predictor` facade (`predictor.py`, thread-safe via a lock; sessions created lazily by `setup_model()`) → `SamRunner`/`Sam2Runner`/`Sam3Runner` (`runner.py`). Preprocessing mirrors ZLabel: SAM/EdgeSAM stretch to 1024², SlimSAM/SAM2 letterbox, SAM3 stretches for PCS and letterboxes for PVS (dual-cached vision features); `Predictor` routes text→PCS (`segment_text`), points/boxes→interactive paths. `backends.py` `OrtSession` retries session creation: drops flaky `SimplifiedLayerNormFusion` optimizer → disables graph optimizations → CPU fallback.
- `app/worker.py` — `ZSamWorker`: prompt → SAM/CV segmentation → contour post-processing → Rect/Polygon/RLE. `AutoMode` is a flag enum (`SAM | CV` combos). `app/ztypes.py` holds pydantic result types.
- `app/db.py` — SQLite via SQLAlchemy; tables created via `Base.metadata.create_all(engine)` on import. Task `anno_id` = md5 of `{oplist_proj_dir}/{project}/{filename}`.
- `assets/mnn/` is dead — the MNN backend was replaced by ONNXRuntime.

## CUDA/ORT memory quirks (respect the code comments)

- SAM3 vision encoder's CUDA arena holds ~7GB and never shrinks; `Sam3Runner` creates the vision session per-encode and releases it (`runner.py:219`) to avoid OOM on 12GB GPUs.
- SAM3 text encoder (1.4GB fp32, trivial compute) is deliberately pinned to CPU (`runner.py:205`).
