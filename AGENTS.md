# AGENTS.md

FastAPI labeling backend for the ZLabel desktop client: **self-hosted** (the server
owns `ZLSERVER_STORAGE_ROOT` and its account table), SQLite, ONNXRuntime (SAM
family). The current API lives in `app/`; v1 was deleted and its last state is the
`onnx` branch (`fc04ef4`). The storage/identity layers were rebuilt without
OpenList (P6 done, see `docs/plan-selfhosted-storage.md`); the design record lives
in `docs/architecture-v2.md` (read it before structural changes).

## Commands

- Setup: `uv sync` (Python 3.13 via `.python-version`)
- Admin CLI: `uv run python -m app.cli user add <name> --role reviewer|annotator|admin` ·
  `user ls` · `user passwd <name>` · `user role <name> <role>` · `project ls` ·
  `project members <project>` · `project add-member <project> <user> --role ...` ·
  `storage usage` · `migrate-layout --root <storage-root> [--dry-run]` (moves
  `<project>/zlabel` to `<project>/.zlabel/annos`) ·
  `migrate-anno-ids [--project <name>] [--dry-run]` (re-keys legacy
  `md5("<project name>/<rel>")` annotation files/rows to `sha256("<project key>/<rel>")`) ·
  `sync-instances [--project <name>]` (re-mirror stored documents into the
  instance registry) ·
  `import-annotations --source <dir> --project <name> [--state approved] [--dry-run]
  [--overwrite]` (migrate a legacy folder: copy each ``*.zlabel`` byte-for-byte under
  its new ``sha256(<project key>/<rel>)`` name and create the annotation/history rows
  + label/instance mirrors; run it after scanning the project)
- **Web admin UI** (`/admin`, `ZLSERVER_ADMIN_ENABLED` to turn it off):
  `starlette-admin` mounted in `create_app` with `ZLabelAuthProvider` - a **cookie**
  session for the admin role, backed by the same `AuthService`/sessions as the API
  (the desktop keeps using Bearer tokens).
  Write pages (hand-built HTML, all calls go through the services):
  **Users** (filter/create/role/enable/password reset - audits and revokes the
  target's sessions), **Projects** (list/filter/create; per-project detail tabs:
  overview with rename/metadata (display name, description, active, timeline), files with
  browse/upload/preview/download/delete,
  members, labels, instances, tasks with a state filter) and **Files** (the whole
  storage
  root). Projects also has a **Danger zone** on the Overview tab: deleting requires
  typing the project name exactly and offers "also delete the files on disk"
  (default on); the list rows link there. The **Labels** tab shows each label's
  0-based ordinal as its `id` (the
  class id the COCO/YOLO exports use), sorted by that position; the **⠿ handle in
  the first column** is the only draggable element (inline script) and dragging just
  reorders the DOM and rewrites the form's hidden ``order`` field — one **Save all
  labels** button then posts the whole table to ``/label/save-all``
  (`ProjectService.save_labels`: field edits audited per label, ``sort`` rewritten to
  0..N-1; `Label.id` is untouched). New labels are appended and get an unused palette
  colour (`app/services/label_palette.py`); the colour control is a dropdown of
  swatch + hex entries plus a hex field (``#rrggbb``/``#rgb``/bare ``rrggbb``). **Dashboard** is the landing page: storage usage, progress, the project
  table, a rescan button, and two status cards (**Server status** / **Inference
  worker**) populated by `GET /admin/status` every 30 s from a shared static JS file. The only remaining starlette-admin ``ModelView`` is the
  read-only **Audit log** (its actor is a plain ``actor`` string field - the ``users``
  table has no ModelView, and a relation field without a target view is rejected).
  Tasks/Labels/Members/Storage/Accounts no longer exist as separate pages.
  ``can_create``/``can_edit``/``can_delete`` must be **sync** methods - 1.x calls
  them without awaiting, so an ``async def`` override is always truthy (`app/admin/`).
- Admin REST (global `admin` role): `/api/v2/admin/users` (+ `/{id}`, `/{id}/password`),
  `/api/v2/admin/storage`, `/api/v2/admin/files` (list/upload/download/delete/mkdir/move).
  Project instances: `/api/v2/projects/{p}/instances` (+ `/{number}`, `/{number}/results`,
  `/statuses`) - the project-scoped objects the documents' `instance_id` points at.
  `DELETE /api/v2/projects/{p}?delete_files=true|false` (admin only) removes the
  project: registry rows + files by default, directory only when `delete_files=false`.
  Project labels: `/api/v2/projects/{p}/labels` (+ `/{id}`, `PUT /order` to set the
  order - the 0-based position is the label's ordinal id / export class id).
  Account writes all go through `AuthService.create_user/update_user/set_password`,
  which audit and revoke sessions. Project membership: `/api/v2/projects/{p}/members`. Project
  visibility follows `ZLSERVER_PROJECT_ACCESS_MODE` (`open` default = pre-P4 behaviour,
  `strict` = members only, global admins always see everything).
- Migrations: `uv run alembic upgrade head` · `uv run alembic revision --autogenerate -m "msg"`
  (URL comes from `ZLSERVER_DATABASE_URL`, `-x db_url=...` overrides; never put it in `alembic.ini`)
- Dev server: `uv run fastapi run app/main.py` (entrypoint `app/main.py`; OpenAPI at `/docs`).
  On Windows run `chcp 65001` or set `PYTHONIOENCODING=utf-8` — FastAPI's rich banner
  crashes on a GBK console with `UnicodeEncodeError`.
- Cross-repo contract test: `tests/app/test_client_contract.py` loads the desktop's
  `zlabel/utils/api_helper.py` (standalone, without PySide6) and routes its
  `requests` calls into the in-process app — every URL/params/body the desktop
  produces is checked against the server. It skips unless the desktop checkout is
  present next door (``zlabel_server/`` normally lives inside it).
- Tests: `uv run pytest` (pyproject defaults to `-m 'not slow and not gpu'`, so an
  ordinary machine stays green and fast) ·
  `uv run pytest -m slow` (real ONNX models, minutes) ·
  `uv run pytest -m gpu` (CPU/CUDA parity; needs a working CUDA/cuBLAS stack) ·
  an explicit `-m` replaces the default, so `-m 'not gpu'` would re-include the slow tests
- Lint: `uv run ruff check .` and `uv run ruff format .` (line-length 110)
- Docker: `docker compose up` (API + inference worker; datasets come from the
  host directory mounted at `ZLSERVER_STORAGE_ROOT`)

## Milestones (see `docs/architecture-v2.md` §12)

| | |
|---|---|
| M0 | v1 frozen baseline (`fc04ef4`) — done |
| M1 | v2 skeleton: app factory, config/errors/logging, models + alembic, `/api/v2/health` — done |
| M1b | inference assets → `inference/`, v1 deleted, infra/docs — done |
| M2 | services + v2 endpoints (auth/projects/tasks/annotations/images/labels/progress/predict) — done |
| M3 | inference worker process (`/infer`, embedding snapshots by image sha256, health, metrics) — done |
| M4 | desktop client switches to `/api/v2` (`docs/client-migration-v2.md`) — in progress |
| M5 | acceptance (DoD §14) |
| P0–P6 | self-hosted storage + accounts, OpenList removed (`docs/plan-selfhosted-storage.md`) — done |

## Contracts that must not break

- **anno_id**: `sha256("<project key>/<project-relative posix path>")` (`app/contracts/ids.py`),
  identical to the desktop client's `zlabel.utils.project.anno_id_for`. The **project key**
  is the dataset's `.zlabel/project.json` `"id"` (the desktop's `Project.id`), stored in
  `projects.key`; `ProjectService.ensure_project_key` adopts it, restores the DB one when
  the file is missing and mints a new one otherwise, so renaming the directory or the
  display name never invalidates annotations. `legacy_anno_id_for` keeps the old
  `md5("<project name>/<rel>")` formula for `app.cli migrate-anno-ids` only.
- **Annotation files live in the storage tree** at
  `{ZLSERVER_STORAGE_ROOT}/{project}/{ZLSERVER_ANNO_DIR}/<anno_id>.zlabel`
  (`anno_dir` defaults to `.zlabel/annos`, history under `_history/<anno_id>/v<n>.zlabel`);
  the DB keeps metadata/versions only.
- **Error model**: `{code, message, detail}` with machine-readable codes
  (`unauthorized`/`session_stale`/`forbidden`/`not_found`/`conflict`/`lease_conflict`/
  `validation_error`/`upstream_error`/`inference_unavailable`). A missing annotation
  is `404 not_found` = "not annotated yet" (safe for the client to create).
- **Capabilities** in `GET /api/v2/health` (`app/api/v2/health.py: CAPABILITIES`):
  the desktop gates claim/review/version UI on them — add a capability whenever a
  new client-visible feature appears.
- **Instance numbers are document references**: `Result.instance_id` /
  `Annotation.instances[number]` name an `instances` row (unique per project), so
  the number is **never renumbered** by the server. `InstanceService.sync_document`
  mirrors every annotation save into `instances`/`instance_results` (instances are
  created on first sight: status from the document, colour from the palette; the
  admin's edits win, an empty status is filled from later documents), and
  `app.cli sync-instances` backfills documents saved before the tables existed.
  Deleting an instance removes only the registry row + links; a later save that
  still uses the number recreates it.
- **Auth**: the client holds an opaque server session token (`Authorization:
  Bearer`); the DB stores only its sha256. Passwords are scrypt hashes in `users`
  (`LocalIdentity`), never plaintext. Role/enable/password changes revoke the
  account's sessions (`AuthService.update_user`/`set_password`, which the CLI's
  `user passwd`/`user role` also go through), and the desktop re-logs in on 401:
  `resolve` answers `401 session_stale` for a known-but-dead session (revoked /
  expired / disabled account) and plain `401 unauthorized` for a missing or
  unknown token - keep that split, the desktop renews on either one and
  `tests/app/test_client_contract.py` drives the real client through it.

## Architecture

- `app/app.py` — `create_app(settings, database)`; **all state on `app.state`**
  (`settings`, `db`). Tests build an isolated app with an in-memory DB instead of
  monkeypatching module globals (the v1 pattern that v2 deliberately drops).
- `app/core/` — `config.py` (`ZLSERVER_*` settings, `get_settings()` cached),
  `errors.py` (ApiError hierarchy + handlers), `logging.py` (request-id aware).
- `app/db/` — `base.py` (`Database.session_scope`, `get_session` dependency),
  `models.py` (users, sessions, projects, project_members, labels, tasks, instances,
  instance_results, annotations, annotation_versions, audit_log, link tables),
  `migrations/` (alembic).
  State machine: `draft → submitted → approved|rejected`; claim trio
  `claimed_by/claimed_at/lease_expires_at`.
- `app/api/v2/` — routers only (thin): `auth`, `projects`, `labels`, `instances`,
  `tasks`, `annotations`, `images`, `predict`, `health`; shared dependencies live in
  `app/api/deps.py` (`get_services`, `get_auth`, `require_roles`).
- `app/services/` — business logic: `auth_service` (sessions/roles),
  `project_service` (discovery/sync, labels, progress), `instance_service`
  (project-scoped instances: mirrored from documents + CRUD), `task_service`
  (listing, claim+lease, submit/review), `annotation_service` (versioned save,
  history), `image_store` (content-addressed uploads), `label_palette`
  (auto-assigned label colours), `status_service` (DB/storage/worker probes +
  the dashboard status JSON), `grouping`, `audit`, `container` (the `Services`
  dataclass built by `create_app`).
- `app/adapters/` — the swappable edges:
  * `storage.py`: the `StorageBackend` protocol (paths + IO methods, no credential
    arguments) and `build_storage()`; `local_disk.py` is the only implementation
    (atomic writes, traversal-proof, every top-level directory is a project).
  * `identity.py`: the `IdentityProvider` protocol + `LocalIdentity` (scrypt hashes
    in `users`, stdlib only); `ZLSERVER_BOOTSTRAP_ADMIN/PASSWORD` creates the first
    admin once (never resets an existing password).
  * `inference.py`: `InferenceClient` → the worker's `/infer`.
- `app/admin/` — the starlette-admin UI: `auth.py` (`ZLabelAuthProvider` cookie
  sessions + `admin_context`), `views.py` (write pages + model views with the
  service-layer hooks), `__init__.py` (`build_admin`/`mount_admin`).
- `app/inference_worker/` — the model's own process: `main.py`
  (`create_worker_app`: `/infer` + `/health` + `/metrics`), `engine.py`
  (`InferenceEngine`: admission queue, embedding snapshots, crop handling,
  metrics), `schemas.py` (job/response models). Run it with
  `uv run fastapi run app/inference_worker/main.py --port 8001`.
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
- **Project scoping is per router, not global**: every router under
  `/projects/{project}/...` (and the anno-id-addressed `/tasks/{anno_id}`) must resolve the
  project through `get_project(..., auth=...)` or `require_access`; a bare
  `get_project(project)` skips membership and only shows up in `strict` mode
  (labels / images / predict were fixed that way - `tests/app/test_members.py`
  probes the whole surface, keep it green when adding an endpoint).
- **A missing file is a 404, not a 500**: `LocalDiskBackend` raises `NotFound` for
  absent paths and `UpstreamError` only for real IO failures; never widen that.
- **SQLite foreign keys are ON in the app**: `Database` installs
  `PRAGMA foreign_keys=ON` on every connection, so the models'
  `ondelete="CASCADE"/"SET NULL"` actually apply (a delete no longer leaves orphan
  link rows). Alembic builds its own engine and keeps the default (off) on purpose:
  batch migrations recreate tables and enforced constraints would get in the way.
  A database that predates the pragma may still hold orphans — `PRAGMA
  foreign_key_check` lists them; `ProjectService.delete_project` still removes its
  children explicitly (works on any backend, and the file removal stays inside the
  same transaction: a permission error rolls everything back).
- `tests/conftest.py` is shared (image fixtures + `FakePredictor`); v2 API tests use
  `tests/app/conftest.py` (in-memory DB, real temp storage, `auth_headers` helper) and
  `tests/app/fakes.py` (`LocalBackendHarness`, `FakeInference`). The fixture that used
  to be called `ol` is `harness`; it seeds the real storage root.
- **Timeline is per project**: `projects.timeline` (default on) decides whether the
  scanner parses `group_name`/`day` from the task path (`species/dish/D{n}.png`).
  With it off the tasks keep `""/0`, and toggling it in the admin UI (Projects ▸
  Overview, or `PATCH /projects/{p}` with `timeline`) recomputes every existing task
  in the same transaction. The desktop hides nothing yet — it just sees empty groups.
- **Scanning is manual only**: the app factory starts no scan thread and the config has no
  `scan_on_startup`/`project_scan_interval` any more. `tests/app/test_startup.py` asserts that a fresh app
  leaves the task table empty (the dataset on disk stays invisible) and that `POST /projects/scan` is the
  trigger - use a **file-backed** DB there (the in-memory StaticPool connection cannot be shared with a
  scan thread).
- **Serving cache = image state snapshots.** `runner.export_image_state()/import_image_state()`
  (and the `Predictor` wrappers) move the encoded image around; the worker caches
  them per `sha256(+crop)` and restores instead of re-encoding. If you add state to
  a runner (a new cached tensor), add it to that runner's `_state_fields` or the
  restored task image silently produces wrong masks.
- `mode` quirks (inherited from the desktop): 1 SAM, 2 CV, 3 SAM|CV, 0 = "SAM & CV"
  (the value the client sends when both toggles are on). Only the rect path
  implements 0/3; the worker validates up front so clients get 422, not a 500.
- Listing parameters the desktop relies on: `state` accepts a comma separated list
  (`draft,rejected`), `limit=0` means "every matching task" (the File dock's Fetch ▸
  "all" sends it; explicit values are capped at `MAX_LIMIT = 10000`, negatives/oversized
  are a 422), `offset` still pages an unlimited request, `order` is
  `sequence|id|recent|random`, and
  `POST /projects/{p}/scan` works for a project that does not exist yet (finding new
  directories under the storage root is the point of a scan).
- Prediction parameters the desktop relies on: ``POST /projects/{p}/predict`` resolves the
  task image from, in order, the uploaded file, ``rel_path`` (a project-relative path in
  the storage tree) or ``image_sha256`` (the content-addressed upload cache behind
  ``PUT /projects/{p}/images/{rel_path}``); an unknown digest answers 404 on purpose,
  which is how the client knows to upload the bytes again. The desktop uploads a local
  task image once and then predicts by digest, so keep that route working
  (`tests/app/test_client_contract.py::test_predict_by_uploaded_digest_costs_no_image_bytes`).
- **Renaming a dataset directory by hand re-keys it**: a scan adopts the key from
  `.zlabel/project.json`; if the same key already belongs to another project (a copied
  dataset, or the row of the old directory name), the scanner mints a new key and
  writes it back. Run `uv run python -m app.cli migrate-anno-ids` afterwards to rename
  the annotation files of that project.
- Claim state: `draft → submitted → approved|rejected` (`reopen` pulls back to draft).
  A claim carries `lease_expires_at`; an expired lease is claimable by anyone, a live
  one answers 409 `lease_conflict` with the holder + expiry. Saves renew the lease and
  a save without `base_version` is only accepted while the task has no stored version.
- `TaskRow` resolves the holder/reviewer names with one extra query: reading them via
  `task.claimer` goes stale the moment `claimed_by` is mutated in the same session.
- `.env*` is gitignored except `.env.example`; the real config is `.env.v2`.

## Decisions (already agreed with the user — do not relitigate)

1. **Self-hosted**: the server owns the storage tree and the accounts (scrypt in
   `users`); the client holds an opaque session token. OpenList support was removed
   in P0–P6 — do not reintroduce a storage/identity abstraction for it.
2. **Only `/api/v2`**; v1 is deleted; server and desktop ship together (no compat
   layer, no gradual rollout) — hence `GET /api/v2/health` version/capability checks.
3. **Inference runs in a separate process**; embeddings cached by image sha256
   (this fixes v1's "predict ran on whichever task was loaded last" bug).
4. **Claim + lease, then submit/review** (`force` and role checks for reviewers).
5. **Fresh database, no v1 data migration** (old projects are not preserved).
