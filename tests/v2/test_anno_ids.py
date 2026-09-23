"""The anno_id contract: ``sha256("<project key>/<rel>")`` and its migration.

The project key is the dataset's ``.zlabel/project.json`` ``"id"`` (the desktop's
``Project.id``), so renaming a project or its directory never invalidates the
annotation files.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from v2.contracts.ids import anno_id_for, legacy_anno_id_for
from v2.db.models import Annotation, AnnotationVersion, Project, Task


def _project_json(settings, name: str = "projA") -> Path:
    return Path(settings.storage_root) / name / ".zlabel" / "project.json"


def _seed_image(settings, name: str = "projA", rel: str = "images/D1.png") -> Path:
    image = Path(settings.storage_root) / name / rel
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png-bytes")
    return image


def test_create_project_persists_a_key(services, settings):
    project = services.projects.create_project("projA", "Project A")

    assert project.key
    payload = json.loads(_project_json(settings).read_text(encoding="utf-8"))
    assert payload["id"] == project.key
    assert payload["name"] == "projA"


def test_scan_adopts_the_dataset_id(services, settings):
    _seed_image(settings)
    _project_json(settings).parent.mkdir(parents=True, exist_ok=True)
    _project_json(settings).write_text(json.dumps({"id": "dataset-key", "name": "projA"}), encoding="utf-8")

    services.projects.scan_and_sync(force=True)

    project = services.projects.get_project("projA")
    assert project.key == "dataset-key"
    task = services.tasks.get_task(anno_id_for("dataset-key", "images/D1.png"))
    assert task.task.rel_path == "images/D1.png"


def test_scan_mints_a_key_and_keeps_the_other_fields(services, settings):
    _seed_image(settings, "projB")
    _project_json(settings, "projB").parent.mkdir(parents=True, exist_ok=True)
    _project_json(settings, "projB").write_text(
        json.dumps({"name": "projB", "description": "kept", "labels": []}), encoding="utf-8"
    )

    services.projects.scan_and_sync(force=True)

    project = services.projects.get_project("projB")
    payload = json.loads(_project_json(settings, "projB").read_text(encoding="utf-8"))
    assert project.key and payload["id"] == project.key
    assert payload["description"] == "kept" and payload["labels"] == []
    assert services.tasks.get_task(anno_id_for(project.key, "images/D1.png")).task.rel_path == "images/D1.png"


def test_a_copied_dataset_gets_its_own_key(services, settings, db):
    for name in ("projA", "projB"):
        _seed_image(settings, name)
        _project_json(settings, name).parent.mkdir(parents=True, exist_ok=True)
        _project_json(settings, name).write_text(json.dumps({"id": "shared"}), encoding="utf-8")

    services.projects.scan_and_sync(force=True)

    with db.session_scope() as session:
        keys = {p.name: p.key for p in session.scalars(select(Project)).all()}
    assert keys["projA"] != keys["projB"]
    for name in ("projA", "projB"):
        payload = json.loads(_project_json(settings, name).read_text(encoding="utf-8"))
        assert payload["id"] == keys[name]


def test_migrate_anno_ids_rewrites_files_and_rows(services, settings, db):
    """A project annotated with the legacy md5 ids is re-keyed in place."""
    project = services.projects.create_project("projA", actor_id=1)
    _seed_image(settings)
    services.projects.scan_and_sync(force=True)
    storage = services.storage
    old_id = legacy_anno_id_for("projA", "images/D1.png")
    new_id = anno_id_for(project.key, "images/D1.png")
    payload = json.dumps(
        {"id": old_id, "image_path": "images/D1.png", "results": {}}
    ).encode("utf-8")

    # simulate the pre-key state: files and rows carry the legacy id
    storage.put_bytes(f"{storage.zlabel_dir('projA')}/{old_id}.zlabel", payload)
    storage.put_bytes(storage.history_path("projA", old_id, 1), payload)
    with db.session_scope() as session:
        task = session.scalar(select(Task))
        task.anno_id = old_id
        session.add(Annotation(task_id=task.id, anno_id=old_id, version=1, path=""))
        session.add(AnnotationVersion(task_id=task.id, version=1, path=""))
    # a stray file with no task row still names its task
    stray_old = legacy_anno_id_for("projA", "images/stray.png")
    storage.put_bytes(
        f"{storage.zlabel_dir('projA')}/{stray_old}.zlabel",
        json.dumps({"image_path": "images/stray.png", "results": {}}).encode("utf-8"),
    )

    stats = services.projects.migrate_anno_ids("projA")

    assert stats["tasks"] == 1 and stats["renamed"] == 3  # current + history + stray
    assert not storage.exists(f"{storage.zlabel_dir('projA')}/{old_id}.zlabel")
    assert storage.exists(f"{storage.zlabel_dir('projA')}/{new_id}.zlabel")
    migrated = json.loads(storage.get_bytes(f"{storage.zlabel_dir('projA')}/{new_id}.zlabel"))
    # the embedded id follows the file name; everything else is kept
    assert migrated == {"id": new_id, "image_path": "images/D1.png", "results": {}}
    assert json.loads(storage.get_bytes(storage.history_path("projA", new_id, 1)))["id"] == new_id
    assert storage.exists(storage.history_path("projA", new_id, 1))
    assert not storage.exists(storage.history_path("projA", old_id, 1))
    assert storage.exists(
        f"{storage.zlabel_dir('projA')}/{anno_id_for(project.key, 'images/stray.png')}.zlabel"
    )
    with db.session_scope() as session:
        assert session.scalar(select(Task)).anno_id == new_id
        annotation = session.scalar(select(Annotation))
        assert annotation.anno_id == new_id and annotation.path.endswith(f"{new_id}.zlabel")
        assert session.scalar(select(AnnotationVersion)).path.endswith(f"{new_id}/v1.zlabel")


def test_migrate_anno_ids_dry_run_changes_nothing(services, settings, db):
    project = services.projects.create_project("projA", actor_id=1)
    _seed_image(settings)
    services.projects.scan_and_sync(force=True)
    storage = services.storage
    old_id = legacy_anno_id_for("projA", "images/D1.png")
    payload = b"{}"
    storage.put_bytes(f"{storage.zlabel_dir('projA')}/{old_id}.zlabel", payload)
    with db.session_scope() as session:
        session.scalar(select(Task)).anno_id = old_id

    stats = services.projects.migrate_anno_ids("projA", dry_run=True)

    assert stats["renamed"] == 1
    assert storage.exists(f"{storage.zlabel_dir('projA')}/{old_id}.zlabel")
    assert not storage.exists(
        f"{storage.zlabel_dir('projA')}/{anno_id_for(project.key, 'images/D1.png')}.zlabel"
    )
    with db.session_scope() as session:
        assert session.scalar(select(Task)).anno_id == old_id
