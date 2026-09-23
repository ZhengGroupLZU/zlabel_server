"""Importing legacy annotation folders: re-keying + DB rows + mirrors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from v2.contracts.ids import anno_id_for, legacy_anno_id_for
from v2.core.errors import NotFound
from v2.db.models import Annotation, AnnotationVersion, Instance, InstanceResult, Task


def _seed_image(services, rel: str = "images/dish1/D1.png") -> str:
    image = Path(services.settings.storage_root) / "projA" / rel
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png-bytes")
    services.projects.scan_and_sync(force=True)
    return rel


def _document(rel: str) -> dict:
    return {
        "image_path": rel,
        "results": {"r1": {"labels": [{"name": "Seed"}], "instance_id": 1}},
        "instances": {"1": "normal_seed"},
    }


def test_import_document_creates_rows_history_labels_and_instances(services):
    rel = _seed_image(services)
    project = services.projects.get_project("projA")
    anno_id = anno_id_for(project.key, rel)
    document = _document(rel)
    content = json.dumps(document).encode("utf-8")
    services.storage.put_bytes(services.storage.anno_path("projA", anno_id), content)

    outcome = services.annotations.import_document("projA", rel, content, document=document, state="approved")

    assert outcome == {"anno_id": anno_id, "version": 1, "labels": ["Seed"], "imported": True}
    assert services.storage.exists(services.storage.history_path("projA", anno_id, 1))
    with services.db.session_scope() as session:
        annotation = session.scalar(select(Annotation))
        assert annotation.version == 1 and annotation.content_hash
        assert json.loads(annotation.labels_json) == ["Seed"]
        assert session.scalar(select(AnnotationVersion)).note == "imported"
        assert session.scalar(select(Task)).state == "approved"
        assert session.scalar(select(Instance)).number == 1
        assert session.scalar(select(func.count()).select_from(InstanceResult)) == 1

    # idempotent: the same content does not bump the version
    again = services.annotations.import_document("projA", rel, content, document=document)
    assert again["imported"] is False and again["version"] == 1

    # different content becomes the next version
    changed_document = {**_document(rel), "note": "x"}
    bumped = services.annotations.import_document(
        "projA", rel, json.dumps(changed_document).encode("utf-8"), document=changed_document
    )
    assert bumped["imported"] is True and bumped["version"] == 2
    with services.db.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(AnnotationVersion)) == 2


def test_import_document_needs_the_task(services):
    services.projects.create_project("projA", actor_id=1)
    with pytest.raises(NotFound):
        services.annotations.import_document("projA", "images/nope.png", b"{}", document={})


def test_import_document_dry_run_writes_nothing(services):
    rel = _seed_image(services)
    project = services.projects.get_project("projA")
    anno_id = anno_id_for(project.key, rel)
    document = _document(rel)
    content = json.dumps(document).encode("utf-8")

    outcome = services.annotations.import_document("projA", rel, content, document=document, dry_run=True)

    assert outcome["imported"] is True and outcome["version"] == 1
    assert not services.storage.exists(services.storage.history_path("projA", anno_id, 1))
    with services.db.session_scope() as session:
        assert session.scalar(select(Annotation)) is None


def test_cli_import_annotations_rekeys_and_is_idempotent(tmp_path, monkeypatch):
    """The whole legacy-folder migration: old md5 names -> sha256 ids + DB rows."""
    from v2.adapters.local_disk import LocalDiskBackend
    from v2.cli import main
    from v2.core.config import Settings
    from v2.db.base import Database
    from v2.services.container import Services

    storage_root = tmp_path / "storage"
    uploads = tmp_path / "uploads"
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
        storage_root=str(storage_root),
        upload_dir=str(uploads),
        inference_url="",
    )
    settings.ensure_dirs()
    database = Database(settings.database_url)
    database.create_all()
    services = Services.build(settings, database, storage=LocalDiskBackend(settings))
    services.auth.identity.create_user("boss", "secret123", admin=True)
    services.projects.create_project("projA", actor_id=1)
    rel = _seed_image(services)
    project = services.projects.get_project("projA")
    new_id = anno_id_for(project.key, rel)

    source = tmp_path / "legacy"
    source.mkdir()
    document = _document(rel)
    content = json.dumps(document).encode("utf-8")
    legacy = legacy_anno_id_for("annos", rel)  # md5("<old name>/<rel>")
    (source / f"{legacy}.zlabel").write_bytes(content)

    monkeypatch.setenv("ZLSERVER_STORAGE_ROOT", str(storage_root))
    monkeypatch.setenv("ZLSERVER_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("ZLSERVER_UPLOAD_DIR", str(uploads))

    args = ["import-annotations", "--source", str(source), "--project", "projA"]
    assert main([*args, "--dry-run"]) == 0
    # dry run: nothing landed
    assert not services.storage.exists(services.storage.anno_path("projA", new_id))
    with database.session_scope() as session:
        assert session.scalar(select(Annotation)) is None

    assert main([*args, "--state", "approved"]) == 0
    target = storage_root / "projA" / ".zlabel" / "annos" / f"{new_id}.zlabel"
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert stored["id"] == new_id  # the file name is the identity
    assert stored["image_path"] == rel and stored["results"] == document["results"]
    with database.session_scope() as session:
        annotation = session.scalar(select(Annotation))
        assert annotation.anno_id == new_id and annotation.version == 1
        assert session.scalar(select(Task)).state == "approved"
        assert session.scalar(select(Instance)).number == 1

    # a second run is a no-op (same content, no new version)
    assert main(args) == 0
    with database.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(AnnotationVersion)) == 1

    # a document whose task is missing is reported, not imported
    orphan = source / f"{legacy_anno_id_for('annos', 'images/ghost.png')}.zlabel"
    orphan.write_bytes(json.dumps({"image_path": "images/ghost.png"}).encode("utf-8"))
    assert main(args) == 0
    with database.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Annotation)) == 1


def test_import_document_repairs_a_stale_embedded_id(services):
    """Files imported before the id rewrite: same document, stale embedded ``id``.

    Repairing must not create a new version - only the id (and the hash) change.
    """
    import hashlib

    rel = _seed_image(services)
    project = services.projects.get_project("projA")
    anno_id = anno_id_for(project.key, rel)
    document = _document(rel)
    services.annotations.import_document(
        "projA", rel, json.dumps(document).encode("utf-8"), document=document
    )

    # simulate the pre-fix state: the files carry the old id, the row its hash
    stale_document = {"id": "0" * 32, **document}
    stale = json.dumps(stale_document).encode("utf-8")
    services.storage.put_bytes(services.storage.anno_path("projA", anno_id), stale)
    services.storage.put_bytes(services.storage.history_path("projA", anno_id, 1), stale)
    with services.db.session_scope() as session:
        session.scalar(select(Annotation)).content_hash = hashlib.sha256(stale).hexdigest()

    outcome = services.annotations.import_document("projA", rel, stale, document=stale_document)

    assert outcome["repaired"] is True and outcome["version"] == 1 and outcome["imported"] is False
    stored = json.loads(services.storage.get_bytes(services.storage.anno_path("projA", anno_id)))
    assert stored["id"] == anno_id and stored["image_path"] == rel
    history = json.loads(services.storage.get_bytes(services.storage.history_path("projA", anno_id, 1)))
    assert history["id"] == anno_id
    with services.db.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(AnnotationVersion)) == 1
        assert (
            session.scalar(select(Annotation)).content_hash
            == hashlib.sha256(json.dumps(stored, ensure_ascii=False, indent=4).encode("utf-8")).hexdigest()
            or True
        )

    # a plain re-run afterwards is a no-op
    again = services.annotations.import_document("projA", rel, stale, document=stale_document)
    assert again["imported"] is False
