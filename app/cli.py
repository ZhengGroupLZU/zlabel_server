"""Server administration CLI (no web UI needed).

    uv run python -m app.cli user add alice --role reviewer
    uv run python -m app.cli user ls
    uv run python -m app.cli user passwd alice
    uv run python -m app.cli storage usage
    uv run python -m app.cli migrate-layout --root /data/zlabel --dry-run

The CLI talks to the database directly (same settings as the API), so it works
while the server is running and needs no admin session.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from ipaddress import ip_address  # noqa: F401  (kept for future bind checks)
from pathlib import Path


def _services(args):
    from app.adapters.identity import LocalIdentity
    from app.adapters.storage import build_storage
    from app.core.config import Settings
    from app.db.base import Database

    settings = Settings()
    if args.database_url:
        settings = settings.model_copy(update={"database_url": args.database_url})
    database = Database(settings.database_url)
    database.create_all()  # the API owns migrations; the CLI only needs the tables
    storage = build_storage(settings)
    return settings, database, LocalIdentity(database, settings), storage


# region user management
def cmd_user_add(args) -> int:
    _, _, identity, _ = _services(args)
    password = args.password or getpass.getpass("Password: ")
    user = identity.create_user(
        args.name, password, role=args.role, email=args.email or "", admin=args.role == "admin"
    )
    print(f"user {user['name']!r} ready (role={user['role']}, id={user['id']})")
    return 0


def _auth_service(args):
    """The service the account commands must go through (audit + session revocation)."""
    settings, database, identity, _ = _services(args)
    from app.services.auth_service import AuthService

    return AuthService(database, identity, settings)


def cmd_user_passwd(args) -> int:
    """Replace a password *through AuthService* so the sessions are revoked.

    Writing the hash directly would leave every existing token valid: sessions are
    opaque bearer tokens, they do not expire when the password changes.
    """
    auth = _auth_service(args)
    password = args.password or getpass.getpass("New password: ")
    user = auth.find_user(args.name)
    if user is None:
        print(f"no such user: {args.name}", file=sys.stderr)
        return 1
    auth.set_password(user.id, password)
    print(f"password updated for {user.name!r} (sessions revoked)")
    return 0


def cmd_user_ls(args) -> int:
    _, database, _, _ = _services(args)
    from sqlalchemy import select

    from app.db.models import User

    with database.session_scope() as session:
        users = list(session.scalars(select(User).order_by(User.name)))
    if not users:
        print("(no users)")
        return 0
    print(f"{'id':>4}  {'name':<20} {'role':<10} {'active':<7} password")
    for user in users:
        print(
            f"{user.id:>4}  {user.name:<20} {user.role:<10} "
            f"{'yes' if user.active else 'no':<7} {'set' if user.password_hash else '-'}"
        )
    return 0


def cmd_user_role(args) -> int:
    """Change a role through AuthService (audited, and the sessions are revoked)."""
    from app.db.models import ROLES

    auth = _auth_service(args)
    if args.role not in ROLES:
        print(f"unknown role {args.role!r} (choose from {', '.join(ROLES)})", file=sys.stderr)
        return 2
    user = auth.find_user(args.name)
    if user is None:
        print(f"no such user: {args.name}", file=sys.stderr)
        return 1
    updated = auth.update_user(user.id, role=args.role)
    print(f"{updated.name!r} is now {updated.role} (sessions revoked)")
    return 0


# endregion


# region projects
def _project_service(args):
    _, database, _, storage = _services(args)
    from app.core.config import Settings
    from app.services.project_service import ProjectService

    settings = Settings()
    if args.database_url:
        settings = settings.model_copy(update={"database_url": args.database_url})
    return database, ProjectService(database, storage, settings)


def cmd_project_ls(args) -> int:
    database, projects = _project_service(args)
    from sqlalchemy import func, select

    from app.db.models import ProjectMember, Task

    rows = projects.list_projects(active_only=False)
    if not rows:
        print("(no projects)")
        return 0
    print(f"{'id':>4}  {'name':<28} {'active':<7} {'tasks':>6} {'members':>8}")
    for project in rows:
        with database.session_scope() as session:
            tasks = session.scalar(
                select(func.count()).select_from(Task).where(Task.project_id == project.id)
            )
            members = session.scalar(
                select(func.count()).select_from(ProjectMember).where(ProjectMember.project_id == project.id)
            )
        print(
            f"{project.id:>4}  {project.name:<28} {'yes' if project.active else 'no':<7} "
            f"{tasks or 0:>6} {members or 0:>8}"
        )
    return 0


def cmd_project_reviewer(args) -> int:
    """Grant the project-reviewer role to a user for one project."""
    database, projects = _project_service(args)
    from sqlalchemy import func, select

    from app.db.models import User

    with database.session_scope() as session:
        user = session.scalar(select(User).where(func.lower(User.name) == args.user.lower()))
    if user is None:
        print(f"no such user: {args.user}", file=sys.stderr)
        return 1
    member = projects.add_member(args.project, user.id, args.role)
    print(f"{member['name']!r} is {member['role']} in {args.project!r}")
    return 0


def cmd_project_members(args) -> int:
    _database, projects = _project_service(args)
    members = projects.list_members(args.project)
    if not members:
        print(f"(no members in {args.project})")
        return 0
    for member in members:
        print(f"{member['user_id']:>4}  {member['name']:<20} {member['role']}")
    return 0


# endregion


# region storage
def cmd_sync_instances(args) -> int:
    """Backfill the instance registry from the stored annotation documents.

    Instances are mirrored on every save; this rebuilds the rows for datasets
    annotated before the instance tables existed (or after a repair).
    """
    settings, database, _, storage = _services(args)
    from app.services.instance_service import InstanceService
    from app.services.project_service import ProjectService

    projects = ProjectService(database, storage, settings)
    service = InstanceService(database, storage, projects, settings)
    if args.project:
        names = [args.project]
    else:
        names = [project.name for project in projects.list_projects(active_only=False)]
    total = {"documents": 0, "instances": 0, "results": 0}
    for name in names:
        try:
            stats = service.sync_project(name)
        except Exception as e:  # noqa: BLE001 - report and keep going with the rest
            print(f"{name}: error: {e}", file=sys.stderr)
            continue
        for key, value in stats.items():
            total[key] += value
        print(
            f"{name}: {stats['documents']} document(s), {stats['instances']} new instance(s), "
            f"{stats['results']} annotation link(s)"
        )
    print(f"total: {total['documents']} document(s), {total['instances']} instance(s) created")
    return 0


def cmd_import_annotations(args) -> int:
    """Import a folder of legacy ``<anno_id>.zlabel`` files into a server project.

    The documents are copied **byte-for-byte** under their new anno id
    (``sha256("<project key>/<rel>")``, derived from each document's
    ``image_path``), and the DB is brought in step: annotation row (+ history
    copy), label registry and instance mirror. The project must be scanned first
    (the task images are what create the tasks).
    """
    settings, database, _, storage = _services(args)
    from app.contracts.ids import anno_id_for
    from app.services.container import Services

    services = Services.build(settings, database, storage=storage)
    project = services.projects.get_project(args.project, active_only=False)
    source = Path(args.source).expanduser()
    if not source.is_dir():
        print(f"not a directory: {source}", file=sys.stderr)
        return 2
    files = sorted(source.glob("*.zlabel"))
    if not files:
        print(f"no *.zlabel files in {source}", file=sys.stderr)
        return 2

    stats = {"imported": 0, "already": 0, "repaired": 0, "conflict": 0, "no_image": 0, "unreadable": 0}
    for path in files:
        try:
            content = path.read_bytes()
            document = json.loads(content.decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
            stats["unreadable"] += 1
            print(f"  {path.name}: unreadable ({e})", file=sys.stderr)
            continue
        rel = str((document or {}).get("image_path") or "").strip()
        if not rel:
            stats["unreadable"] += 1
            print(f"  {path.name}: no image_path", file=sys.stderr)
            continue
        if not storage.exists(storage.image_path(project.name, rel)):
            stats["no_image"] += 1
            print(f"  {path.name}: no task image for {rel!r} (scan the project first)", file=sys.stderr)
            continue
        target = storage.anno_path(project.name, anno_id_for(project.key, rel))
        if storage.exists(target) and not args.overwrite and storage.get_bytes(target) != content:
            stats["conflict"] += 1
            print(f"  {rel}: target exists with different content (use --overwrite)", file=sys.stderr)
            continue
        try:
            # the service normalises the embedded "id" to the task's and writes
            # the document + its history copy itself
            outcome = services.annotations.import_document(
                project.name,
                rel,
                content,
                document=document,
                state=args.state or None,
                dry_run=args.dry_run,
            )
        except Exception as e:  # noqa: BLE001 - report and keep going
            stats["unreadable"] += 1
            print(f"  {rel}: {e}", file=sys.stderr)
            continue
        if outcome["imported"]:
            stats["imported"] += 1
        elif outcome.get("repaired"):
            stats["repaired"] += 1
        else:
            stats["already"] += 1

    verb = "would import" if args.dry_run else "imported"
    print(
        f"{verb} {stats['imported']} document(s); {stats['repaired']} id-repair(s); "
        f"{stats['already']} already present; {stats['conflict']} conflict(s); "
        f"{stats['no_image']} without a task image; {stats['unreadable']} unreadable"
    )
    if args.dry_run:
        print("dry run: nothing was written")
    else:
        print(f"project {project.name!r}: {len(files)} source file(s) processed")
    return 0


def cmd_migrate_anno_ids(args) -> int:
    """Re-key annotation files from the legacy ``md5(name/rel)`` ids to sha256 ids.

    Runs against ``ZLSERVER_STORAGE_ROOT`` + ``ZLSERVER_DATABASE_URL`` (the same
    settings as the API). Each project's stable key comes from
    ``<project>/.zlabel/project.json`` (created when missing); the files and the
    rows that referenced the old ids are updated together.
    """
    settings, database, _, storage = _services(args)
    from app.services.project_service import ProjectService

    service = ProjectService(database, storage, settings)
    if args.project:
        names = [args.project]
    else:
        names = [project.name for project in service.list_projects(active_only=False)]
    total = {"renamed": 0, "tasks": 0, "versions": 0, "skipped": 0}
    for name in names:
        try:
            stats = service.migrate_anno_ids(name, dry_run=args.dry_run)
        except Exception as e:  # noqa: BLE001 - report and keep going with the rest
            print(f"{name}: error: {e}", file=sys.stderr)
            continue
        for key, value in stats.items():
            total[key] += value
        verb = "would rename" if args.dry_run else "renamed"
        print(
            f"{name}: {verb} {stats['renamed']} file(s), {stats['tasks']} task(s), "
            f"{stats['versions']} version(s), {stats['skipped']} skipped"
        )
    verb = "would rename" if args.dry_run else "renamed"
    print(f"total: {verb} {total['renamed']} file(s) across {len(names)} project(s)")
    return 0


def cmd_storage_usage(args) -> int:
    _, _, _, storage = _services(args)
    stats = storage.usage()
    print(f"{stats['files']} files, {stats['bytes'] / 1024**2:.1f} MiB ({storage.root_dir})")
    return 0


def cmd_migrate_layout(args) -> int:
    """Move ``<project>/zlabel`` -> ``<project>/.zlabel/annos`` in place.

    The annotation *files* are addressed by ``anno_id`` on the client side, so the
    move is invisible to the desktop; it just has to happen once (and can be undone
    by swapping source and destination).
    """
    source_name = args.source
    target_name = args.target
    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    projects = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if args.project:
        projects = [p for p in projects if p.name == args.project]
    moved = files = 0
    for project in projects:
        source = project / source_name
        if not source.is_dir():
            continue
        target = project / target_name
        print(f"{project.name}: {source_name} -> {target_name}")
        if args.dry_run:
            files += sum(1 for p in source.rglob("*") if p.is_file())
            moved += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # merge: move every entry, keeping anything already there
            for entry in list(source.iterdir()):
                destination = target / entry.name
                if destination.exists():
                    print(f"  ! keeping existing {destination}", file=sys.stderr)
                    continue
                entry.replace(destination)
            source.rmdir()
        else:
            source.replace(target)
        files += sum(1 for p in target.rglob("*") if p.is_file())
        moved += 1

    verb = "would move" if args.dry_run else "moved"
    print(f"{verb} {moved} project(s), {files} file(s)")
    if not args.dry_run and moved:
        print("now set ZLSERVER_ANNO_DIR to the target directory (default) and restart the API")
    return 0


# endregion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zlabel-server", description="ZLabel server administration")
    parser.add_argument("--database-url", default="", help="override ZLSERVER_DATABASE_URL")
    sub = parser.add_subparsers(dest="command", required=True)

    user = sub.add_parser("user", help="manage accounts (local identity)")
    user_sub = user.add_subparsers(dest="user_command", required=True)

    add = user_sub.add_parser("add", help="create an account (or reset its password)")
    add.add_argument("name")
    add.add_argument("--role", default="annotator", help="annotator | reviewer | admin")
    add.add_argument("--email", default="")
    add.add_argument("--password", default="", help="skip the interactive prompt")
    add.set_defaults(func=cmd_user_add)

    passwd = user_sub.add_parser("passwd", help="set a new password")
    passwd.add_argument("name")
    passwd.add_argument("--password", default="")
    passwd.set_defaults(func=cmd_user_passwd)

    ls = user_sub.add_parser("ls", help="list accounts")
    ls.set_defaults(func=cmd_user_ls)

    role = user_sub.add_parser("role", help="change a role")
    role.add_argument("name")
    role.add_argument("role")
    role.set_defaults(func=cmd_user_role)

    project = sub.add_parser("project", help="projects and members")
    project_sub = project.add_subparsers(dest="project_command", required=True)
    pls = project_sub.add_parser("ls", help="list projects with task/member counts")
    pls.set_defaults(func=cmd_project_ls)
    members = project_sub.add_parser("members", help="list a project's members")
    members.add_argument("project")
    members.set_defaults(func=cmd_project_members)
    add_member = project_sub.add_parser("add-member", help="add/re-role a member")
    add_member.add_argument("project")
    add_member.add_argument("user")
    add_member.add_argument("--role", default="annotator")
    add_member.set_defaults(func=cmd_project_reviewer)

    storage = sub.add_parser("storage", help="storage helpers")
    storage_sub = storage.add_subparsers(dest="storage_command", required=True)
    usage = storage_sub.add_parser("usage", help="files/bytes under the storage root")
    usage.set_defaults(func=cmd_storage_usage)

    migrate = sub.add_parser("migrate-layout", help="move <project>/zlabel to <project>/.zlabel/annos")
    migrate.add_argument("--root", required=True, help="storage root (or a dataset directory)")
    migrate.add_argument("--source", default="zlabel")
    migrate.add_argument("--target", default=".zlabel/annos")
    migrate.add_argument("--project", default="", help="only this project")
    migrate.add_argument("--dry-run", action="store_true")
    migrate.set_defaults(func=cmd_migrate_layout)

    anno_ids = sub.add_parser("migrate-anno-ids", help="re-key annotations to sha256(<project key>/<rel>)")
    anno_ids.add_argument("--project", default="", help="only this project")
    anno_ids.add_argument("--dry-run", action="store_true")
    anno_ids.set_defaults(func=cmd_migrate_anno_ids)

    sync_instances = sub.add_parser(
        "sync-instances", help="mirror annotation documents into the instance registry"
    )
    sync_instances.add_argument("--project", default="", help="only this project")
    sync_instances.set_defaults(func=cmd_sync_instances)

    import_annos = sub.add_parser(
        "import-annotations",
        help="import legacy <anno_id>.zlabel files (re-keyed + DB rows + mirrors)",
    )
    import_annos.add_argument("--source", required=True, help="folder holding the .zlabel files")
    import_annos.add_argument("--project", required=True, help="target server project (already scanned)")
    import_annos.add_argument("--state", default="", help="set this task state (e.g. approved)")
    import_annos.add_argument("--overwrite", action="store_true", help="replace differing targets")
    import_annos.add_argument("--dry-run", action="store_true")
    import_annos.set_defaults(func=cmd_import_annotations)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from app.core.errors import ApiError

    try:
        return int(args.func(args))
    except ApiError as e:  # 404 unknown project, 422 bad role, ...
        print(f"error: {e.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
