"""Server administration CLI (no web UI needed).

    uv run python -m v2.cli user add alice --role reviewer
    uv run python -m v2.cli user ls
    uv run python -m v2.cli user passwd alice
    uv run python -m v2.cli storage usage
    uv run python -m v2.cli migrate-layout --root /data/zlabel --dry-run

The CLI talks to the database directly (same settings as the API), so it works
while the server is running and needs no admin session.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from ipaddress import ip_address  # noqa: F401  (kept for future bind checks)
from pathlib import Path


def _services(args):
    from v2.adapters.identity import LocalIdentity
    from v2.adapters.storage import build_storage
    from v2.core.config import Settings
    from v2.db.base import Database

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


def cmd_user_passwd(args) -> int:
    _, _, identity, _ = _services(args)
    password = args.password or getpass.getpass("New password: ")
    if not identity.set_password(args.name, password):
        print(f"no such user: {args.name}", file=sys.stderr)
        return 1
    print(f"password updated for {args.name!r}")
    return 0


def cmd_user_ls(args) -> int:
    _, database, _, _ = _services(args)
    from sqlalchemy import select

    from v2.db.models import User

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
    _, database, _, _ = _services(args)
    from sqlalchemy import func, select

    from v2.db.models import ROLES, User

    if args.role not in ROLES:
        print(f"unknown role {args.role!r} (choose from {', '.join(ROLES)})", file=sys.stderr)
        return 2
    with database.session_scope() as session:
        user = session.scalar(select(User).where(func.lower(User.name) == args.name.lower()))
        if user is None:
            print(f"no such user: {args.name}", file=sys.stderr)
            return 1
        user.role = args.role
        print(f"{user.name!r} is now {user.role}")
    return 0


# endregion


# region storage
def cmd_storage_usage(args) -> int:
    _, _, _, storage = _services(args)
    if not hasattr(storage, "usage"):
        print(f"backend {storage.kind!r} has no local usage information")
        return 0
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
