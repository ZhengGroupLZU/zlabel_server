#!/bin/sh
# Container entrypoint for both services (API and inference worker).
#
# Three things the image alone cannot solve live here:
#   1. `./data` is a bind mount: the API creates `data/storage`, `data/uploads`,
#      the sqlite file and every annotation from inside the container, so the
#      mount must belong to the *runtime* user - not to root, which is what an
#      unconfigured container writes as.
#   2. The venv in /app belongs to root, so the server is started through
#      `/app/.venv/bin/*` instead of `uv run` (which would try to re-sync the
#      environment and fail on write access).
#   3. Ownership has to be repaired *before* privileges are dropped, and the
#      schema has to be migrated before the API accepts traffic.
#
# Knobs (all optional):
#   ZLABEL_UID / ZLABEL_GID  start as root, then run as this uid:gid.
#                            Unset or 0 = stay root (previous behaviour).
#                            These are also the ids used by the host-side
#                            `chown` hints below: pass `id -u` / `id -g`.
#   ZLABEL_CHOWN             auto (default) | always | never
#                            auto   = repair `data/` only when the storage root
#                                     is not owned by ZLABEL_UID (the first start
#                                     after a root-owned deployment, or after
#                                     somebody used `sudo` on the mount)
#                            always = walk and fix on every start
#                            never  = never touch ownership (NFS / read-only,
#                                     or when the host already owns the mount)
#   ZLABEL_MIGRATE           1 = run `alembic upgrade head` before starting.
#                            The worker service must leave this at 0.
#   ZLABEL_UMASK             umask for created files (default 022; use 002 with a
#                            shared group so several operators can write)
#   ZLABEL_DATA_DIR          data root to prepare (default /app/data)
#
# `docker compose exec` does *not* go through this script. With ZLABEL_UID set,
# reach for an admin command with:
#   docker compose exec -u "$ZLABEL_UID:$ZLABEL_GID" zlabel_server \
#       uv run python -m app.cli user ls
set -eu

APP_DIR="${APP_DIR:-/app}"
DATA_DIR="${ZLABEL_DATA_DIR:-$APP_DIR/data}"
STORAGE_ROOT="${ZLSERVER_STORAGE_ROOT:-$DATA_DIR/storage}"
RUNTIME_UID="${ZLABEL_UID:-0}"
RUNTIME_GID="${ZLABEL_GID:-0}"
CHOWN_MODE="${ZLABEL_CHOWN:-auto}"

log() { printf '[entrypoint] %s\n' "$*" >&2; }

umask "${ZLABEL_UMASK:-022}"

# --- ownership of the bind mount -----------------------------------------
repair_ownership() {
    # `find ! -user` stats every entry but chowns only the wrong ones, so a
    # second start is a cheap no-op even on a 10^5-file dataset.
    if find "$1" ! -user "$RUNTIME_UID" -exec chown "$RUNTIME_UID:$RUNTIME_GID" {} + 2>/dev/null; then
        log "ownership under $1 repaired (uid=$RUNTIME_UID gid=$RUNTIME_GID)"
    else
        log "WARNING: could not repair ownership under $1 (NFS/read-only mount?); continuing"
    fi
}

# Top level only: `data/` itself, the sqlite file + its -wal/-shm siblings, and
# the storage/upload roots. Cheap enough to run on every start.
fix_data_root() {
    if [ "$RUNTIME_UID" = "0" ] || [ "$CHOWN_MODE" = "never" ]; then
        return 0
    fi
    find "$DATA_DIR" -maxdepth 1 ! -user "$RUNTIME_UID" \
        -exec chown "$RUNTIME_UID:$RUNTIME_GID" {} + 2>/dev/null || true
}

prepare_data() {
    mkdir -p "$DATA_DIR"
    if [ "$RUNTIME_UID" = "0" ]; then
        return 0
    fi
    case "$CHOWN_MODE" in
        never)
            return 0
            ;;
        always)
            repair_ownership "$DATA_DIR"
            ;;
        *)
            if [ -n "$(find "$STORAGE_ROOT" -maxdepth 0 ! -user "$RUNTIME_UID" -print 2>/dev/null)" ]; then
                log "$STORAGE_ROOT is owned by another uid: repairing $DATA_DIR once"
                log "  (host alternative: sudo chown -R $RUNTIME_UID:$RUNTIME_GID <data dir>)"
                repair_ownership "$DATA_DIR"
            else
                fix_data_root
            fi
            ;;
    esac
}

# --- privilege drop ------------------------------------------------------
# The runtime uid has no passwd entry, so the services are exec'd through
# gosu/setpriv/su-exec (whichever the image ships). $DROPPER is that command as
# an unquoted word list; empty means "stay as we are" (root).
DROPPER=""

resolve_dropper() {
    if [ "$RUNTIME_UID" = "0" ]; then
        log "ZLABEL_UID unset: running as root (files in ./data stay root-owned)"
        return 0
    fi
    if command -v gosu >/dev/null 2>&1; then
        DROPPER="gosu $RUNTIME_UID:$RUNTIME_GID"
    elif command -v setpriv >/dev/null 2>&1; then
        DROPPER="setpriv --reuid $RUNTIME_UID --regid $RUNTIME_GID --clear-groups"
    elif command -v su-exec >/dev/null 2>&1; then
        DROPPER="su-exec $RUNTIME_UID:$RUNTIME_GID"
    else
        log "ERROR: no gosu/setpriv/su-exec in the image: cannot drop privileges, staying root"
        return 0
    fi
    log "running as $RUNTIME_UID:$RUNTIME_GID (via ${DROPPER%% *})"
    return 0
}

# A venv whose pyvenv.cfg `home =` points at a base interpreter the runtime uid
# cannot read (a uv-managed toolchain under /root, say) starts and *then* dies
# with "Fatal Python error: Failed to import encodings module", which says
# nothing about the cause. Report it here; this check only ever warns.
check_python() {
    interpreter="$APP_DIR/.venv/bin/python"
    if [ ! -x "$interpreter" ]; then
        return 0
    fi
    # shellcheck disable=SC2086  # $DROPPER is an unquoted command list, possibly empty
    if $DROPPER "$interpreter" -c "" >/dev/null 2>&1; then
        return 0
    fi
    base_home=$(sed -n 's/^home *= *//p' "$APP_DIR/.venv/pyvenv.cfg" 2>/dev/null | head -1)
    log "ERROR: $interpreter cannot start (base interpreter: ${base_home:-unknown})"
    log "       If that path sits under a directory this uid cannot read (e.g. /root,"
    log "       mode 0700), rebuild the image with UV_PYTHON_INSTALL_DIR outside it."
    return 0
}

# --- main ----------------------------------------------------------------
if [ "$(id -u)" = "0" ]; then
    prepare_data
    if [ "${ZLABEL_MIGRATE:-0}" = "1" ]; then
        log "alembic upgrade head"
        "$APP_DIR/.venv/bin/alembic" upgrade head
    fi
    # The migration above may have just created the sqlite file as root.
    fix_data_root
    resolve_dropper
    check_python
    # shellcheck disable=SC2086  # $DROPPER is an unquoted command list, possibly empty
    exec $DROPPER "$@"
fi

# Already started as a non-root user (compose `user:`): ownership cannot be
# repaired from here (no CAP_CHOWN), so only report what is about to break.
if [ ! -w "$DATA_DIR" ]; then
    log "ERROR: $DATA_DIR is not writable by $(id -u):$(id -g): annotation writes will fail"
    log "       Fix on the host: sudo chown -R $(id -u):$(id -g) <data dir>"
fi
check_python
exec "$@"
