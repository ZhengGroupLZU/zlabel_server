"""Admin UI views: dashboard, users, projects (with per-project detail) and files.

Two layers, on purpose:

- **Write pages** (Users, Projects, Files) call the services directly, exactly like
  the API does, so roles/permissions/audit have one implementation. Everything with
  an invariant goes through ``AuthService`` / ``ProjectService``.
- **Audit log** stays a read-only starlette-admin ``ModelView``.

The menu is Dashboard / Users / Projects / Files / Audit log; project-scoped data
(tasks, labels, members, files) lives on the project detail page (``?project=<name>``
plus a ``tab`` query parameter).
"""

from __future__ import annotations

import io
import re
from typing import Any
from urllib.parse import urlencode

from markupsafe import Markup
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette_admin.contrib.sqla import ModelView
from starlette_admin.fields import StringField
from starlette_admin.flash import flash
from starlette_admin.routing import route
from starlette_admin.security.csrf import csrf_input
from starlette_admin.views import CustomView

from v2.admin.auth import admin_context
from v2.core.errors import ApiError
from v2.core.logging import get_logger
from v2.db.models import ROLE_ANNOTATOR, ROLES, STATES, AuditLog
from v2.services.container import Services
from v2.services.label_palette import LABEL_PALETTE, normalize_color, pick_color

logger = get_logger("zlabel.v2.admin")

ROLE_BADGES = {"admin": "bg-primary", "reviewer": "bg-warning", "annotator": "bg-secondary"}
STATE_BADGES = {
    "draft": "bg-secondary",
    "submitted": "bg-warning",
    "approved": "bg-success",
    "rejected": "bg-danger",
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
TASK_PAGE_SIZE = 50


# region html helpers
def _e(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fmt_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _card(title: str, body: str, *, actions: str = "") -> str:
    header = f'<div class="card-header"><h3 class="card-title">{_e(title)}</h3></div>'
    if actions:
        header = (
            '<div class="card-header d-flex justify-content-between align-items-center">'
            f'<h3 class="card-title">{_e(title)}</h3><div>{actions}</div></div>'
        )
    return f'<div class="card">{header}<div class="card-body">{body}</div></div>'


def _badge(text: str, color: str) -> str:
    return f'<span class="badge {color} text-white">{_e(text)}</span>'


def _role_badge(role: str) -> str:
    return _badge(role, ROLE_BADGES.get(role, "bg-secondary"))


def _state_badge(state: str) -> str:
    return _badge(state, STATE_BADGES.get(state, "bg-secondary"))


def _table(
    headers: list[str], rows: list[str], empty: str, *, colspan: int | None = None, table_id: str = ""
) -> str:
    span = colspan or len(headers)
    body = "".join(rows) or f'<tr><td colspan="{span}" class="text-muted">({_e(empty)})</td></tr>'
    head = "".join(f"<th>{_e(header)}</th>" for header in headers)
    identity = f' id="{_e(table_id)}"' if table_id else ""
    return (
        f'<div class="table-responsive"><table{identity} class="table table-sm table-vcenter">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_SUFFIXES)


# region label colour picker (dropdown of swatch + hex, or a typed hex code)
LABEL_PICKER_STYLE = """<style>
.zl-color{display:flex;flex-wrap:nowrap;gap:.35rem;align-items:center}
.zl-hex{width:7.5rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.zl-preview{width:1.6rem;height:1.6rem;border-radius:.3rem;border:1px solid rgba(0,0,0,.2);display:inline-block;flex:0 0 auto}
.zl-color-menu{position:fixed;z-index:1080;max-height:16rem;overflow:auto;min-width:9rem;padding:.25rem;
  background:var(--tblr-bg-surface,#fff);border:1px solid rgba(0,0,0,.15);border-radius:.35rem;
  box-shadow:0 6px 18px rgba(0,0,0,.18)}
.zl-color-item{display:flex;align-items:center;gap:.5rem;width:100%;padding:.2rem .45rem;border:0;
  background:transparent;border-radius:.25rem;cursor:pointer;text-align:left}
.zl-color-item:hover,.zl-color-item.active{background:rgba(32,107,196,.14)}
.zl-color-dot{width:1rem;height:1rem;border-radius:.2rem;border:1px solid rgba(0,0,0,.2);display:inline-block;flex:0 0 auto}
</style>"""

LABEL_PICKER_SCRIPT = """<script>
(function () {
  function norm(value) {
    var v = String(value || '').trim().toLowerCase().replace(/^#/, '');
    if (/^[0-9a-f]{3}$/.test(v)) v = v[0] + v[0] + v[1] + v[1] + v[2] + v[2];
    return /^[0-9a-f]{6}$/.test(v) ? '#' + v : '';
  }
  function paint(input) {
    var color = norm(input.value);
    var preview = document.querySelector('[data-preview-for="' + input.id + '"]');
    if (preview) preview.style.background = color || '#ffffff';
    var dot = document.querySelector('[data-toggle-preview="' + input.id + '"] [data-toggle-dot]');
    if (dot) dot.style.background = color || '#ffffff';
    document.querySelectorAll('[data-target="' + input.id + '"]').forEach(function (item) {
      item.classList.toggle('active', item.dataset.color === color);
    });
  }
  function closeMenus() {
    document.querySelectorAll('[data-zl-color-menu]').forEach(function (menu) { menu.hidden = true; });
  }
  function openMenu(toggle, menu) {
    closeMenus();
    var box = toggle.getBoundingClientRect();
    menu.hidden = false;
    menu.style.left = box.left + 'px';
    menu.style.top = (box.bottom + 2) + 'px';
  }
  document.addEventListener('click', function (e) {
    var toggle = e.target.closest ? e.target.closest('[data-zl-color-toggle]') : null;
    if (toggle) {
      var menu = toggle.parentElement.querySelector('[data-zl-color-menu]');
      if (menu) { if (menu.hidden) { openMenu(toggle, menu); } else { closeMenus(); } }
      return;
    }
    var item = e.target.closest ? e.target.closest('[data-color-swatch]') : null;
    if (item) {
      var input = document.getElementById(item.dataset.target);
      if (input) {
        input.value = item.dataset.color;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        paint(input);
      }
      closeMenus();
      return;
    }
    closeMenus();
  });
  document.addEventListener('input', function (e) {
    var input = e.target;
    if (input && input.classList && input.classList.contains('zl-hex')) paint(input);
  });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeMenus(); });
  window.addEventListener('scroll', closeMenus, true);
})();
</script>"""


def _color_picker(input_id: str, value: str, *, name: str = "color") -> str:
    """One label colour control: a hex field plus a dropdown of palette entries.

    Every entry shows its colour and the hex code; the field accepts ``#rrggbb``,
    ``#rgb`` and a bare ``rrggbb`` (the server normalises too), so a code can
    always be typed instead of picked. ``name`` is the submitted form field (the
    bulk form on the Labels page uses ``color-<label id>``).
    """
    current = normalize_color(value) or "#000000"
    items = "".join(
        f'<button type="button" class="zl-color-item{" active" if color == current else ""}" '
        f'data-color-swatch data-color="{color}" data-target="{input_id}">'
        f'<span class="zl-color-dot" style="background:{color}"></span><code>{color}</code></button>'
        for color in LABEL_PALETTE
    )
    return (
        '<div class="zl-color">'
        f'<input type="text" name="{name}" id="{input_id}" class="form-control form-control-sm zl-hex" '
        f'value="{_e(value)}" placeholder="#rrggbb" '
        'pattern="#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})" maxlength="7" autocomplete="off" '
        'spellcheck="false" title="hex colour: #rrggbb, #rgb or rrggbb">'
        f'<span class="zl-preview" data-preview-for="{input_id}" style="background:{current}"></span>'
        '<div class="zl-color-dropdown">'
        '<button type="button" class="btn btn-sm btn-outline-secondary d-flex align-items-center gap-1" '
        f'data-zl-color-toggle data-toggle-preview="{input_id}" title="pick a palette colour">'
        f'<span class="zl-color-dot" data-toggle-dot style="background:{current}"></span>\u25be</button>'
        f'<div class="zl-color-menu" data-zl-color-menu hidden>{items}</div>'
        "</div></div>"
    )


# endregion


def _media_type(name: str) -> str:
    lowered = name.lower()
    for suffix, media in MEDIA_TYPES.items():
        if lowered.endswith(suffix):
            return media
    return "application/octet-stream"


def _thumbnail(content: bytes, size: int = 320) -> bytes | None:
    """Downscale an image for the preview grid; ``None`` when it cannot be read.

    Pillow is already a dependency (the client uploads task images as PNG/JPEG), and
    the grid uses ``loading="lazy"`` so only visible cells pay for this.
    """
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((size, size))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception as e:  # noqa: BLE001 - a broken image is not a server error
        logger.warning(f"thumbnail failed: {e}")
        return None


class _PageView(CustomView):
    """A CustomView that renders hand-built HTML inside the admin layout."""

    path = "/"
    menu_label = ""

    def __init__(self, services: Services) -> None:
        super().__init__()
        self.services = services

    # region plumbing
    def render(self, request: Request, html: str) -> Response:
        assert self._admin is not None, "view must be mounted before use"
        return self._admin._template_response(
            request=request,
            name="index.html",
            context={
                "title": self.title(request),
                "widget": None,
                # index.html does ``{{ widget_html }}`` and Jinja autoescapes plain
                # strings, so the page has to be marked safe (the library's own
                # render_widget returns Markup for the same reason).
                "widget_html": Markup(html),
                "widget_additional_css": [],
                "widget_additional_js": [],
            },
        )

    def url(self, request: Request, suffix: str = "", **params: Any) -> str:
        """Absolute URL of this page (or a route below it) plus query parameters.

        A leading slash on ``suffix`` must not produce ``//``: the dashboard's own
        path is ``/``, and ``/admin//scan`` used to 404.
        """
        base = str(self._admin.base_url if self._admin is not None else "/admin").rstrip("/")
        target = re.sub(r"/{2,}", "/", f"{base}{self.path}{suffix}")
        query = urlencode({key: value for key, value in params.items() if value not in (None, "")})
        return f"{target}?{query}" if query else target

    def back(
        self,
        request: Request,
        suffix: str = "",
        *,
        ok: str = "",
        error: str = "",
        **params: Any,
    ) -> RedirectResponse:
        if ok:
            flash(request, ok, "success")
        if error:
            flash(request, error, "error")
        return RedirectResponse(self.url(request, suffix, **params), status_code=303)

    def actor_id(self, request: Request) -> int | None:
        ctx = admin_context(self.services, request)
        return ctx.user_id if ctx is not None else None

    def csrf(self, request: Request) -> str:
        return str(csrf_input(request))

    # endregion


# region dashboard
class DashboardView(_PageView):
    """Landing page: storage, task progress, the project table and a rescan button."""

    def __init__(self, services: Services) -> None:
        super().__init__(services)
        self.menu_label = "Dashboard"
        self.icon = "fa-solid fa-gauge"
        self.path = "/"
        # route_name must stay "index": starlette-admin derives the admin index URL from it
        self.route_name = "index"

    @route("")
    async def index(self, request: Request) -> Response:  # noqa: D102 - rendered page
        return self.render(request, self.body(request))

    @route("/scan", methods=["POST"])
    async def scan(self, request: Request) -> Response:
        try:
            stats = self.services.projects.scan_and_sync(force=True)
        except ApiError as e:
            return self.back(request, error=e.message)
        return self.back(
            request,
            ok=(
                f"scan done: {stats['projects']} project(s), {stats['tasks']} task(s), "
                f"{stats['missing']} missing, {stats['deactivated']} deactivated"
            ),
        )

    # region body
    def body(self, request: Request) -> str:
        storage = self.services.storage
        usage = storage.usage()
        projects = self.services.projects.list_projects(active_only=False)
        stats = self.services.projects.project_stats()
        progress = self.services.projects.progress()

        cards = [
            (
                "Projects",
                f"{len(projects)} ({sum(1 for p in projects if p.active)} active)",
                self.url(request, suffix="projects"),
            ),
            ("Tasks", str(progress.get("total", 0)), self.url(request, suffix="projects")),
            (
                "Reviewed",
                " · ".join(f"{state} {progress.get(state, 0)}" for state in STATES),
                self.url(request, suffix="audit-log/list"),
            ),
            (
                "Storage",
                f"{usage['files']} files, {_fmt_bytes(usage['bytes'])}<br>"
                f"<code>{_e(storage.root_dir)}</code>",
                self.url(request, suffix="files"),
            ),
        ]
        parts = ['<div class="row row-cards">']
        for title, value, link in cards:
            parts.append(
                '<div class="col-sm-6 col-lg-3"><div class="card"><div class="card-body">'
                f'<div class="text-muted">{_e(title)}</div>'
                f'<div class="h2 m-0">{value}</div>'
                f'<a class="small" href="{link}">open</a>'
                "</div></div></div>"
            )
        parts.append("</div>")

        rows = [
            "<tr>"
            f'<td><a href="{self.url(request, suffix="/projects", project=project.name)}">'
            f"<code>{_e(project.name)}</code></a></td>"
            f"<td>{_e(project.display_name)}</td>"
            f"<td>{_badge('active' if project.active else 'inactive', 'bg-success' if project.active else 'bg-secondary')}</td>"
            f"<td>{stats.get(project.name, {}).get('tasks', 0)}</td>"
            f"<td>{stats.get(project.name, {}).get('members', 0)}</td>"
            "</tr>"
            for project in projects
        ]
        table = _table(["project", "display name", "state", "tasks", "members"], rows, "no projects yet")
        scan = (
            f'<form method="post" action="{self.url(request, "/scan")}" class="d-inline">'
            f"{self.csrf(request)}"
            '<button class="btn btn-sm btn-outline-primary" type="submit">Rescan storage</button>'
            "</form>"
        )
        info = (
            f"<div><span class='text-muted'>root</span> <code>{_e(storage.root_dir)}</code></div>"
            f"<div><span class='text-muted'>annotations</span> "
            f"<code>{_e(self.services.settings.anno_dir_clean)}</code></div>"
            f"<div><span class='text-muted'>usage</span> {usage['files']} files, "
            f"{_fmt_bytes(usage['bytes'])}</div>"
        )
        return (
            "".join(parts)
            + '<div class="row row-cards mt-3">'
            + f'<div class="col-lg-4">{_card("Storage", info)}</div>'
            + f'<div class="col-12">{_card("Projects", table, actions=scan)}</div>'
            + "</div>"
        )

    # endregion


# endregion


# region users
class UsersView(_PageView):
    """Accounts: list + filter + create + role/enabled + password reset.

    Every write goes through :class:`~v2.services.auth_service.AuthService`, which
    audits it and revokes the target's sessions when the role changes or the
    account is disabled.
    """

    def __init__(self, services: Services) -> None:
        super().__init__(services)
        self.menu_label = "Users"
        self.icon = "fa-solid fa-users"
        self.path = "/users"

    @route("")
    async def index(self, request: Request) -> Response:  # noqa: D102 - rendered page
        return self.render(request, self.body(request))

    @route("/create", methods=["POST"])
    async def create(self, request: Request) -> Response:
        form = await request.form()
        try:
            user = self.services.auth.create_user(
                str(form.get("name") or ""),
                str(form.get("password") or ""),
                role=str(form.get("role") or ROLE_ANNOTATOR),
                email=str(form.get("email") or ""),
                actor_id=self.actor_id(request),
            )
        except ApiError as e:
            return self.back(request, error=e.message)
        return self.back(request, ok=f"created {user.name!r}")

    @route("/update", methods=["POST"])
    async def update(self, request: Request) -> Response:
        form = await request.form()
        try:
            user = self.services.auth.update_user(
                int(str(form.get("user_id") or 0)),
                role=str(form.get("role") or "") or None,
                active="active" in form,
                actor_id=self.actor_id(request),
            )
        except (ApiError, ValueError) as e:
            return self.back(request, error=getattr(e, "message", str(e)))
        return self.back(request, ok=f"updated {user.name!r} (sessions revoked)")

    @route("/password", methods=["POST"])
    async def password(self, request: Request) -> Response:
        form = await request.form()
        try:
            self.services.auth.set_password(
                int(str(form.get("user_id") or 0)),
                str(form.get("password") or ""),
                actor_id=self.actor_id(request),
            )
        except (ApiError, ValueError) as e:
            return self.back(request, error=getattr(e, "message", str(e)))
        return self.back(request, ok="password updated (sessions revoked)")

    # region body
    @staticmethod
    def _role_select(name: str, current: str) -> str:
        options = "".join(
            f'<option value="{role}"{" selected" if role == current else ""}>{role}</option>'
            for role in ROLES
        )
        return f'<select name="{name}" class="form-select form-select-sm w-auto">{options}</select>'

    def body(self, request: Request) -> str:
        csrf = self.csrf(request)
        users = self.services.auth.list_users()

        query = str(request.query_params.get("q") or "").strip().lower()
        role = str(request.query_params.get("role") or "")
        active = str(request.query_params.get("active") or "")

        def matches(user) -> bool:
            if query and query not in user.name.lower() and query not in (user.email or "").lower():
                return False
            if role and user.role != role:
                return False
            if active == "1" and not user.active:
                return False
            if active == "0" and user.active:
                return False
            return True

        rows = []
        for user in [u for u in users if matches(u)]:
            rows.append(
                "<tr>"
                f"<td><code>{_e(user.name)}</code></td>"
                f"<td>{_e(user.email)}</td>"
                f"<td>{user.finished_count}</td>"
                f"<td>{_e(user.last_login_at.strftime('%Y-%m-%d %H:%M') if user.last_login_at else '')}</td>"
                "<td>"
                f'<form method="post" action="{self.url(request, "/update")}" class="d-flex gap-2 align-items-center">'
                f"{csrf}<input type='hidden' name='user_id' value='{user.id}'>"
                f"{self._role_select('role', user.role)}"
                "<label class='form-check form-check-inline m-0'>"
                f"<input type='checkbox' name='active' value='1' class='form-check-input'"
                f"{' checked' if user.active else ''}>"
                "<span class='form-check-label'>active</span></label>"
                "<button class='btn btn-sm btn-primary' type='submit'>Save</button></form>"
                "</td>"
                "<td>"
                f'<form method="post" action="{self.url(request, "/password")}" class="d-flex gap-2">'
                f"{csrf}<input type='hidden' name='user_id' value='{user.id}'>"
                "<input type='password' name='password' class='form-control form-control-sm' "
                "placeholder='new password' autocomplete='new-password' required>"
                "<button class='btn btn-sm btn-outline-secondary' type='submit'>Reset</button></form>"
                "</td>"
                "</tr>"
            )
        table = _table(
            ["name", "email", "finished", "last login", "role / enabled", "password"],
            rows,
            "no accounts match these filters",
        )
        filters = (
            f'<form method="get" action="{self.url(request)}" class="row g-2 align-items-end mb-3">'
            '<div class="col-auto"><label class="form-label">search</label>'
            f'<input name="q" value="{_e(query)}" class="form-control form-control-sm"></div>'
            '<div class="col-auto"><label class="form-label">role</label><select name="role" '
            'class="form-select form-select-sm">'
            f'<option value="">(any)</option>'
            + "".join(
                f'<option value="{item}"{" selected" if item == role else ""}>{item}</option>'
                for item in ROLES
            )
            + "</select></div>"
            '<div class="col-auto"><label class="form-label">enabled</label>'
            '<select name="active" class="form-select form-select-sm">'
            f'<option value=""{" selected" if not active else ""}>(any)</option>'
            f'<option value="1"{" selected" if active == "1" else ""}>active</option>'
            f'<option value="0"{" selected" if active == "0" else ""}>disabled</option>'
            "</select></div>"
            '<div class="col-auto"><button class="btn btn-sm btn-outline-primary" type="submit">Filter</button> '
            f'<a class="btn btn-sm btn-link" href="{self.url(request)}">reset</a></div></form>'
        )
        create = (
            f'<form method="post" action="{self.url(request, "/create")}" class="vstack gap-2">'
            f"{csrf}"
            '<div><label class="form-label">name</label>'
            '<input name="name" class="form-control" required autocomplete="off"></div>'
            '<div><label class="form-label">email</label><input name="email" class="form-control"></div>'
            '<div><label class="form-label">role</label>'
            f"{self._role_select('role', ROLE_ANNOTATOR)}</div>"
            '<div><label class="form-label">password</label>'
            '<input type="password" name="password" class="form-control" required '
            'autocomplete="new-password" minlength="6"></div>'
            '<button class="btn btn-primary" type="submit">Create account</button>'
            "</form>"
        )
        return (
            f'<div class="row row-cards">'
            f'<div class="col-lg-4">{_card("Create account", create)}</div>'
            f'<div class="col-lg-8">{_card("Accounts", filters + table)}</div>'
            f"</div>"
        )

    # endregion


# endregion


# region projects
class ProjectsView(_PageView):
    """Projects: list/filter/create, and a per-project page for everything scoped to it.

    Detail tabs: overview (rename/metadata), files (browse/upload/preview/delete),
    members, labels and tasks (the task table with its state filter).
    """

    def __init__(self, services: Services) -> None:
        super().__init__(services)
        self.menu_label = "Projects"
        self.icon = "fa-solid fa-folder-tree"
        self.path = "/projects"

    # region routes
    @route("")
    async def index(self, request: Request) -> Response:  # noqa: D102 - rendered page
        project = str(request.query_params.get("project") or "").strip()
        if not project:
            return self.render(request, self._list_body(request))
        try:
            return self.render(request, self._detail_body(request, project))
        except ApiError as e:
            flash(request, e.message, "error")
            return RedirectResponse(self.url(request), status_code=303)

    @route("/create", methods=["POST"])
    async def create(self, request: Request) -> Response:
        form = await request.form()
        try:
            project = self.services.projects.create_project(
                str(form.get("name") or ""),
                str(form.get("display_name") or ""),
                timeline="timeline" in form,
                actor_id=self.actor_id(request),
            )
        except ApiError as e:
            return self.back(request, error=e.message)
        return self.back(request, ok=f"created project {project.name!r}", project=project.name)

    @route("/update", methods=["POST"])
    async def update(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        try:
            self.services.projects.update_project(
                project,
                display_name=str(form.get("display_name") or ""),
                description=str(form.get("description") or ""),
                active="active" in form,
                timeline="timeline" in form,
                actor_id=self.actor_id(request),
            )
        except ApiError as e:
            return self.back(request, error=e.message, project=project)
        return self.back(request, ok="project updated", project=project)

    @route("/scan", methods=["POST"])
    async def scan(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        try:
            stats = self.services.projects.scan_and_sync(force=True)
        except ApiError as e:
            return self.back(request, error=e.message, project=project)
        return self.back(
            request,
            ok=f"scan done: {stats['projects']} project(s), {stats['tasks']} task(s)",
            project=project,
        )

    @route("/delete", methods=["POST"])
    async def delete(self, request: Request) -> Response:
        """Delete a project: registry rows + (optionally) the directory on disk."""
        form = await request.form()
        project = str(form.get("project") or "").strip()
        confirm = str(form.get("confirm") or "").strip()
        if confirm != project:
            return self.back(
                request,
                error=f"type {project!r} exactly to confirm the deletion",
                project=project,
                tab="overview",
            )
        try:
            stats = self.services.projects.delete_project(
                project,
                delete_files="delete_files" in form,
                actor_id=self.actor_id(request),
            )
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="overview")
        return self.back(
            request,
            ok=(
                f"project {project!r} deleted ({stats['tasks']} task(s)"
                + (", files removed)" if stats["files_deleted"] else ", files kept)")
            ),
        )

    @route("/member/save", methods=["POST"])
    async def member_save(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        try:
            member = self.services.projects.add_member(
                project,
                int(str(form.get("user_id") or 0)),
                str(form.get("role") or ROLE_ANNOTATOR),
                actor_id=self.actor_id(request),
            )
        except (ApiError, ValueError) as e:
            return self.back(request, error=getattr(e, "message", str(e)), project=project, tab="members")
        return self.back(
            request, ok=f"{member['name']!r} is {member['role']}", project=project, tab="members"
        )

    @route("/member/delete", methods=["POST"])
    async def member_delete(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        try:
            self.services.projects.remove_member(
                project,
                int(str(form.get("user_id") or 0)),
                actor_id=self.actor_id(request),
            )
        except (ApiError, ValueError) as e:
            return self.back(request, error=getattr(e, "message", str(e)), project=project, tab="members")
        return self.back(request, ok="member removed", project=project, tab="members")

    @route("/label/save", methods=["POST"])
    async def label_save(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        label_id = str(form.get("label_id") or "").strip()
        name = str(form.get("name") or "").strip()
        color = str(form.get("color") or "#000000")
        raw_sort = str(form.get("sort") or "").strip()
        try:
            sort = int(raw_sort) if raw_sort else None
        except ValueError:
            sort = None
        try:
            if label_id:
                self.services.projects.update_label(
                    project,
                    int(label_id),
                    name=name,
                    color=color,
                    sort=sort,
                    archived="archived" in form,
                    actor_id=self.actor_id(request),
                )
                ok = f"label {name!r} updated"
            else:
                self.services.projects.create_label(
                    project, name, color=color, sort=sort, actor_id=self.actor_id(request)
                )
                ok = f"label {name!r} created"
        except (ApiError, ValueError) as e:
            return self.back(request, error=getattr(e, "message", str(e)), project=project, tab="labels")
        return self.back(request, ok=ok, project=project, tab="labels")

    @route("/label/save-all", methods=["POST"])
    async def label_save_all(self, request: Request) -> Response:
        """Apply every row of the Labels table (fields + drag order) in one go."""
        form = await request.form()
        project = str(form.get("project") or "")
        raw = str(form.get("order") or "")
        try:
            order = [int(part) for part in raw.split(",") if part.strip()]
        except ValueError:
            return self.back(request, error="invalid label order", project=project, tab="labels")
        try:
            labels = self.services.projects.list_labels(project, include_archived=True)
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="labels")
        edits = {
            label.id: {
                "name": form.get(f"name-{label.id}"),
                "color": form.get(f"color-{label.id}"),
                "archived": f"archived-{label.id}" in form,
            }
            for label in labels
        }
        try:
            result = self.services.projects.save_labels(
                project, edits=edits, order=order, actor_id=self.actor_id(request)
            )
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="labels")
        ok = f"saved {result['changed']} label(s)" if result["changed"] else "no changes"
        if result["reordered"]:
            ok += " (order updated)"
        return self.back(request, ok=ok, project=project, tab="labels")

    @route("/label/delete", methods=["POST"])
    async def label_delete(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        try:
            self.services.projects.delete_label(
                project, int(str(form.get("label_id") or 0)), actor_id=self.actor_id(request)
            )
        except (ApiError, ValueError) as e:
            return self.back(request, error=getattr(e, "message", str(e)), project=project, tab="labels")
        return self.back(request, ok="label deleted", project=project, tab="labels")

    @route("/instance/create", methods=["POST"])
    async def instance_create(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        raw_number = str(form.get("number") or "").strip()
        try:
            number = int(raw_number) if raw_number else None
        except ValueError:
            return self.back(
                request, error="the instance number must be an integer", project=project, tab="instances"
            )
        try:
            created = self.services.instances.create_instance(
                project,
                number=number,
                name=str(form.get("name") or ""),
                note=str(form.get("note") or ""),
                status=str(form.get("status") or ""),
                color=str(form.get("color") or ""),
                actor_id=self.actor_id(request),
            )
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="instances")
        return self.back(
            request, ok=f"instance {created['number']} created", project=project, tab="instances"
        )

    @route("/instance/save-all", methods=["POST"])
    async def instance_save_all(self, request: Request) -> Response:
        """Apply every row of the Instances table in one go."""
        form = await request.form()
        project = str(form.get("project") or "")
        try:
            rows = self.services.instances.list_instances(project, include_archived=True, with_stats=False)
            edits = {
                row["number"]: {
                    "name": form.get(f"name-{row['number']}"),
                    "status": form.get(f"status-{row['number']}"),
                    "color": form.get(f"color-{row['number']}"),
                    "note": form.get(f"note-{row['number']}"),
                    "archived": f"archived-{row['number']}" in form,
                }
                for row in rows
            }
            result = self.services.instances.save_instances(
                project, edits=edits, actor_id=self.actor_id(request)
            )
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="instances")
        ok = f"saved {result['changed']} instance(s)" if result["changed"] else "no changes"
        return self.back(request, ok=ok, project=project, tab="instances")

    @route("/instance/delete", methods=["POST"])
    async def instance_delete(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        try:
            number = int(str(form.get("number") or 0))
        except ValueError:
            return self.back(request, error="invalid instance number", project=project, tab="instances")
        try:
            self.services.instances.delete_instance(project, number, actor_id=self.actor_id(request))
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="instances")
        return self.back(
            request,
            ok=f"instance {number} removed from the registry",
            project=project,
            tab="instances",
        )

    @route("/upload", methods=["POST"])
    async def upload(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        rel = self._clean_rel(str(form.get("rel") or ""))
        upload = form.get("file")
        if upload is None or not getattr(upload, "filename", ""):
            return self.back(request, error="no file selected", project=project, tab="files", rel=rel)
        content = await upload.read()
        limit = self.services.settings.max_upload_bytes
        if len(content) > limit:
            return self.back(
                request, error=f"file exceeds {limit} bytes", project=project, tab="files", rel=rel
            )
        target = f"{self._virtual(project, rel)}/{upload.filename}"
        try:
            self.services.storage.put_bytes(target, content)
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="files", rel=rel)
        return self.back(request, ok=f"uploaded {upload.filename}", project=project, tab="files", rel=rel)

    @route("/mkdir", methods=["POST"])
    async def mkdir(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        rel = self._clean_rel(str(form.get("rel") or ""))
        name = str(form.get("name") or "").strip().strip("/")
        if not name or "/" in name or name in (".", ".."):
            return self.back(request, error="invalid directory name", project=project, tab="files", rel=rel)
        try:
            self.services.storage.ensure_dir(f"{self._virtual(project, rel)}/{name}")
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="files", rel=rel)
        return self.back(request, ok=f"created {name!r}", project=project, tab="files", rel=rel)

    @route("/delete-file", methods=["POST"])
    async def delete_file(self, request: Request) -> Response:
        form = await request.form()
        project = str(form.get("project") or "")
        rel = self._clean_rel(str(form.get("rel") or ""))
        try:
            self.services.storage.delete(self._virtual(project, rel))
        except ApiError as e:
            parent = rel.rpartition("/")[0]
            return self.back(request, error=e.message, project=project, tab="files", rel=parent)
        return self.back(
            request, ok=f"deleted {rel!r}", project=project, tab="files", rel=rel.rpartition("/")[0]
        )

    @route("/download")
    async def download(self, request: Request) -> Response:
        project = str(request.query_params.get("project") or "")
        rel = self._clean_rel(str(request.query_params.get("rel") or ""))
        try:
            content = self.services.storage.get_bytes(self._virtual(project, rel))
        except ApiError as e:
            return self.back(request, error=e.message, project=project, tab="files")
        name = rel.rpartition("/")[2] or "download.bin"
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @route("/preview")
    async def preview(self, request: Request) -> Response:
        """Inline image for the preview grid / a full-size view."""
        project = str(request.query_params.get("project") or "")
        rel = self._clean_rel(str(request.query_params.get("rel") or ""))
        thumb = request.query_params.get("thumb") == "1"
        try:
            content = self.services.storage.get_bytes(self._virtual(project, rel))
        except ApiError as e:
            return Response(content=e.message, media_type="text/plain", status_code=404)
        media = _media_type(rel)
        if thumb and media != "application/octet-stream":
            small = _thumbnail(content)
            if small is not None:
                content, media = small, "image/png"
        return Response(content=content, media_type=media, headers={"Cache-Control": "private, max-age=60"})

    # endregion

    # region helpers
    @staticmethod
    def _clean_rel(rel: str) -> str:
        """A project-relative path with any traversal segments dropped ("" = root)."""
        parts = [part for part in str(rel).replace("\\", "/").split("/") if part not in ("", ".", "..")]
        return "/".join(parts)

    def _virtual(self, project: str, rel: str) -> str:
        root = self.services.storage.project_dir(project)
        clean = self._clean_rel(rel)
        return f"{root}/{clean}" if clean else root

    def _counts(self) -> dict[str, dict[str, int]]:
        return self.services.projects.project_stats()

    # endregion

    # region list
    def _list_body(self, request: Request) -> str:
        csrf = self.csrf(request)
        query = str(request.query_params.get("q") or "").strip().lower()
        active = str(request.query_params.get("active") or "")
        stats = self._counts()
        projects = self.services.projects.list_projects(active_only=False)

        def matches(project) -> bool:
            if query and query not in project.name.lower() and query not in project.display_name.lower():
                return False
            if active == "1" and not project.active:
                return False
            if active == "0" and project.active:
                return False
            return True

        rows = []
        for project in [p for p in projects if matches(p)]:
            counts = stats.get(project.name, {})
            rows.append(
                "<tr>"
                f'<td><a href="{self.url(request, project=project.name)}"><code>{_e(project.name)}</code></a></td>'
                f"<td>{_e(project.display_name)}</td>"
                f"<td>{_badge('active' if project.active else 'inactive', 'bg-success' if project.active else 'bg-secondary')}</td>"
                f"<td>{_badge('timeline' if project.timeline else 'flat', 'bg-primary' if project.timeline else 'bg-secondary')}</td>"
                f"<td>{counts.get('tasks', 0)}</td>"
                f"<td>{counts.get('members', 0)}</td>"
                f"<td>{_e(project.updated_at.strftime('%Y-%m-%d %H:%M') if project.updated_at else '')}</td>"
                f'<td class="d-flex gap-1"><a class="btn btn-sm btn-outline-primary" '
                f'href="{self.url(request, project=project.name)}">open</a>'
                f'<a class="btn btn-sm btn-outline-danger" title="open the project to confirm deletion" '
                f'href="{self.url(request, project=project.name, tab="overview")}#danger-zone">delete</a></td>'
                "</tr>"
            )
        table = _table(
            ["project", "display name", "state", "tasks", "timeline", "members", "updated", ""],
            rows,
            "no projects match these filters",
        )
        filters = (
            f'<form method="get" action="{self.url(request)}" class="row g-2 align-items-end mb-3">'
            '<div class="col-auto"><label class="form-label">search</label>'
            f'<input name="q" value="{_e(query)}" class="form-control form-control-sm"></div>'
            '<div class="col-auto"><label class="form-label">state</label>'
            '<select name="active" class="form-select form-select-sm">'
            f'<option value=""{" selected" if not active else ""}>(any)</option>'
            f'<option value="1"{" selected" if active == "1" else ""}>active</option>'
            f'<option value="0"{" selected" if active == "0" else ""}>inactive</option>'
            "</select></div>"
            '<div class="col-auto"><button class="btn btn-sm btn-outline-primary" type="submit">Filter</button> '
            f'<a class="btn btn-sm btn-link" href="{self.url(request)}">reset</a></div></form>'
        )
        create = (
            f'<form method="post" action="{self.url(request, "/create")}" class="vstack gap-2">'
            f"{csrf}"
            '<div><label class="form-label">directory name</label>'
            '<input name="name" class="form-control" required autocomplete="off" placeholder="projA"></div>'
            '<div><label class="form-label">display name</label>'
            '<input name="display_name" class="form-control"></div>'
            "<label class='form-check form-check-inline'>"
            "<input type='checkbox' name='timeline' value='1' class='form-check-input' checked>"
            "<span class='form-check-label'>timeline (sequence: group + D{n} day)</span></label>"
            '<button class="btn btn-primary" type="submit">Create project</button></form>'
        )
        scan = (
            f'<form method="post" action="{self.url(request, "/scan")}" class="d-inline-flex gap-2">'
            f"{csrf}"
            '<span class="text-muted small align-self-center">scanning is manual only</span>'
            '<button class="btn btn-sm btn-outline-primary" type="submit" '
            'title="re-walk the storage tree and sync the task table">Refresh projects</button>'
            "</form>"
        )
        return (
            '<div class="row row-cards">'
            f'<div class="col-lg-4">{_card("Create project", create)}</div>'
            f'<div class="col-lg-8">{_card("Projects", filters + table, actions=scan)}</div>'
            "</div>"
        )

    # endregion

    # region detail
    def _tabs(self, request: Request, project: str, active: str) -> str:
        tabs = (
            ("overview", "Overview"),
            ("files", "Files"),
            ("tasks", "Tasks"),
            ("labels", "Labels"),
            ("instances", "Instances"),
            ("members", "Members"),
        )
        links = "".join(
            f'<li class="nav-item"><a class="nav-link{" active" if key == active else ""}" '
            f'href="{self.url(request, project=project, tab=key)}">{label}</a></li>'
            for key, label in tabs
        )
        return f'<ul class="nav nav-tabs mb-3">{links}</ul>'

    def _detail_body(self, request: Request, project_name: str) -> str:
        project = self.services.projects.get_project(project_name, active_only=False)
        tab = str(request.query_params.get("tab") or "overview")
        counts = self._counts().get(project.name, {})
        header = (
            '<div class="d-flex justify-content-between align-items-center mb-2">'
            f'<div><h2 class="m-0">{_e(project.display_name or project.name)} '
            f"{_badge('active' if project.active else 'inactive', 'bg-success' if project.active else 'bg-secondary')}</h2>"
            f'<div class="text-muted"><code>{_e(project.name)}</code> · {counts.get("tasks", 0)} tasks · '
            f"{counts.get('members', 0)} members</div></div>"
            f'<a class="btn btn-sm btn-outline-secondary" href="{self.url(request)}">← all projects</a></div>'
        )
        sections = {
            "overview": self._overview,
            "files": self._files,
            "members": self._members,
            "labels": self._labels,
            "instances": self._instances,
            "tasks": self._tasks,
        }
        render = sections.get(tab, self._overview)
        return (
            header
            + self._tabs(request, project.name, tab)
            + f'<div class="row row-cards">{render(request, project.name)}</div>'
        )

    # region detail: overview
    def _overview(self, request: Request, project_name: str) -> str:
        csrf = self.csrf(request)
        project = self.services.projects.get_project(project_name, active_only=False)
        counts = self._counts().get(project.name, {})
        root = self.services.storage.project_dir(project.name)
        form = (
            f'<form method="post" action="{self.url(request, "/update")}" class="vstack gap-2">'
            f"{csrf}"
            f'<input type="hidden" name="project" value="{_e(project.name)}">'
            '<div><label class="form-label">display name (rename)</label>'
            f'<input name="display_name" class="form-control" value="{_e(project.display_name)}"></div>'
            '<div><label class="form-label">description</label>'
            f'<textarea name="description" class="form-control" rows="3">{_e(project.description)}</textarea></div>'
            "<label class='form-check form-check-inline'>"
            f"<input type='checkbox' name='active' class='form-check-input'"
            f"{' checked' if project.active else ''}>"
            "<span class='form-check-label'>active (visible to clients)</span></label>"
            "<label class='form-check form-check-inline'>"
            f"<input type='checkbox' name='timeline' class='form-check-input'"
            f"{' checked' if project.timeline else ''}>"
            "<span class='form-check-label'>timeline (sequence: group + D{n} day)</span></label>"
            '<button class="btn btn-primary" type="submit">Save</button>'
            "</form>"
        )
        info = (
            f"<div><span class='text-muted'>identity</span> <code>{_e(project.name)}</code></div>"
            f"<div><span class='text-muted'>key</span> <code>{_e(project.key)}</code> "
            "<span class='text-muted'>(anno_id namespace, from .zlabel/project.json)</span></div>"
            f"<div><span class='text-muted'>directory</span> <code>{_e(root)}</code></div>"
            f"<div><span class='text-muted'>files</span> "
            f'<a href="{self.url(request, project=project.name, tab="files")}">browse</a></div>'
            f"<div><span class='text-muted'>tasks / members</span> "
            f"{counts.get('tasks', 0)} / {counts.get('members', 0)}</div>"
            f"<div><span class='text-muted'>created / updated</span> "
            f"{_e(project.created_at.strftime('%Y-%m-%d %H:%M') if project.created_at else '')} / "
            f"{_e(project.updated_at.strftime('%Y-%m-%d %H:%M') if project.updated_at else '')}</div>"
        )
        scan = (
            f'<form method="post" action="{self.url(request, "/scan")}" class="d-inline">'
            f"{csrf}<input type='hidden' name='project' value='{_e(project.name)}'>"
            '<button class="btn btn-sm btn-outline-primary" type="submit">Rescan storage</button></form>'
        )
        danger = (
            f'<form method="post" action="{self.url(request, "/delete")}" '
            f"onsubmit=\"return confirm('Permanently delete project {_e(project.name)}?')\">{csrf}"
            f"<input type='hidden' name='project' value='{_e(project.name)}'>"
            '<p class="text-muted mb-2">Removes the project from the registry (tasks, annotations, '
            "history, labels and members) and, when the box is ticked, the whole directory "
            f"<code>{_e(root)}</code>. This cannot be undone.</p>"
            '<div class="row g-2 align-items-end">'
            '<div class="col-auto"><label class="form-label">type the project name to confirm</label>'
            f'<input name="confirm" class="form-control" placeholder="{_e(project.name)}" '
            'required autocomplete="off"></div>'
            "<div class='col-auto'><label class='form-check'>"
            "<input type='checkbox' name='delete_files' value='1' class='form-check-input' checked>"
            "<span class='form-check-label'>also delete the files on disk</span></label></div>"
            "<div class='col-auto'><button class='btn btn-danger' type='submit'>Delete project</button></div>"
            "</div></form>"
        )
        return (
            f'<div class="col-lg-6">{_card("Metadata", form)}</div>'
            f'<div class="col-lg-6">{_card("Storage", info, actions=scan)}</div>'
            '<div class="col-12"><p class="text-muted m-0">Annotations are addressed by the project key '
            "above (stored in <code>.zlabel/project.json</code>), so the display name can be renamed "
            "freely. Renaming the directory by hand requires "
            "<code>uv run python -m v2.cli migrate-anno-ids</code>. <strong>Timeline</strong> off "
            "means the tasks are single images: the task table keeps <code>group</code>/"
            "<code>day</code> empty (toggling recomputes the existing tasks).</p></div>"
            f'<div class="col-12" id="danger-zone">{_card("Danger zone", danger)}</div>'
        )

    # endregion

    # region detail: files
    def _files(self, request: Request, project_name: str) -> str:
        csrf = self.csrf(request)
        rel = self._clean_rel(str(request.query_params.get("rel") or ""))
        path = self._virtual(project_name, rel)
        parent = rel.rpartition("/")[0]
        directories = self.services.storage.list_dirs(path)
        files: list[tuple[str, str, int]] = []
        for full in self.services.storage.glob_files(path):
            folder, _, name = full.rpartition("/")
            if folder != path.rstrip("/"):
                continue
            files.append((name, full, self.services.storage.file_info(full).size))

        crumbs = [f'<a href="{self.url(request, project=project_name, tab="files")}">{_e(project_name)}</a>']
        walked = ""
        for part in rel.split("/") if rel else []:
            walked = f"{walked}/{part}" if walked else part
            crumbs.append(
                f'<a href="{self.url(request, project=project_name, tab="files", rel=walked)}">{_e(part)}</a>'
            )

        rows = []
        for name in directories:
            child = f"{rel}/{name}" if rel else name
            rows.append(
                "<tr>"
                f'<td><a href="{self.url(request, project=project_name, tab="files", rel=child)}">'
                f"📁 {_e(name)}</a></td><td></td><td></td></tr>"
            )
        for name, _full, size in files:
            child = f"{rel}/{name}" if rel else name
            thumb = (
                f'<img src="{self.url(request, "/preview", project=project_name, rel=child, thumb=1)}" '
                'class="rounded border" style="height:36px" loading="lazy" alt="">'
                if _is_image(name)
                else ""
            )
            preview = (
                f'<a class="btn btn-sm btn-outline-secondary" target="_blank" '
                f'href="{self.url(request, "/preview", project=project_name, rel=child)}">Preview</a>'
                if _is_image(name)
                else ""
            )
            rows.append(
                "<tr>"
                f"<td>{thumb} {_e(name)}</td>"
                f"<td>{_fmt_bytes(size)}</td>"
                "<td class='d-flex gap-1'>"
                f"{preview}"
                f'<a class="btn btn-sm btn-outline-secondary" href="{self.url(request, "/download", project=project_name, rel=child)}">Download</a>'
                f'<form method="post" action="{self.url(request, "/delete-file")}" '
                f"onsubmit=\"return confirm('Delete {_e(name)}?')\">{csrf}"
                f"<input type='hidden' name='project' value='{_e(project_name)}'>"
                f"<input type='hidden' name='rel' value='{_e(child)}'>"
                "<button class='btn btn-sm btn-outline-danger' type='submit'>Delete</button></form>"
                "</td></tr>"
            )
        listing = (
            '<p class="mb-2">'
            + (
                f'<a href="{self.url(request, project=project_name, tab="files", rel=parent)}">↑ up</a> · '
                if rel
                else ""
            )
            + " / ".join(crumbs)
            + "</p>"
            + _table(["name", "size", ""], rows, "empty directory")
        )
        upload = (
            f'<form method="post" action="{self.url(request, "/upload")}" enctype="multipart/form-data" '
            f'class="vstack gap-2">{csrf}'
            f"<input type='hidden' name='project' value='{_e(project_name)}'>"
            f"<input type='hidden' name='rel' value='{_e(rel)}'>"
            '<div><label class="form-label">upload into <code>'
            f"{_e(rel or '/')}</code></label>"
            '<input type="file" name="file" class="form-control" required></div>'
            '<button class="btn btn-primary" type="submit">Upload</button></form>'
        )
        mkdir = (
            f'<form method="post" action="{self.url(request, "/mkdir")}" class="vstack gap-2">{csrf}'
            f"<input type='hidden' name='project' value='{_e(project_name)}'>"
            f"<input type='hidden' name='rel' value='{_e(rel)}'>"
            '<div><label class="form-label">new sub-directory</label>'
            '<input name="name" class="form-control" required></div>'
            '<button class="btn btn-outline-primary" type="submit">Create directory</button></form>'
        )
        return (
            f'<div class="col-12">{_card("Files", listing)}</div>'
            f'<div class="col-lg-6">{_card("Upload", upload)}</div>'
            f'<div class="col-lg-6">{_card("New directory", mkdir)}</div>'
        )

    # endregion

    # region detail: members
    def _members(self, request: Request, project_name: str) -> str:
        csrf = self.csrf(request)
        members = self.services.projects.list_members(project_name)
        member_ids = {member["user_id"] for member in members}
        users = [u for u in self.services.auth.list_users() if u.active]
        rows = []
        for member in members:
            options = "".join(
                f'<option value="{role}"{" selected" if role == member["role"] else ""}>{role}</option>'
                for role in ROLES
            )
            rows.append(
                "<tr>"
                f"<td><code>{_e(member['name'])}</code></td>"
                "<td>"
                f'<form method="post" action="{self.url(request, "/member/save")}" class="d-flex gap-2">'
                f"{csrf}<input type='hidden' name='project' value='{_e(project_name)}'>"
                f"<input type='hidden' name='user_id' value='{member['user_id']}'>"
                f'<select name="role" class="form-select form-select-sm w-auto">{options}</select>'
                "<button class='btn btn-sm btn-primary' type='submit'>Save</button></form>"
                "</td>"
                "<td>"
                f'<form method="post" action="{self.url(request, "/member/delete")}">{csrf}'
                f"<input type='hidden' name='project' value='{_e(project_name)}'>"
                f"<input type='hidden' name='user_id' value='{member['user_id']}'>"
                "<button class='btn btn-sm btn-outline-danger' type='submit'>Remove</button></form>"
                "</td></tr>"
            )
        table = _table(["user", "role", ""], rows, "no members yet")
        candidates = [u for u in users if u.id not in member_ids]
        user_options = "".join(f'<option value="{u.id}">{_e(u.name)}</option>' for u in candidates)
        role_options = "".join(f'<option value="{role}">{role}</option>' for role in ROLES)
        add = (
            f'<form method="post" action="{self.url(request, "/member/save")}" class="d-flex gap-2">'
            f"{csrf}<input type='hidden' name='project' value='{_e(project_name)}'>"
            f'<select name="user_id" class="form-select" required>'
            f'<option value="">(choose a user)</option>{user_options}</select>'
            f'<select name="role" class="form-select w-auto">{role_options}</select>'
            '<button class="btn btn-primary" type="submit">Add member</button></form>'
        )
        return (
            f'<div class="col-12">{_card("Members", table)}</div>'
            f'<div class="col-lg-6">{_card("Add participant", add)}</div>'
        )

    # endregion

    # region detail: labels
    def _labels(self, request: Request, project_name: str) -> str:
        """The label registry: one form for the whole table, saved by one button.

        Every row contributes ``name-<id>`` / ``color-<id>`` / ``archived-<id>``; the
        drag handle rewrites the hidden ``order`` field without submitting, so the
        admin can reorder and edit freely and persist everything with "Save all".
        Delete uses the HTML ``form="..."`` attribute (a nested form would be invalid).
        """
        csrf = self.csrf(request)
        labels = self.services.projects.list_labels(project_name, include_archived=True)
        rows = []
        delete_forms = []
        for index, label in enumerate(labels):
            color = label.color if str(label.color).startswith("#") else "#000000"
            rows.append(
                f'<tr data-label-id="{label.id}">'
                '<td class="text-muted" style="cursor:grab; user-select: none" draggable="true" '
                'title="drag to reorder">⠿</td>'
                f'<td class="text-muted">{index}</td>'
                "<td>"
                f'<input name="name-{label.id}" class="form-control form-control-sm" '
                f'style="max-width:14rem" value="{_e(label.name)}" required>'
                "</td><td>"
                + _color_picker(f"label-color-{label.id}", color, name=f"color-{label.id}")
                + "</td><td>"
                "<label class='form-check form-check-inline m-0'>"
                f"<input type='checkbox' name='archived-{label.id}' value='1' class='form-check-input'"
                f"{' checked' if label.archived else ''}><span class='form-check-label'>archived</span></label>"
                "</td><td>"
                f'<button class="btn btn-sm btn-outline-danger" type="submit" '
                f'form="label-delete-{label.id}" '
                f"onclick=\"return confirm('Delete label {_e(label.name)}?')\">Delete</button>"
                "</td></tr>"
            )
            delete_forms.append(
                f'<form id="label-delete-{label.id}" method="post" '
                f'action="{self.url(request, "/label/delete")}">{csrf}'
                f"<input type='hidden' name='project' value='{_e(project_name)}'>"
                f"<input type='hidden' name='label_id' value='{label.id}'></form>"
            )
        table = _table(["", "id", "label", "colour", "", ""], rows, "no labels yet", table_id="label-table")
        bulk = (
            f'<form id="label-bulk-form" method="post" action="{self.url(request, "/label/save-all")}">'
            f"{csrf}<input type='hidden' name='project' value='{_e(project_name)}'>"
            "<input type='hidden' name='order' id='label-order-value' value=''>" + table + "</form>"
        )
        # The drag only reorders the DOM and rewrites the hidden order field; the
        # "Save all" button persists fields + order together. The script must come
        # *after* the table (a parser-time lookup of an element below returns null)
        # and ``dropEffect = 'move'`` matters: without it the cursor stays the
        # "no-drop" icon even though the drop is allowed.
        script = """
<script>
(function () {
  function init() {
    var table = document.getElementById('label-table');
    var form = document.getElementById('label-bulk-form');
    var order = document.getElementById('label-order-value');
    var hint = document.getElementById('label-dirty');
    if (!table || !form || !order || !table.tBodies.length) return;
    var tbody = table.tBodies[0];
    var dragged = null;
    function rowOf(target) {
      return target && target.closest ? target.closest('tr[data-label-id]') : null;
    }
    function syncOrder() {
      order.value = Array.prototype.map.call(
        tbody.querySelectorAll('tr[data-label-id]'),
        function (row) { return row.dataset.labelId; }
      ).join(',');
    }
    function markDirty() { if (hint) hint.hidden = false; }
    syncOrder();
    form.addEventListener('input', markDirty);
    form.addEventListener('change', markDirty);
    tbody.addEventListener('dragstart', function (e) {
      var row = rowOf(e.target);
      if (!row) return;
      dragged = row;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', row.dataset.labelId || 'row');
      if (e.dataTransfer.setDragImage) e.dataTransfer.setDragImage(row, 12, 12);
      row.classList.add('table-active');
    });
    tbody.addEventListener('dragend', function () {
      if (dragged) dragged.classList.remove('table-active');
      dragged = null;
    });
    tbody.addEventListener('dragover', function (e) {
      if (!dragged) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      var row = rowOf(e.target);
      if (!row || row === dragged) return;
      var box = row.getBoundingClientRect();
      var after = (e.clientY - box.top) > box.height / 2;
      tbody.insertBefore(dragged, after ? row.nextSibling : row);
    });
    tbody.addEventListener('drop', function (e) {
      e.preventDefault();
      if (!dragged) return;
      syncOrder();
      markDirty();
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
"""
        save_all = (
            '<div class="d-flex align-items-center gap-2 mt-2">'
            '<button class="btn btn-primary" type="submit" form="label-bulk-form" id="label-save-all">'
            "Save all labels</button>"
            '<span class="text-warning" id="label-dirty" hidden>unsaved changes</span>'
            "</div>"
        )
        create = (
            f'<form method="post" action="{self.url(request, "/label/save")}" '
            'class="d-flex flex-wrap gap-2 align-items-center">'
            f"{csrf}<input type='hidden' name='project' value='{_e(project_name)}'>"
            '<input name="name" class="form-control" style="max-width:16rem" '
            'placeholder="label name" required>'
            + _color_picker("label-color-new", pick_color(label.color for label in labels))
            + '<button class="btn btn-primary" type="submit">Add label</button></form>'
        )
        note = (
            '<p class="text-muted mt-2">The <strong>id</strong> is the 0-based position '
            "(the class id used by the COCO/YOLO exports). Drag a row by its first-column "
            "<strong>⠿</strong> handle to reorder, edit the fields, then click "
            "<strong>Save all labels</strong> to persist fields and order together. "
            "Colour: open the palette list (colour + hex per entry) or type a code "
            "(<code>#rrggbb</code>, <code>#rgb</code> or a bare <code>rrggbb</code>).</p>"
        )
        return (
            f'<div class="col-12">{_card("Labels", LABEL_PICKER_STYLE + bulk + save_all + note + "".join(delete_forms) + LABEL_PICKER_SCRIPT + script)}</div>'
            f'<div class="col-lg-6">{_card("Add label", create)}</div>'
        )

    # endregion

    # region detail: instances
    def _instances(self, request: Request, project_name: str) -> str:
        """The project's instances: mirrored from the documents, edited here.

        The number is the id the annotation documents reference, so it is shown
        read-only; status/name/note/colour/archived are editable and saved with one
        button (like the Labels tab).
        """
        csrf = self.csrf(request)
        instances = self.services.instances.list_instances(project_name, include_archived=True)
        presets = self.services.instances.status_presets(project_name)
        options = "".join(f'<option value="{_e(status)}"></option>' for status in presets)
        datalist = f'<datalist id="instance-statuses">{options}</datalist>'
        rows = []
        delete_forms = []
        for item in instances:
            number = item["number"]
            color = item["color"] if str(item["color"]).startswith("#") else "#000000"
            rows.append(
                f'<tr data-instance-number="{number}">'
                f'<td class="text-muted">{number}</td>'
                "<td>"
                f'<input name="name-{number}" class="form-control form-control-sm" style="max-width:11rem" '
                f'value="{_e(item["name"])}" placeholder="(optional)">'
                "</td><td>"
                f'<input name="status-{number}" class="form-control form-control-sm" style="max-width:11rem" '
                f'list="instance-statuses" value="{_e(item["status"])}">'
                "</td><td>"
                + _color_picker(f"instance-color-{number}", color, name=f"color-{number}")
                + "</td><td>"
                "<label class='form-check form-check-inline m-0'>"
                f"<input type='checkbox' name='archived-{number}' value='1' class='form-check-input'"
                f"{' checked' if item['archived'] else ''}><span class='form-check-label'>archived</span></label>"
                "</td>"
                f'<td class="text-muted">{item["results"]}</td>'
                f'<td class="text-muted">{item["tasks"]}</td>'
                "<td>"
                f'<input name="note-{number}" class="form-control form-control-sm" style="min-width:10rem" '
                f'value="{_e(item["note"])}" placeholder="(optional)">'
                "</td><td>"
                f'<button class="btn btn-sm btn-outline-danger" type="submit" '
                f'form="instance-delete-{number}" '
                f"onclick=\"return confirm('Remove instance {number} from the registry?')\">Delete</button>"
                "</td></tr>"
            )
            delete_forms.append(
                f'<form id="instance-delete-{number}" method="post" '
                f'action="{self.url(request, "/instance/delete")}">{csrf}'
                f"<input type='hidden' name='project' value='{_e(project_name)}'>"
                f"<input type='hidden' name='number' value='{number}'></form>"
            )
        table = _table(
            ["id", "name", "status", "colour", "", "results", "tasks", "note", ""],
            rows,
            "no instances yet - save an annotation that uses instance ids",
        )
        bulk = (
            f'<form id="instance-bulk-form" method="post" '
            f'action="{self.url(request, "/instance/save-all")}">'
            f"{csrf}<input type='hidden' name='project' value='{_e(project_name)}'>"
            + datalist
            + table
            + "</form>"
        )
        save_all = (
            '<div class="d-flex align-items-center gap-2 mt-2">'
            '<button class="btn btn-primary" type="submit" form="instance-bulk-form" '
            'id="instance-save-all">Save all instances</button>'
            '<span class="text-warning" id="instance-dirty" hidden>unsaved changes</span>'
            "</div>"
        )
        script = """<script>
(function () {
  function init() {
    var form = document.getElementById('instance-bulk-form');
    var hint = document.getElementById('instance-dirty');
    if (!form || !hint) return;
    function markDirty() { hint.hidden = false; }
    form.addEventListener('input', markDirty);
    form.addEventListener('change', markDirty);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>"""
        create = (
            f'<form method="post" action="{self.url(request, "/instance/create")}" '
            'class="d-flex flex-wrap gap-2 align-items-center">'
            f"{csrf}<input type='hidden' name='project' value='{_e(project_name)}'>"
            '<input name="number" class="form-control" style="width:6rem" placeholder="id" '
            'title="leave empty for the next free number">'
            '<input name="name" class="form-control" style="max-width:12rem" placeholder="name">'
            '<input name="status" class="form-control" style="max-width:11rem" list="instance-statuses" '
            'placeholder="status">'
            + _color_picker("instance-color-new", pick_color(item["color"] for item in instances))
            + '<button class="btn btn-primary" type="submit">Add instance</button></form>'
        )
        note = (
            '<p class="text-muted mt-2">Instances are mirrored from the annotation documents '
            "(<code>results[*].instance_id</code> / <code>annotations.instances</code>); the "
            "<strong>id</strong> is the number the documents reference, so it cannot be "
            "renumbered here. <strong>results</strong>/<strong>tasks</strong> count the "
            "annotations currently linked to the instance. Removing a row only clears the "
            "registry - a later save that still uses the number recreates it.</p>"
        )
        return (
            f'<div class="col-12">{_card("Instances", LABEL_PICKER_STYLE + bulk + save_all + note + "".join(delete_forms) + LABEL_PICKER_SCRIPT + script)}</div>'
            f'<div class="col-lg-6">{_card("Add instance", create)}</div>'
        )

    # endregion

    # region detail: tasks
    def _tasks(self, request: Request, project_name: str) -> str:
        state = str(request.query_params.get("state") or "")
        query = str(request.query_params.get("q") or "").strip()
        offset = max(0, int(str(request.query_params.get("offset") or 0) or 0))
        try:
            rows, total = self.services.tasks.list_tasks(
                project_name, state=state or None, search=query or None, limit=TASK_PAGE_SIZE, offset=offset
            )
        except ApiError as e:
            return f'<p class="text-danger">{_e(e.message)}</p>'

        body = []
        for row in rows:
            task = row.task
            preview = (
                f'<a class="btn btn-sm btn-outline-secondary" target="_blank" '
                f'href="{self.url(request, "/preview", project=project_name, rel=task.rel_path)}">Preview</a>'
            )
            body.append(
                "<tr>"
                f"<td>{preview}</td>"
                f"<td><code>{_e(task.rel_path)}</code></td>"
                f"<td>{_e(task.group_name)}</td>"
                f"<td>{task.day}</td>"
                f"<td>{_state_badge(task.state)}</td>"
                f"<td>{row.version}</td>"
                f"<td>{_e(row.holder)}</td>"
                f"<td>{_e(task.updated_at.strftime('%Y-%m-%d %H:%M') if task.updated_at else '')}</td>"
                "</tr>"
            )
        table = _table(
            ["", "task", "group", "day", "state", "version", "claimed by", "updated"],
            body,
            "no tasks match these filters",
        )
        state_options = "".join(
            f'<option value="{item}"{" selected" if item == state else ""}>{item}</option>' for item in STATES
        )
        filters = (
            f'<form method="get" action="{self.url(request)}" class="row g-2 align-items-end mb-3">'
            f'<input type="hidden" name="project" value="{_e(project_name)}">'
            '<input type="hidden" name="tab" value="tasks">'
            '<div class="col-auto"><label class="form-label">state</label>'
            f'<select name="state" class="form-select form-select-sm">'
            f'<option value=""{" selected" if not state else ""}>(any)</option>{state_options}</select></div>'
            '<div class="col-auto"><label class="form-label">search</label>'
            f'<input name="q" value="{_e(query)}" class="form-control form-control-sm" placeholder="path"></div>'
            '<div class="col-auto"><button class="btn btn-sm btn-outline-primary" type="submit">Filter</button> '
            f'<a class="btn btn-sm btn-link" href="{self.url(request, project=project_name, tab="tasks")}">reset</a>'
            "</div></form>"
        )
        pages = ""
        if total > TASK_PAGE_SIZE:
            links = []
            if offset > 0:
                links.append(
                    f'<a class="btn btn-sm btn-outline-secondary" href="'
                    f'{self.url(request, project=project_name, tab="tasks", state=state, q=query, offset=max(0, offset - TASK_PAGE_SIZE))}">← newer</a>'
                )
            if offset + TASK_PAGE_SIZE < total:
                links.append(
                    f'<a class="btn btn-sm btn-outline-secondary" href="'
                    f'{self.url(request, project=project_name, tab="tasks", state=state, q=query, offset=offset + TASK_PAGE_SIZE)}">older →</a>'
                )
            pages = (
                f'<div class="d-flex justify-content-between align-items-center mb-2">'
                f'<span class="text-muted">{offset + 1}–{min(offset + TASK_PAGE_SIZE, total)} of {total}</span>'
                f'<span class="d-flex gap-2">{"".join(links)}</span></div>'
            )
        return f'<div class="col-12">{_card("Tasks", filters + pages + table)}</div>'

    # endregion

    # endregion


# endregion


# region files
class FilesView(_PageView):
    """Browse the whole storage root: upload, preview, download and clean up."""

    def __init__(self, services: Services) -> None:
        super().__init__(services)
        self.menu_label = "Files"
        self.icon = "fa-solid fa-folder-open"
        self.path = "/files"

    # region routes
    @route("")
    async def index(self, request: Request) -> Response:  # noqa: D102 - rendered page
        try:
            return self.render(request, self.body(request))
        except ApiError as e:
            flash(request, e.message, "error")
            return RedirectResponse(self.url(request), status_code=303)

    @route("/upload", methods=["POST"])
    async def upload(self, request: Request) -> Response:
        form = await request.form()
        path = str(form.get("path") or "/")
        upload = form.get("file")
        if upload is None or not getattr(upload, "filename", ""):
            return self.back(request, error="no file selected", path=path)
        content = await upload.read()
        limit = self.services.settings.max_upload_bytes
        if len(content) > limit:
            return self.back(request, error=f"file exceeds {limit} bytes", path=path)
        target = f"{path.rstrip('/')}/{upload.filename}"
        try:
            self.services.storage.put_bytes(target, content)
        except ApiError as e:
            return self.back(request, error=e.message, path=path)
        return self.back(request, ok=f"uploaded {upload.filename}", path=path)

    @route("/mkdir", methods=["POST"])
    async def mkdir(self, request: Request) -> Response:
        form = await request.form()
        path = str(form.get("path") or "/")
        name = str(form.get("name") or "").strip().strip("/")
        if not name or "/" in name or name in (".", ".."):
            return self.back(request, error="invalid directory name", path=path)
        try:
            self.services.storage.ensure_dir(f"{path.rstrip('/')}/{name}")
        except ApiError as e:
            return self.back(request, error=e.message, path=path)
        return self.back(request, ok=f"created {name!r}", path=path)

    @route("/delete", methods=["POST"])
    async def delete(self, request: Request) -> Response:
        form = await request.form()
        target = str(form.get("path") or "")
        try:
            self.services.storage.delete(target)
        except ApiError as e:
            return self.back(request, error=e.message, path=target.rpartition("/")[0])
        return self.back(request, ok=f"deleted {target!r}", path=target.rpartition("/")[0])

    @route("/download")
    async def download(self, request: Request) -> Response:
        target = str(request.query_params.get("path") or "")
        try:
            content = self.services.storage.get_bytes(target)
        except ApiError as e:
            return self.back(request, error=e.message)
        name = target.rstrip("/").rpartition("/")[2] or "download.bin"
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @route("/preview")
    async def preview(self, request: Request) -> Response:
        target = str(request.query_params.get("path") or "")
        thumb = request.query_params.get("thumb") == "1"
        try:
            content = self.services.storage.get_bytes(target)
        except ApiError as e:
            return Response(content=e.message, media_type="text/plain", status_code=404)
        media = _media_type(target)
        if thumb and media != "application/octet-stream":
            small = _thumbnail(content)
            if small is not None:
                content, media = small, "image/png"
        return Response(content=content, media_type=media, headers={"Cache-Control": "private, max-age=60"})

    # endregion

    # region body
    def body(self, request: Request) -> str:
        csrf = self.csrf(request)
        path = str(request.query_params.get("path") or "/")
        storage = self.services.storage
        parent = path.rstrip("/").rpartition("/")[0] or "/"
        directories = storage.list_dirs(path)
        files: list[tuple[str, str, int]] = []
        for full in storage.glob_files(path):
            folder, _, name = full.rpartition("/")
            if folder != path.rstrip("/"):
                continue
            files.append((name, full, storage.file_info(full).size))

        crumbs = [f'<a href="{self.url(request)}">/</a>']
        walked = ""
        for part in [p for p in path.strip("/").split("/") if p]:
            walked += "/" + part
            crumbs.append(f'<a href="{self.url(request, path=walked)}">{_e(part)}</a>')

        rows = []
        for name in directories:
            full = f"{path.rstrip('/')}/{name}"
            rows.append(
                "<tr>"
                f'<td><a href="{self.url(request, path=full)}">📁 {_e(name)}</a></td>'
                "<td></td><td></td></tr>"
            )
        for name, full, size in files:
            thumb = (
                f'<img src="{self.url(request, "/preview", path=full, thumb=1)}" '
                'class="rounded border" style="height:36px" loading="lazy" alt="">'
                if _is_image(name)
                else ""
            )
            preview = (
                f'<a class="btn btn-sm btn-outline-secondary" target="_blank" '
                f'href="{self.url(request, "/preview", path=full)}">Preview</a>'
                if _is_image(name)
                else ""
            )
            rows.append(
                "<tr>"
                f"<td>{thumb} {_e(name)}</td>"
                f"<td>{_fmt_bytes(size)}</td>"
                "<td class='d-flex gap-1'>"
                f"{preview}"
                f'<a class="btn btn-sm btn-outline-secondary" href="{self.url(request, "/download", path=full)}">Download</a>'
                f'<form method="post" action="{self.url(request, "/delete")}" '
                f"onsubmit=\"return confirm('Delete {_e(name)}?')\">{csrf}"
                f"<input type='hidden' name='path' value='{_e(full)}'>"
                "<button class='btn btn-sm btn-outline-danger' type='submit'>Delete</button></form>"
                "</td></tr>"
            )
        listing = (
            '<p class="mb-2">'
            + (f'<a href="{self.url(request, path=parent)}">↑ up</a> · ' if path not in ("", "/") else "")
            + " / ".join(crumbs)
            + "</p>"
            + _table(["name", "size", ""], rows, "empty directory")
        )
        upload = (
            f'<form method="post" action="{self.url(request, "/upload")}" enctype="multipart/form-data" '
            f'class="vstack gap-2">{csrf}'
            f"<input type='hidden' name='path' value='{_e(path)}'>"
            '<div><label class="form-label">file</label>'
            '<input type="file" name="file" class="form-control" required></div>'
            '<button class="btn btn-primary" type="submit">Upload</button></form>'
        )
        mkdir = (
            f'<form method="post" action="{self.url(request, "/mkdir")}" class="vstack gap-2">{csrf}'
            f"<input type='hidden' name='path' value='{_e(path)}'>"
            '<div><label class="form-label">new sub-directory</label>'
            '<input name="name" class="form-control" required></div>'
            '<button class="btn btn-outline-primary" type="submit">Create directory</button></form>'
        )
        return (
            '<div class="row row-cards">'
            f'<div class="col-12">{_card("Files", listing)}</div>'
            f'<div class="col-lg-6">{_card("Upload", upload)}</div>'
            f'<div class="col-lg-6">{_card("New directory", mkdir)}</div>'
            "</div>"
        )

    # endregion


# endregion


# region audit log
class AuditLogAdmin(ModelView):
    """Read-only audit trail: claim/submit/review/save/label/role actions.

    The actor is rendered as a plain string: the model's ``user`` relationship
    points at the ``users`` table, which no longer has a starlette-admin view of
    its own (Users is a write page now), and starlette-admin refuses relation
    fields whose target view is missing.
    """

    fields = (
        AuditLog.id,
        AuditLog.ts,
        StringField("actor", label="user", read_only=True),
        AuditLog.action,
        AuditLog.target_type,
        AuditLog.target_id,
        AuditLog.detail_json,
    )
    sortable_fields = (AuditLog.id, AuditLog.ts, AuditLog.action)
    searchable_fields = (AuditLog.action, AuditLog.target_id)
    page_size = 100

    async def serialize(self, obj, request, **kwargs):
        data = await super().serialize(obj, request, **kwargs)
        data["actor"] = obj.user.name if obj.user is not None else ""
        return data

    def can_create(self, request: Request) -> bool:
        return False

    def can_edit(self, request: Request) -> bool:
        return False

    def can_delete(self, request: Request) -> bool:
        return False

    label = "Audit"
    icon = "fa-solid fa-clipboard-list"


# endregion
