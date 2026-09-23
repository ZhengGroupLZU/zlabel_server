/* Dashboard status cards: poll /admin/status and update the two cards in place.

   The cards carry their endpoint and refresh interval as data attributes so
   the script never hardcodes the admin path (ZLSERVER_ADMIN_PATH).
*/
(() => {
    "use strict";

    const cards = Array.from(document.querySelectorAll(".zlabel-status"));
    if (cards.length === 0) {
        return;
    }

    const url = cards[0].dataset.statusUrl;
    const refreshMs = parseInt(cards[0].dataset.refreshMs || "30000", 10);
    const bodies = new Map(
        cards.map((card) => [card.dataset.statusKind, card.querySelector(".zlabel-status-body")]),
    );
    const footers = new Map(
        cards.map((card) => [card.dataset.statusKind, card.querySelector(".zlabel-status-updated")]),
    );

    let lastData = null;
    let timer = null;
    let stopped = false;

    function setText(parent, text) {
        parent.replaceChildren(document.createTextNode(text));
    }

    function badge(text, color) {
        const span = document.createElement("span");
        span.className = `badge ${color} text-white`;
        span.textContent = text;
        return span;
    }

    function addRow(container, label, value, valueNode, title) {
        const row = document.createElement("div");
        row.className = "d-flex justify-content-between align-items-center py-1";
        const labelEl = document.createElement("span");
        labelEl.className = "text-muted";
        labelEl.textContent = label;
        const valueEl = document.createElement("span");
        let node = valueNode;
        if (!node || (node.nodeType === Node.TEXT_NODE && node.textContent === "")) {
            node = document.createTextNode(value == null || value === "" ? "—" : String(value));
        }
        valueEl.appendChild(node);
        if (title) {
            valueEl.title = title;
        }
        row.append(labelEl, valueEl);
        container.appendChild(row);
    }

    function textValue(text) {
        return document.createTextNode(text == null || text === "" ? "—" : String(text));
    }

    function statusBadge(status) {
        if (status === "ok") return badge(status, "bg-success");
        if (status === "error") return badge(status, "bg-danger");
        if (status === "degraded" || status === "unavailable") return badge(status, "bg-warning");
        return badge(status, "bg-secondary");
    }

    function formatUptime(seconds) {
        if (seconds == null) return "—";
        const s = Math.max(0, Math.floor(seconds));
        const days = Math.floor(s / 86400);
        const hours = Math.floor((s % 86400) / 3600);
        const minutes = Math.floor((s % 3600) / 60);
        if (days) return `${days}d ${hours}h`;
        if (hours) return `${hours}h ${minutes}m`;
        if (minutes) return `${minutes}m ${s % 60}s`;
        return `${s}s`;
    }

    function formatTime(iso) {
        if (!iso) return "";
        const date = new Date(iso);
        return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString();
    }

    function renderServer(body, footer, data) {
        const server = data.server || {};
        const checks = server.checks || {};
        const db = checks.db || {};
        const storage = checks.storage || {};
        body.replaceChildren();
        addRow(body, "status", server.status || "unknown", statusBadge(server.status || "unknown"));
        addRow(body, "version", `${server.name || "ZLabel"} v${server.version || "?"}`, textValue(""));
        addRow(
            body,
            "db",
            db.status || "unknown",
            statusBadge(db.status || "unknown"),
            db.message || "",
        );
        addRow(
            body,
            "storage",
            storage.status || "unknown",
            statusBadge(storage.status || "unknown"),
            storage.message || storage.root || "",
        );
        addRow(
            body,
            "api uptime",
            formatUptime(server.uptime_s),
            textValue(formatUptime(server.uptime_s)),
            server.started_at ? `up since ${new Date(server.started_at).toLocaleString()}` : "",
        );
        footer.textContent = "";
    }

    function renderInference(body, footer, data) {
        const inf = data.inference || {};
        const health = inf.health || {};
        const metrics = inf.metrics || {};
        const queue = health.queue || {};
        const cache = health.cache || {};
        const latency = metrics.latency_ms || {};
        body.replaceChildren();
        addRow(body, "status", inf.status || "unknown", statusBadge(inf.status || "unknown"), inf.message || "");
        if (inf.configured) {
            addRow(body, "model", health.model || "—", textValue(health.model));
            addRow(body, "backend", health.backend || "—", textValue(health.backend));
            addRow(
                body,
                "loaded",
                health.loaded ? "yes" : "no",
                badge(health.loaded ? "loaded" : "not loaded", health.loaded ? "bg-success" : "bg-danger"),
            );
            const hitRate =
                metrics.cache_hit_rate != null
                    ? `${Math.round(metrics.cache_hit_rate * 100)}%`
                    : cache.hit_rate != null
                      ? `${Math.round(cache.hit_rate * 100)}%`
                      : "—";
            addRow(body, "cache", `${cache.entries ?? metrics.cache_hits ?? "—"} entries · hit ${hitRate}`, textValue(""));
            addRow(body, "queue", `waiting ${queue.waiting ?? "—"} / ${queue.concurrency ?? "—"}`, textValue(""));
            addRow(body, "jobs", health.jobs ?? metrics.jobs ?? "—", textValue(health.jobs ?? metrics.jobs));
            addRow(
                body,
                "latency",
                latency.p50 != null ? `p50 ${latency.p50} ms · p95 ${latency.p95 ?? "—"} ms` : "—",
                textValue(""),
            );
            addRow(body, "worker uptime", formatUptime(metrics.uptime_s), textValue(formatUptime(metrics.uptime_s)));
            if (inf.metrics_error) {
                addRow(body, "metrics", "n/a", badge("n/a", "bg-secondary"), inf.metrics_error);
            }
        } else {
            addRow(body, "worker", "not configured", textValue(""), "set ZLSERVER_INFERENCE_URL to enable predictions");
        }
        footer.textContent = "";
    }

    function render(data) {
        lastData = data;
        renderServer(bodies.get("server"), footers.get("server"), data);
        renderInference(bodies.get("inference"), footers.get("inference"), data);
        const checkedAt = formatTime(data.checked_at);
        const duration = data.duration_ms != null ? `${data.duration_ms} ms` : "";
        for (const footer of footers.values()) {
            setText(footer, checkedAt ? `updated ${checkedAt} · ${duration}` : `updated · ${duration}`);
        }
    }

    function fail(message) {
        for (const [kind, body] of bodies) {
            if (!lastData) {
                body.replaceChildren();
                const p = document.createElement("div");
                p.className = "text-danger";
                p.textContent = "status unavailable";
                body.appendChild(p);
            }
            const footer = footers.get(kind);
            if (footer) setText(footer, message);
        }
    }

    function sessionExpired() {
        if (stopped) return;
        stopped = true;
        if (timer) window.clearInterval(timer);
        for (const body of bodies.values()) {
            body.replaceChildren();
            const p = document.createElement("div");
            p.className = "text-warning";
            p.textContent = "session expired — reload";
            body.appendChild(p);
        }
        for (const footer of footers.values()) setText(footer, "stopped");
    }

    function refresh() {
        if (stopped) return;
        fetch(url, { headers: { Accept: "application/json" }, cache: "no-store" })
            .then((response) => {
                if (response.status === 401) {
                    sessionExpired();
                    return null;
                }
                if (!response.ok) {
                    fail(`update failed ${new Date().toLocaleTimeString()} (HTTP ${response.status})`);
                    return null;
                }
                return response.json();
            })
            .then((data) => {
                if (data) render(data);
            })
            .catch(() => {
                fail(`update failed ${new Date().toLocaleTimeString()} (network)`);
            });
    }

    refresh();
    timer = window.setInterval(refresh, Math.max(5000, refreshMs));
})();
