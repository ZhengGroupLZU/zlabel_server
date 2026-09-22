"""Audit trail: every state-changing action writes one row.

Kept deliberately dumb (append-only, no foreign keys to tasks) so it survives
task deletion and is cheap to query by action/target/time.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from v2.db.models import AuditLog


def record(
    session: Session,
    *,
    action: str,
    user_id: int | None = None,
    target_type: str = "",
    target_id: str = "",
    detail: dict[str, Any] | None = None,
) -> AuditLog:
    entry = AuditLog(
        user_id=user_id,
        action=action,
        target_type=target_type,
        target_id=str(target_id),
        detail_json=json.dumps(detail or {}, ensure_ascii=False, default=str),
    )
    session.add(entry)
    return entry
