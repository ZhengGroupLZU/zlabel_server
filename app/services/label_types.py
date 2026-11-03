"""The seed-germination spec's ``instance_type`` x ``part`` table (``docs/export_spec.md`` §1.1).

A label (part) may only appear inside the instance types listed for it, and it
carries a *default* type that a freshly created instance starts with. Both facts
live on the label row itself (``labels.default_instance_type`` /
``labels.allowed_instance_types``), so this module is only

- the **spec default** the server seeds a spec-named label with (``seed_for``), and
- the vocabulary check that keeps a mapping from naming a type nobody offers.

The client mirrors the same table in ``zlabel/utils/instance_types.py``; it is the
fallback when a label has ``allowed_instance_types = NULL`` (unconfigured). Keep
the two tables in sync — they are transcribed from the spec document, not derived.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelTypes:
    """Spec defaults of one part label."""

    #: the instance type a new instance of this label starts with ("" = none)
    default: str = ""
    #: instance types this label may appear in (empty = any)
    allowed: tuple[str, ...] = ()


#: Part labels of the spec, in the order the preset writes them.
SPEC_LABEL_TYPES: dict[str, LabelTypes] = {
    "Seed": LabelTypes("normal_seed", ("normal_seed", "moldy_seed", "dead_seed")),
    "Root": LabelTypes("normal_seedling", ("normal_seedling", "abnormal_seedling", "moldy_seed")),
    "Shoot": LabelTypes("normal_seedling", ("normal_seedling", "abnormal_seedling")),
    # the emptied testa: appears on an intact seed, and as the remnant of a seedling
    "Seed Coat": LabelTypes("normal_seed", ("normal_seed", "normal_seedling", "abnormal_seedling")),
    "Dish": LabelTypes("dish", ("dish",)),
    "Timestamp": LabelTypes("text", ("text",)),
    "Number": LabelTypes("text", ("text",)),
}

#: Labels of the spec's ``Text`` instance type (rectangles carrying recognised text).
TEXT_PART_LABELS: tuple[str, ...] = ("timestamp", "number", "时间戳", "编号")


def spec_for(name: str) -> LabelTypes | None:
    """The spec defaults of a part label (``None`` for a project-specific label)."""
    return SPEC_LABEL_TYPES.get(str(name).strip())


def is_text_part_label(name: str) -> bool:
    return str(name).strip().lower() in TEXT_PART_LABELS


def filter_known_types(allowed: list[str] | None, vocabulary: list[str]) -> list[str] | None:
    """Drop types the project does not offer; ``None`` (unconfigured) stays ``None``.

    The mapping is metadata: a stale/typo'd type is silently dropped instead of
    rejecting the whole save, and the caller reports what happened in the log.
    """
    if allowed is None:
        return None
    known = [str(v) for v in vocabulary]
    return [t for t in dict.fromkeys(str(a) for a in allowed) if t in known]
