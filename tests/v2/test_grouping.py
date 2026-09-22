"""Grouping must stay in sync with what the desktop client used to guess."""

from __future__ import annotations

import pytest

from v2.services.grouping import parse_group


@pytest.mark.parametrize(
    ("rel_path", "expected"),
    [
        ("species/dish01/D1.png", ("species/dish01", 1)),
        ("species/dish01/D12.jpg", ("species/dish01", 12)),
        ("dish01/D3.png", ("dish01", 3)),
        ("images/2020-z-1004-1/D7.png", ("images/2020-z-1004-1", 7)),
        ("dishA_001.jpg", ("dishA", 1)),
        ("dishA-002.png", ("dishA", 2)),
        ("abc_12_3.png", ("abc_12", 3)),
        ("plain.png", ("", 0)),
        (r"species\dish01\D4.png", ("species/dish01", 4)),
        # a bare D-file at the project root: the tail regex yields group "D"
        # (identical to the client's old fallback)
        ("D1.png", ("D", 1)),
    ],
)
def test_parse_group(rel_path, expected):
    assert parse_group(rel_path) == expected
