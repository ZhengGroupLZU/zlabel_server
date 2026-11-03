"""Label colours: auto-created labels get an unused colour from the palette."""

from __future__ import annotations

from app.services.label_palette import LABEL_PALETTE, normalize_color, pick_color


def test_pick_color_stays_in_the_palette_and_avoids_used_colours():
    used: list[str] = []
    for _ in range(len(LABEL_PALETTE)):
        color = pick_color(used)
        assert color in LABEL_PALETTE
        assert color not in used
        used.append(color)
    # every colour is taken: it still has to return a palette entry
    assert pick_color(used) in LABEL_PALETTE
    # case does not matter when checking what is used
    assert pick_color([color.upper() for color in used]) in LABEL_PALETTE


def test_normalize_color():
    assert normalize_color("#AABBCC") == "#aabbcc"
    assert normalize_color(" #112233 ") == "#112233"
    # the shorthand and a missing '#' are accepted (typed by hand)
    assert normalize_color("#abc") == "#aabbcc"
    assert normalize_color("abc") == "#aabbcc"
    assert normalize_color("336699") == "#336699"
    assert normalize_color("red") == ""
    assert normalize_color("#abcd") == ""
    assert normalize_color("") == "" and normalize_color(None) == ""


def test_new_labels_append_and_the_ordinal_id_follows_the_order(client, auth_headers):
    """The displayed id is the 0-based position; new labels are appended."""
    headers = auth_headers(client)
    client.post("/api/v2/projects", json={"name": "projA"}, headers=headers)
    for name in ("Seed", "Root", "Shoot"):
        resp = client.post("/api/v2/projects/projA/labels", json={"name": name}, headers=headers)
        assert resp.status_code == 201, resp.text

    labels = client.get("/api/v2/projects/projA/labels", headers=headers).json()
    assert [(label["name"], label["sort"]) for label in labels] == [
        ("Seed", 0),
        ("Root", 1),
        ("Shoot", 2),
    ]


def test_reorder_rewrites_sort_but_keeps_the_label_ids(client, auth_headers):
    headers = auth_headers(client)
    client.post("/api/v2/projects", json={"name": "projA"}, headers=headers)
    for name in ("Seed", "Root", "Shoot"):
        client.post("/api/v2/projects/projA/labels", json={"name": name}, headers=headers)
    before = client.get("/api/v2/projects/projA/labels", headers=headers).json()
    ids = {label["name"]: label["id"] for label in before}

    resp = client.put(
        "/api/v2/projects/projA/labels/order",
        json={"order": [ids["Root"], ids["Seed"], ids["Shoot"]]},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert [(label["name"], label["sort"], label["id"]) for label in resp.json()] == [
        ("Root", 0, ids["Root"]),
        ("Seed", 1, ids["Seed"]),
        ("Shoot", 2, ids["Shoot"]),
    ]


def test_reorder_rejects_an_incomplete_order(client, auth_headers):
    headers = auth_headers(client)
    client.post("/api/v2/projects", json={"name": "projA"}, headers=headers)
    for name in ("Seed", "Root"):
        client.post("/api/v2/projects/projA/labels", json={"name": name}, headers=headers)
    labels = client.get("/api/v2/projects/projA/labels", headers=headers).json()

    resp = client.put(
        "/api/v2/projects/projA/labels/order", json={"order": [labels[0]["id"]]}, headers=headers
    )
    assert resp.status_code == 422 and resp.json()["code"] == "validation_error"


def test_api_assigns_unused_palette_colours(client, auth_headers):
    headers = auth_headers(client)
    assert client.post("/api/v2/projects", json={"name": "projA"}, headers=headers).status_code == 201

    colors = []
    for name in ("Root", "Shoot", "Seed"):
        resp = client.post("/api/v2/projects/projA/labels", json={"name": name}, headers=headers)
        assert resp.status_code == 201, resp.text
        colors.append(resp.json()["color"])
    assert len(set(colors)) == 3 and all(color in LABEL_PALETTE for color in colors)

    # an explicit colour is kept as sent (and normalised)
    explicit = client.post(
        "/api/v2/projects/projA/labels", json={"name": "Dish", "color": "#AABBCC"}, headers=headers
    )
    assert explicit.status_code == 201 and explicit.json()["color"] == "#aabbcc"
    # shorthand / bare hex codes are normalised too
    shorthand = client.post(
        "/api/v2/projects/projA/labels", json={"name": "Shorthand", "color": "#abc"}, headers=headers
    )
    assert shorthand.json()["color"] == "#aabbcc"
    bare = client.post(
        "/api/v2/projects/projA/labels", json={"name": "Bare", "color": "336699"}, headers=headers
    )
    assert bare.json()["color"] == "#336699"
    # junk falls back to an unused palette colour instead of being stored
    junk = client.post(
        "/api/v2/projects/projA/labels", json={"name": "Junk", "color": "red"}, headers=headers
    )
    assert junk.json()["color"] in LABEL_PALETTE
    # a colour outside the palette is allowed too, as long as it is valid hex
    custom = client.post(
        "/api/v2/projects/projA/labels", json={"name": "Timestamp", "color": "#123456"}, headers=headers
    )
    assert custom.json()["color"] == "#123456"


def test_labels_introduced_by_an_annotation_get_palette_colours(client, auth_headers, harness):
    from tests.app.test_projects import seed

    seed(harness, "projA", files=("images/a/D1.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    anno_id = client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"][0]["anno_id"]

    saved = client.put(
        f"/api/v2/projects/projA/annotations/{anno_id}",
        json={"results": {"r1": {"labels": [{"name": "Root"}, {"name": "Shoot"}]}}},
        headers=headers,
    )
    assert saved.status_code == 200, saved.text

    labels = client.get("/api/v2/projects/projA/labels", headers=headers).json()
    colors = {label["name"]: label["color"] for label in labels}
    assert set(colors) == {"Root", "Shoot"}
    assert all(color in LABEL_PALETTE for color in colors.values())
    assert len(set(colors.values())) == 2  # two new labels, two different colours


# region instance-type mapping (docs/export_spec.md §1.1)
def test_spec_named_labels_are_seeded_with_the_spec_table(client, auth_headers):
    headers = auth_headers(client)
    client.post("/api/v2/projects", json={"name": "projA"}, headers=headers)
    resp = client.post("/api/v2/projects/projA/labels", json={"name": "Root"}, headers=headers)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["default_instance_type"] == "normal_seedling"
    assert body["allowed_instance_types"] == ["normal_seedling", "abnormal_seedling", "moldy_seed"]


def test_project_specific_labels_stay_unconfigured(client, auth_headers):
    headers = auth_headers(client)
    client.post("/api/v2/projects", json={"name": "projA"}, headers=headers)
    body = client.post("/api/v2/projects/projA/labels", json={"name": "Needle"}, headers=headers).json()
    # None = unconfigured (the client keeps its spec-table fallback), not []
    assert body["default_instance_type"] == ""
    assert body["allowed_instance_types"] is None


def test_explicit_mapping_wins_and_unknown_types_are_dropped(client, auth_headers):
    headers = auth_headers(client)
    client.post("/api/v2/projects", json={"name": "projA"}, headers=headers)
    resp = client.post(
        "/api/v2/projects/projA/labels",
        json={
            "name": "Seed",
            "default_instance_type": "dead_seed",
            "allowed_instance_types": ["dead_seed", "nonsense"],
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["default_instance_type"] == "dead_seed"
    assert resp.json()["allowed_instance_types"] == ["dead_seed"]


def test_patch_distinguishes_omitted_null_and_empty(client, auth_headers):
    headers = auth_headers(client)
    client.post("/api/v2/projects", json={"name": "projA"}, headers=headers)
    label = client.post("/api/v2/projects/projA/labels", json={"name": "Seed"}, headers=headers).json()
    url = f"/api/v2/projects/projA/labels/{label['id']}"

    # an omitted key keeps the mapping
    kept = client.patch(url, json={"name": "Seed"}, headers=headers).json()
    assert kept["allowed_instance_types"] == ["normal_seed", "moldy_seed", "dead_seed"]

    # an explicit empty list = "any type is allowed"
    empty = client.patch(url, json={"allowed_instance_types": []}, headers=headers).json()
    assert empty["allowed_instance_types"] == []

    # an explicit null = back to unconfigured (client spec fallback)
    reset = client.patch(url, json={"allowed_instance_types": None}, headers=headers).json()
    assert reset["allowed_instance_types"] is None

    # the default type lives in its own field ("": cleared)
    cleared = client.patch(url, json={"default_instance_type": ""}, headers=headers).json()
    assert cleared["default_instance_type"] == ""


def test_labels_introduced_by_saves_get_the_spec_mapping(client, auth_headers, harness):
    from tests.app.test_projects import seed

    seed(harness, "projA", files=("images/a/D1.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    anno_id = client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"][0]["anno_id"]
    client.put(
        f"/api/v2/projects/projA/annotations/{anno_id}",
        json={"results": {"r1": {"labels": [{"name": "Dish"}, {"name": "Needle"}]}}},
        headers=headers,
    )
    labels = {lbl["name"]: lbl for lbl in client.get("/api/v2/projects/projA/labels", headers=headers).json()}
    assert labels["Dish"]["default_instance_type"] == "dish"
    assert labels["Dish"]["allowed_instance_types"] == ["dish"]
    assert labels["Needle"]["allowed_instance_types"] is None


# endregion
