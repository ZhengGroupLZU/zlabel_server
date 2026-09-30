"""``?max_side=`` downscaling: the pure renderer and its on-disk variant cache."""

from __future__ import annotations

import io
import json

from PIL import Image

from app.core.config import Settings
from app.services import image_scaler
from app.services.image_scaler import ImageScaler, render_variant


def _png(size: tuple[int, int] = (1024, 512), color=(10, 200, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg(size: tuple[int, int] = (3000, 1500), orientation: int = 6) -> bytes:
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation  # Orientation
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format="JPEG", quality=95, exif=exif)
    return buffer.getvalue()


def _size(content: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(content)) as image:
        return image.size


# region render_variant
def test_render_downscales_the_long_edge_and_reports_the_original():
    variant = render_variant(_png((1024, 512)), 256)
    assert variant is not None
    assert _size(variant.content) == (256, 128)  # aspect ratio preserved
    assert variant.original_size == (1024, 512)  # the *stored* size, not the served one
    assert variant.media_type == "image/png"


def test_render_keeps_the_jpeg_container_and_its_exif():
    variant = render_variant(_jpeg(), 1280)
    assert variant is not None and variant.media_type == "image/jpeg"
    assert variant.original_size == (3000, 1500)
    with Image.open(io.BytesIO(variant.content)) as image:
        assert image.size == (1280, 640)
        # orientation travels with the variant: the server must not rotate, because
        # every client shows the stored pixel order
        assert image.getexif().get(274) == 6


def test_render_never_upscales():
    assert render_variant(_png((256, 128)), 256) is None
    assert render_variant(_png((256, 128)), 4096) is None


def test_render_falls_back_on_an_undecodable_file():
    assert render_variant(b"not an image at all", 128) is None


def test_render_reports_the_stored_size_when_the_ratio_is_not_exact():
    variant = render_variant(_png((1000, 333)), 500)
    assert variant is not None
    assert variant.original_size == (1000, 333)
    assert _size(variant.content) == (500, round(333 * 0.5))


def test_render_keeps_the_original_when_re_encoding_grows_it(monkeypatch):
    """A tiny, heavily compressed image can grow when re-encoded: keep the stored bytes."""
    content = _jpeg((600, 300))
    original_save = Image.Image.save

    def _huge_save(self, fp, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        original_save(self, fp, *args, **kwargs)
        fp.write(b"\x00" * len(content))

    monkeypatch.setattr(Image.Image, "save", _huge_save)
    assert render_variant(content, 500) is None


def test_render_handles_a_degenerate_target():
    variant = render_variant(_png((1024, 512)), 1)
    assert variant is not None
    assert min(_size(variant.content)) >= 1


# endregion


# region clamp
def test_clamp_maps_zero_to_the_stored_file(settings: Settings):
    scaler = ImageScaler(settings)
    assert scaler.clamp(0) == 0
    assert scaler.clamp(-5) == 0


def test_clamp_honours_the_server_ceiling(settings: Settings):
    settings.image_max_side_limit = 512
    scaler = ImageScaler(settings)
    assert scaler.clamp(1280) == 512
    assert scaler.clamp(200) == 200


def test_clamp_without_a_ceiling_passes_the_request_through(settings: Settings):
    settings.image_max_side_limit = 0
    scaler = ImageScaler(settings)
    assert scaler.clamp(16384) == 16384


# endregion


# region variant cache
def test_variant_is_cached_per_etag_and_size(settings: Settings, monkeypatch):
    scaler = ImageScaler(settings)
    content = _png((1024, 512))
    calls: list[int] = []
    real = image_scaler.render_variant

    def _counting(c: bytes, max_side: int, **kwargs):
        calls.append(max_side)
        return real(c, max_side, **kwargs)

    monkeypatch.setattr(image_scaler, "render_variant", _counting)

    first = scaler.variant(content, max_side=256, etag='"1-a"')
    second = scaler.variant(content, max_side=256, etag='"1-a"')
    assert calls == [256]  # the second call read the cache
    assert first == second and first is not None

    scaler.variant(content, max_side=256, etag='"2-b"')  # a new revision -> re-render
    scaler.variant(content, max_side=128, etag='"1-a"')  # a new size -> re-render
    assert calls == [256, 256, 128]


def test_variant_cache_survives_a_new_scaler_instance(settings: Settings):
    content = _png((1024, 512))
    first = ImageScaler(settings).variant(content, max_side=256, etag='"1-a"')
    # a fresh instance over the same directory still finds it: a restart keeps the work
    second = ImageScaler(settings).variant(content, max_side=256, etag='"1-a"')
    assert first == second and first is not None


def test_variant_cache_entry_is_dropped_when_unreadable(settings: Settings):
    scaler = ImageScaler(settings)
    content = _png((1024, 512))
    assert scaler.variant(content, max_side=256, etag='"1-a"') is not None
    data_path, meta_path = scaler._paths(scaler._key('"1-a"', 256))

    meta_path.write_text(json.dumps({"media_type": "image/png"}), encoding="utf-8")  # no size
    assert scaler.variant(content, max_side=256, etag='"1-a"') is not None  # rebuilt, no error

    data_path.write_bytes(b"truncated")
    meta_path.unlink()
    assert scaler.variant(content, max_side=256, etag='"1-a"') is not None


def test_variant_cache_write_failure_is_not_fatal(settings: Settings, monkeypatch):
    """A read-only cache directory must still serve the variant."""
    scaler = ImageScaler(settings)
    content = _png((1024, 512))

    def _boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.write_bytes", _boom)
    variant = scaler.variant(content, max_side=256, etag='"1-a"')
    assert variant is not None and _size(variant.content) == (256, 128)


# endregion
