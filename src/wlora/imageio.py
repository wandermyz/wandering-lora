"""Image loading, HEIC conversion and inference-time downscaling."""
from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CONVERT_EXTS = {".heic", ".heif", ".avif"}
# Camera raw formats LibRaw can read; converted to JPEG on ingest.
RAW_EXTS = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def is_image(p: Path) -> bool:
    return p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_for_inference(p: Path, max_pixels: int) -> Image.Image:
    """Open, apply EXIF orientation, convert to RGB and downscale (aspect kept)."""
    with Image.open(p) as im:
        im.draft("RGB", (4096, 4096))  # cheap JPEG DCT downscale for huge photos
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    w, h = im.size
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
    return im


def convert_to_jpeg(src: Path, dst: Path, quality: int) -> None:
    """Convert (e.g. HEIC) to JPEG, baking in orientation and keeping EXIF/ICC."""
    with Image.open(src) as im:
        exif = im.getexif()
        icc = im.info.get("icc_profile")
        out = ImageOps.exif_transpose(im).convert("RGB")
    if exif:
        exif[0x0112] = 1  # orientation already applied
    tmp = dst.with_name(f".{dst.name}.tmp")
    kwargs = {"quality": quality, "subsampling": 0, "exif": exif.tobytes() if exif else b""}
    if icc:
        kwargs["icc_profile"] = icc
    out.save(tmp, "JPEG", **kwargs)
    tmp.replace(dst)


def _write_atomic_bytes(dst: Path, data: bytes) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(dst)


# LibRaw `flip` -> PIL transpose (3: 180°, 5: 90° CCW, 6: 90° CW)
_FLIP = {3: Image.Transpose.ROTATE_180, 5: Image.Transpose.ROTATE_90, 6: Image.Transpose.ROTATE_270}


def convert_raw_to_jpeg(src: Path, dst: Path, quality: int, source: str = "embedded",
                        min_embedded_pixels: int = 4_000_000) -> str:
    """Convert a camera raw to JPEG. Returns the method used.

    `embedded`: use the camera-rendered full-size JPEG preview stored in the raw
    (camera colours/picture style, no re-encode when no rotation is needed);
    falls back to `develop` when the preview is missing or small.
    `develop`: demosaic with LibRaw (camera white balance, sRGB, 8 bit).
    """
    import rawpy

    with rawpy.imread(str(src)) as raw:
        flip = raw.sizes.flip
        if source == "embedded":
            try:
                thumb = raw.extract_thumb()
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                thumb = None
            if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
                with Image.open(io.BytesIO(thumb.data)) as im:
                    big_enough = im.width * im.height >= min_embedded_pixels
                    if big_enough and flip not in _FLIP:
                        _write_atomic_bytes(dst, thumb.data)  # byte-exact, no recompression
                        return "embedded"
                    if big_enough:
                        exif = im.getexif()
                        exif[0x0112] = 1
                        out = im.transpose(_FLIP[flip]).convert("RGB")
                        tmp = dst.with_name(f".{dst.name}.tmp")
                        out.save(tmp, "JPEG", quality=quality, subsampling=0, exif=exif.tobytes())
                        tmp.replace(dst)
                        return "embedded+rotated"
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=False)
    out = Image.fromarray(rgb)  # LibRaw already applied `flip`
    tmp = dst.with_name(f".{dst.name}.tmp")
    out.save(tmp, "JPEG", quality=quality, subsampling=0)
    tmp.replace(dst)
    return "developed"
