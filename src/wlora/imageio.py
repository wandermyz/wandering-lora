"""Image loading, HEIC conversion and inference-time downscaling."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CONVERT_EXTS = {".heic", ".heif", ".avif"}


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
