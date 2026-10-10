"""Caption sidecars: <stem>.txt (caption) and <stem>.caption.json (everything else).

See docs/sidecar-format.md for the contract shared with wandering-gallery.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "wandering-lora/caption@1"
STATUSES = ("generated", "edited", "approved", "needs_regen")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write(path: Path, data: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


@dataclass
class Sidecar:
    image: Path

    @property
    def txt_path(self) -> Path:
        return self.image.with_suffix(".txt")

    @property
    def json_path(self) -> Path:
        return self.image.with_name(self.image.stem + ".caption.json")

    def read_txt_raw(self) -> str | None:
        try:
            with open(self.txt_path, encoding="utf-8", newline="") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def read_txt(self) -> str | None:
        raw = self.read_txt_raw()
        return raw.strip() if raw is not None else None

    def read_json(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.json_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def state(self) -> tuple[str, dict | None, str | None]:
        """Return (effective_status, json, txt). Status 'none' if never processed.

        Must agree with wandering-gallery's effectiveStatus(): the hash covers the raw
        .txt bytes, and unknown statuses count as human edits.
        """
        data, raw = self.read_json(), self.read_txt_raw()
        txt = raw.strip() if raw is not None else None
        if data is None:
            return ("edited" if raw is not None else "none"), None, txt
        status = data.get("status", "generated")
        if status not in STATUSES:
            status = "edited"
        if status in ("generated", "approved") and raw is not None and sha256_text(raw) != data.get("caption_sha256"):
            status = "edited"  # .txt changed behind our back: a human edit, no longer the approved text
        if status == "generated" and txt is None:
            status = "missing_txt"
        return status, data, txt

    def write(self, data: dict, caption: str | None) -> None:
        """Write .txt (if given) then json. Caller must hold the decision logic."""
        if caption is not None:
            atomic_write(self.txt_path, caption)
            data["caption"] = caption
            data["caption_sha256"] = sha256_text(caption)
        data["schema"] = SCHEMA
        data["updated_at"] = now_iso()
        atomic_write(self.json_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
