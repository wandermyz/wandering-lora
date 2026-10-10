"""Ingest, batch captioning, status and export."""
from __future__ import annotations

import fnmatch
import json
import logging
import shutil
import time
import traceback
from pathlib import Path

from . import __version__
from .config import Dataset
from .imageio import (CONVERT_EXTS, IMAGE_EXTS, RAW_EXTS, convert_raw_to_jpeg, convert_to_jpeg, is_image,
                      load_for_inference, sha256_file)
from .models import get_backend
from .normalize import Vocabulary, coerce_classification, normalize
from .prompt import build_prompt, extract_json
from .sidecar import Sidecar, now_iso, sha256_text

log = logging.getLogger("wlora")


def setup_logging(ds: Dataset) -> Path:
    ds.state_dir.mkdir(parents=True, exist_ok=True)
    path = ds.state_dir / "wlora.log"
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger("wlora")
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    return path


def list_images(ds: Dataset) -> list[Path]:
    return sorted(p for p in ds.images_dir.iterdir() if is_image(p)) if ds.images_dir.exists() else []


# ---------------------------------------------------------------- ingest

def ingest(ds: Dataset, src: Path, mode: str | None, dry_run: bool) -> None:
    mode = mode or ds.cfg["ingest"]["mode"]
    quality = int(ds.cfg["ingest"]["jpeg_quality"])
    raw_cfg = ds.cfg["ingest"].get("raw", {})
    ds.images_dir.mkdir(parents=True, exist_ok=True)
    taken = {p.stem.lower() for p in ds.images_dir.iterdir() if p.is_file()}
    files = sorted(p for p in src.iterdir() if p.is_file() and not p.name.startswith("."))
    n = 0
    for p in files:
        ext = p.suffix.lower()
        if ext not in IMAGE_EXTS | CONVERT_EXTS | RAW_EXTS:
            print(f"skip (unsupported) {p.name}")
            continue
        if time.time() - p.stat().st_mtime < 30:
            print(f"skip (modified <30s ago, may still be copying) {p.name}")
            continue
        stem = p.stem
        if stem.endswith(".caption"):
            stem = stem[: -len(".caption")]
        i = 1
        while stem.lower() in taken:  # stems must be unique: IMG.txt maps to one image
            i += 1
            stem = f"{p.stem}_{i}"
        convert = ext in CONVERT_EXTS or ext in RAW_EXTS
        out_ext = ".jpg" if convert else p.suffix
        dst = ds.images_dir / f"{stem}{out_ext}"
        action = "convert" if convert else mode
        print(f"{action:8} {p.name} -> {dst.relative_to(ds.root)}", end="", flush=True)
        n += 1
        if dry_run:
            print()
            taken.add(stem.lower())
            continue
        if ext in RAW_EXTS:
            how = convert_raw_to_jpeg(p, dst, quality, raw_cfg.get("source", "embedded"))
            print(f" ({how})", end="")
            if mode == "move":
                p.unlink()
        elif ext in CONVERT_EXTS:
            convert_to_jpeg(p, dst, quality)
            if mode == "move":
                p.unlink()
        elif mode == "move":
            shutil.move(p, dst)
        else:
            shutil.copy2(p, dst)
        print()
        taken.add(stem.lower())
    print(f"{'would ingest' if dry_run else 'ingested'} {n} file(s)")


# ---------------------------------------------------------------- run

def decide(status: str, force: bool) -> str:
    """-> 'primary' | 'candidate' | 'skip'"""
    if status in ("none", "needs_regen", "missing_txt"):
        return "primary"
    if status == "generated":
        return "primary" if force else "skip"
    # edited / approved: human-owned
    return "candidate" if force else "skip"


def run(ds: Dataset, only: list[str] | None, force: bool, limit: int | None, dry_run: bool,
        model_name: str | None = None) -> int:
    logpath = setup_logging(ds)
    mcfg = dict(ds.cfg["model"])
    if model_name:
        mcfg["name"] = model_name
    images = list_images(ds)
    if only:
        images = [p for p in images if any(fnmatch.fnmatch(p.name, pat) or p.stem == pat for pat in only)]
    todo = []
    for img in images:
        status, _, _ = Sidecar(img).state()
        action = decide(status, force)
        if action != "skip":
            todo.append((img, action))
    if limit:
        todo = todo[:limit]
    print(f"{len(images)} image(s), {len(todo)} to process "
          f"({sum(a == 'candidate' for _, a in todo)} as candidates) with {mcfg['name']}")
    if dry_run:
        for img, a in todo:
            print(f"  {a:9} {img.name}")
        return 0
    if not todo:
        return 0

    backend = get_backend(mcfg["name"], max_new_tokens=mcfg["max_new_tokens"], temperature=mcfg["temperature"])
    if not backend.structured:
        raise SystemExit(f"{mcfg['name']} has no structured output; use it with `wlora compare` only")
    import torch
    t0 = time.time()
    backend.load()
    log.info("loaded %s in %.1fs", backend.repo, time.time() - t0)
    revision = backend.revision()
    prompt = build_prompt(ds)
    vocab = Vocabulary.load(ds)
    settings = {k: v for k, v in mcfg.items() if k != "name"}
    settings["prompt_sha256"] = sha256_text(prompt)

    ok = failed = 0
    started = time.time()
    try:
        for i, (img, action) in enumerate(todo, 1):
            try:
                pil = load_for_inference(img, int(mcfg["max_pixels"]))
                out = backend.run(pil, prompt)
                parsed = extract_json(out.text)
                cls = coerce_classification(ds, parsed.get("classification", {}))
                norm = normalize(ds, vocab, cls, parsed.get("tags", {}))
                result = {
                    "caption": norm.caption,
                    "tags": norm.tags,
                    "description": str(parsed.get("description", "")).strip(),
                    "classification": cls,
                    "raw": {"output": out.text, "dropped_tags": norm.dropped},
                    "generation": {
                        "model": backend.repo, "model_name": mcfg["name"], "revision": revision,
                        "settings": settings, "pipeline_version": __version__, "created_at": now_iso(),
                        "seconds": round(out.seconds, 2), "input_size": list(pil.size),
                    },
                }
                write_result(ds, img, result, action)
                ok += 1
                rate = (time.time() - started) / i
                print(f"[{i}/{len(todo)}] {action:9} {img.name}  {out.seconds:.1f}s  "
                      f"eta {rate * (len(todo) - i) / 60:.1f}m  peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB\n"
                      f"    {norm.caption}", flush=True)
                log.info("ok %s %s %.1fs", action, img.name, out.seconds)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(todo)}] FAILED {img.name}: {e}", flush=True)
                log.error("failed %s: %s\n%s", img.name, e, traceback.format_exc())
    finally:
        backend.unload()
    print(f"done: {ok} ok, {failed} failed in {(time.time() - started) / 60:.1f}m (log: {logpath})")
    return 1 if failed else 0


def write_result(ds: Dataset, img: Path, result: dict, action: str) -> None:
    sc = Sidecar(img)
    # Re-read right before writing: the gallery may have changed things during inference.
    status, data, txt = sc.state()
    if status in ("edited", "approved"):
        action = "candidate"  # possibly became human-owned while we were busy
    data = data or {}
    data.setdefault("image", img.name)
    data["image_sha256"] = sha256_file(img)
    data["subject"] = {k: ds.subject.get(k) for k in ("name", "type", "trigger_word")}
    if action == "candidate":
        data["candidate"] = result
        sc.write(data, caption=None)
        return
    if data.get("caption") or txt:
        data["previous"] = {"caption": txt if txt is not None else data.get("caption"),
                            "status": data.get("status"), "replaced_at": now_iso()}
    data.update({k: v for k, v in result.items() if k != "caption"})
    data["status"] = "generated"
    data["candidate"] = None
    sc.write(data, caption=result["caption"])


# ---------------------------------------------------------------- status / mark / export

def status(ds: Dataset, verbose: bool) -> None:
    counts: dict[str, int] = {}
    for img in list_images(ds):
        st, data, txt = Sidecar(img).state()
        cand = bool(data and data.get("candidate"))
        counts[st] = counts.get(st, 0) + 1
        if cand:
            counts["with_candidate"] = counts.get("with_candidate", 0) + 1
        if verbose:
            print(f"{st:12} {'C' if cand else ' '} {img.name:50} {txt or ''}")
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no images")


def mark(ds: Dataset, names: list[str], new_status: str) -> None:
    if new_status not in ("approved", "needs_regen"):
        raise SystemExit("status must be approved or needs_regen")
    for img in list_images(ds):
        if not any(fnmatch.fnmatch(img.name, n) or img.stem == n for n in names):
            continue
        sc = Sidecar(img)
        _, data, txt = sc.state()
        if data is None and txt is None:
            print(f"skip {img.name}: no caption yet")
            continue
        data = data or {"image": img.name}
        raw = sc.read_txt_raw()
        if new_status == "approved" and raw is not None:  # approving accepts the .txt as it is now
            data["caption"], data["caption_sha256"] = txt, sha256_text(raw)
        data["status"] = new_status
        data["review"] = {"updated_at": now_iso(), "by": "cli"}
        sc.write(data, caption=None)
        print(f"{new_status:12} {img.name}")


def export(ds: Dataset, name: str | None, include: list[str], dry_run: bool) -> None:
    name = name or time.strftime("%Y%m%d-%H%M%S")
    dest = ds.exports_dir / name
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} exists and is not empty")
    rows, skipped = [], 0
    for img in list_images(ds):
        st, data, txt = Sidecar(img).state()
        if st not in include or not txt:
            skipped += 1
            continue
        rows.append((img, txt, st, data or {}))
    print(f"exporting {len(rows)} image(s) to {dest} (skipped {skipped}; statuses: {', '.join(include)})")
    if dry_run:
        return
    dest.mkdir(parents=True)
    with open(dest / "manifest.jsonl", "w") as mf:
        for img, txt, st, data in rows:
            shutil.copy2(img, dest / img.name)
            (dest / f"{img.stem}.txt").write_text(txt + "\n", encoding="utf-8")
            mf.write(json.dumps({"image": img.name, "caption": txt, "status": st,
                                 "classification": data.get("classification"),
                                 "caption_sha256": sha256_text(txt)}, ensure_ascii=False) + "\n")
