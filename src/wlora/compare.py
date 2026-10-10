"""Run several models over sample images and write a side-by-side HTML report."""
from __future__ import annotations

import base64
import html
import io
import json
import time
from pathlib import Path

import torch

from .config import Dataset
from .imageio import load_for_inference
from .models import get_backend
from .normalize import Vocabulary, coerce_classification, normalize
from .prompt import build_prompt, extract_json

DEFAULT_RUNS = ["qwen3.5-4b", "qwen3.5-9b", "joycaption:descriptive", "joycaption:danbooru", "wd14"]


def _thumb(img: Path) -> str:
    im = load_for_inference(img, 400 * 400)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def compare(ds: Dataset, images: list[Path], runs: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(ds)
    vocab = Vocabulary.load(ds)
    max_pixels = int(ds.cfg["model"]["max_pixels"])
    results: dict[str, dict[str, dict]] = {}
    stats: dict[str, dict] = {}
    by_model: dict[str, list[str]] = {}
    for r in runs:
        m, _, mode = r.partition(":")
        by_model.setdefault(m, []).append(r)
    for model, model_runs in by_model.items():
        backend = get_backend(model)
        t = time.time()
        try:
            backend.load()
        except Exception as e:
            for r in model_runs:
                stats[r] = {"error": f"load failed: {e}"}
            continue
        load_s = time.time() - t
        for r in model_runs:
            mode = r.partition(":")[2]
            secs = []
            for img in images:
                pil = load_for_inference(img, max_pixels)
                try:
                    out = backend.run(pil, prompt if backend.structured else mode)
                    entry = {"raw": out.text, "seconds": out.seconds}
                    if backend.structured:
                        try:
                            parsed = extract_json(out.text)
                            cls = coerce_classification(ds, parsed.get("classification", {}))
                            entry["caption"] = normalize(ds, vocab, cls, parsed.get("tags", {})).caption
                            entry["classification"] = cls
                            entry["description"] = parsed.get("description", "")
                        except Exception as e:
                            entry["error"] = f"parse: {e}"
                    secs.append(out.seconds)
                except Exception as e:
                    entry = {"error": str(e)}
                results.setdefault(img.name, {})[r] = entry
                print(f"{r:24} {img.name:40} {entry.get('seconds', 0):5.1f}s", flush=True)
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            stats[r] = {"load_s": round(load_s, 1), "avg_s": round(sum(secs) / max(1, len(secs)), 2),
                        "peak_vram_gib": round(peak, 2), "n": len(secs)}
        backend.unload()
    (out_dir / "compare.json").write_text(json.dumps({"stats": stats, "results": results}, indent=2))
    path = out_dir / "compare.html"
    path.write_text(_html(images, runs, results, stats))
    return path


def _html(images, runs, results, stats) -> str:
    e = html.escape
    head = "".join(f"<th>{e(r)}<br><small>{e(json.dumps(stats.get(r, {})))}</small></th>" for r in runs)
    rows = []
    for img in images:
        cells = []
        for r in runs:
            x = results.get(img.name, {}).get(r, {})
            parts = []
            if "caption" in x:
                parts.append(f"<div class=cap>{e(x['caption'])}</div>")
                parts.append(f"<div class=cls>{e(json.dumps(x['classification']))}</div>")
                parts.append(f"<div class=desc>{e(x.get('description', ''))}</div>")
                parts.append(f"<details><summary>raw</summary><pre>{e(x['raw'])}</pre></details>")
            elif "raw" in x:
                parts.append(f"<div class=desc>{e(x['raw'])}</div>")
            if "error" in x:
                parts.append(f"<div class=err>{e(x['error'])}</div>")
            cells.append(f"<td>{''.join(parts)}</td>")
        rows.append(f"<tr><td><img src='{_thumb(img)}'><br><small>{e(img.name)}</small></td>{''.join(cells)}</tr>")
    return f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>wlora compare</title><style>
body{{font:13px system-ui;margin:0}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;vertical-align:top;padding:6px;max-width:340px}}
th{{position:sticky;top:0;background:#f4f4f4}} img{{max-width:260px}} .cap{{font-weight:600;color:#063}} .cls{{color:#555;font-family:monospace;font-size:11px;margin:4px 0}}
.err{{color:#b00}} pre{{white-space:pre-wrap;font-size:11px}}</style>
<table><tr><th>image</th>{head}</tr>{''.join(rows)}</table>"""
