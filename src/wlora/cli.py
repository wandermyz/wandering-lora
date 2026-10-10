"""wlora — local caption + classification pipeline for LoRA datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CONFIG_NAME, Dataset, default_config_text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wlora", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a dataset directory with dataset.yaml")
    p.add_argument("dataset")
    p.add_argument("--trigger", help="trigger word (also used as subject name)")
    p.add_argument("--type", help="subject type, e.g. cat, person, character")

    p = sub.add_parser("ingest", help="move/copy images into the dataset (HEIC is converted to JPEG)")
    p.add_argument("dataset")
    p.add_argument("source", type=Path)
    p.add_argument("--copy", action="store_true", help="copy instead of the configured mode")
    p.add_argument("--move", action="store_true", help="move instead of the configured mode")
    p.add_argument("-n", "--dry-run", action="store_true")

    p = sub.add_parser("run", help="caption + classify images that need it")
    p.add_argument("dataset")
    p.add_argument("images", nargs="*", help="only these file names / stems / globs")
    p.add_argument("--force", action="store_true",
                   help="reprocess generated captions; for edited/approved ones store a candidate")
    p.add_argument("--limit", type=int)
    p.add_argument("--model", help="override model.name")
    p.add_argument("-n", "--dry-run", action="store_true")

    p = sub.add_parser("status", help="summarize review status")
    p.add_argument("dataset")
    p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("mark", help="set review status (approved, needs_regen, ...)")
    p.add_argument("dataset")
    p.add_argument("status")
    p.add_argument("images", nargs="+")

    p = sub.add_parser("export", help="copy images + .txt captions into exports/<name>")
    p.add_argument("dataset")
    p.add_argument("--name")
    p.add_argument("--include", default="approved",
                   help="comma-separated statuses to include (default: approved)")
    p.add_argument("-n", "--dry-run", action="store_true")

    p = sub.add_parser("compare", help="run several models on sample images -> HTML report")
    p.add_argument("dataset")
    p.add_argument("--models", help="comma-separated runs, e.g. qwen3.5-4b,joycaption:descriptive,wd14")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("prompt", help="print the prompt sent to the model")
    p.add_argument("dataset")

    sub.add_parser("models", help="list available models")

    a = ap.parse_args(argv)

    if a.cmd == "models":
        from .models import REGISTRY
        for k, (kind, kw) in REGISTRY.items():
            print(f"{k:18} {kind:11} {kw['repo']} {kw.get('quant') or ''}")
        return 0
    if a.cmd == "init":
        root = Path(a.dataset).expanduser()
        (root / "images").mkdir(parents=True, exist_ok=True)
        cfg = root / CONFIG_NAME
        if cfg.exists():
            print(f"{cfg} already exists")
            return 0
        text = default_config_text()
        if a.trigger:
            text = text.replace('trigger_word: "siggraph"', f'trigger_word: "{a.trigger}"')
            text = text.replace('name: "Siggraph"', f'name: "{a.trigger.title()}"')
        if a.type:
            text = text.replace('type: "cat"', f'type: "{a.type}"').replace('class_tags: ["cat"]', f'class_tags: ["{a.type}"]')
        cfg.write_text(text)
        print(f"created {cfg} — review subject/identity/taxonomy before running")
        return 0

    ds = Dataset.load(a.dataset)
    from . import pipeline
    if a.cmd == "ingest":
        mode = "copy" if a.copy else "move" if a.move else None
        pipeline.ingest(ds, a.source.expanduser(), mode, a.dry_run)
    elif a.cmd == "run":
        return pipeline.run(ds, a.images, a.force, a.limit, a.dry_run, a.model)
    elif a.cmd == "status":
        pipeline.status(ds, a.verbose)
    elif a.cmd == "mark":
        pipeline.mark(ds, a.images, a.status)
    elif a.cmd == "export":
        pipeline.export(ds, a.name, [s.strip() for s in a.include.split(",")], a.dry_run)
    elif a.cmd == "prompt":
        from .prompt import build_prompt
        print(build_prompt(ds))
    elif a.cmd == "compare":
        from .compare import DEFAULT_RUNS, compare
        imgs = pipeline.list_images(ds)[: a.limit]
        runs = a.models.split(",") if a.models else DEFAULT_RUNS
        out = compare(ds, imgs, runs, a.out or ds.state_dir / "compare")
        print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
