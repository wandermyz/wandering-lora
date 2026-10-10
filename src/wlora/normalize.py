"""Turn a model's structured answer into classification + a normalized training caption."""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from .config import Dataset


def _clean(tag: str) -> str:
    t = tag.strip().lower().replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,;:'\"`-*#")


@dataclass
class Vocabulary:
    aliases: dict[str, str] = field(default_factory=dict)
    blocklist: list[str] = field(default_factory=list)
    exclusive: list[list[str]] = field(default_factory=list)
    implies: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, ds: Dataset) -> "Vocabulary":
        base = yaml.safe_load(resources.files("wlora.defaults").joinpath("vocabulary.yaml").read_text())
        extra_path = ds.cfg["caption"].get("vocabulary")
        if extra_path:
            p = Path(extra_path).expanduser()
            p = p if p.is_absolute() else ds.root / p
            extra = yaml.safe_load(p.read_text()) or {}
            base["aliases"].update(extra.get("aliases", {}))
            base["blocklist"] += extra.get("blocklist", [])
            base["exclusive"] += extra.get("exclusive", [])
            base["implies"].update(extra.get("implies", {}))
        return cls(
            aliases={_clean(k): _clean(v or "") for k, v in base["aliases"].items()},
            blocklist=[_clean(b) for b in base["blocklist"]],
            exclusive=[[_clean(t) for t in g] for g in base["exclusive"]],
            implies={_clean(k): _clean(v) for k, v in base["implies"].items()},
        )

    def canonical(self, tag: str) -> str:
        t = _clean(tag)
        return self.aliases.get(t, t)

    def blocked(self, tag: str) -> bool:
        return any(fnmatch.fnmatchcase(tag, b) for b in self.blocklist)


@dataclass
class Normalized:
    caption: str
    tags: list[str]
    classification: dict
    dropped: list[dict]


def coerce_classification(ds: Dataset, raw: dict) -> dict:
    """Validate model classification against the taxonomy; unknown -> None."""
    out: dict = {}
    raw = {_clean(k).replace(" ", "_"): v for k, v in (raw or {}).items()}
    for name, dim in ds.taxonomy.items():
        v = raw.get(name)
        if dim.get("type", "enum") == "bool":
            if isinstance(v, str):
                v = {"true": True, "yes": True, "false": False, "no": False}.get(v.strip().lower())
            out[name] = v if isinstance(v, bool) else None
        else:
            v = _clean(str(v)).replace(" ", "_").replace("-", "_") if v is not None else None
            out[name] = v if v in dim["values"] else None
    return out


def normalize(ds: Dataset, vocab: Vocabulary, classification: dict, tag_groups: dict[str, list]) -> Normalized:
    cap = ds.cfg["caption"]
    order: list[str] = cap["order"]
    ident = ds.cfg["identity"]
    dropped: list[dict] = []

    # (category, tag) candidates in priority order: fixed/taxonomy tags first.
    cands: list[tuple[str, str, str]] = []  # (category, tag, source)
    cands.append(("trigger", ds.trigger.lower(), "config"))
    for t in ds.subject.get("class_tags", []):
        cands.append(("class", _clean(t), "config"))
    tax_tags: list[tuple[str, str]] = []
    for name, dim in ds.taxonomy.items():
        v = classification.get(name)
        if v is None:
            continue
        if dim.get("type", "enum") == "bool":
            t = dim.get("tag_true" if v else "tag_false")
        else:
            t = (dim["values"].get(v) or {}).get("tag")
        if t:
            cat = dim.get("category") or (name if name in order else "other")
            tax_tags.append((cat, _clean(t)))
    # "no humans"-style bool tags go into the fixed slot, ahead of fixed_tags
    for cat, t in tax_tags:
        if cat == "fixed":
            cands.append(("fixed", t, "taxonomy"))
    for t in cap.get("fixed_tags", []):
        cands.append(("fixed", _clean(t), "config"))
    for cat, t in tax_tags:
        if cat != "fixed":
            cands.append((cat, t, "taxonomy"))
    for cat, tags in (tag_groups or {}).items():
        c = _clean(cat)
        c = c if c in order else "other"
        for t in tags if isinstance(tags, list) else [tags]:
            if isinstance(t, str):
                cands.append((c, t, "model"))

    seen: set[str] = set()
    kept: list[tuple[str, str]] = []
    for cat, raw_tag, src in cands:
        t = vocab.canonical(raw_tag) if src == "model" else raw_tag
        if not t:
            dropped.append({"tag": raw_tag, "reason": "alias-drop"})
            continue
        if src == "model":
            if vocab.blocked(t):
                dropped.append({"tag": raw_tag, "reason": "blocklist"}); continue
            if len(t.split()) > 4 or len(t) > 40:
                dropped.append({"tag": raw_tag, "reason": "too-long"}); continue
            if ds.trigger.lower() in t or t in {_clean(x) for x in ds.subject.get("class_tags", [])}:
                dropped.append({"tag": raw_tag, "reason": "duplicate"}); continue
            if ident.get("mode", "drop") == "drop" and t not in ident.get("keep_tags", []) and any(
                fnmatch.fnmatchcase(t, p) for p in ident.get("drop_tags", [])
            ):
                dropped.append({"tag": raw_tag, "reason": "identity"}); continue
        if t in seen:
            if src == "model":
                dropped.append({"tag": raw_tag, "reason": "duplicate"})
            continue
        seen.add(t)
        kept.append((cat, t))

    # contradictions: earlier (higher priority) wins
    present = [t for _, t in kept]
    remove: set[str] = set()
    for group in vocab.exclusive:
        hits = [t for t in present if t in group and t not in remove]
        for loser in hits[1:]:
            remove.add(loser)
            dropped.append({"tag": loser, "reason": f"contradicts:{hits[0]}"})
    for a, b in vocab.implies.items():
        if a in present and b in present and b not in remove:
            remove.add(b)
            dropped.append({"tag": b, "reason": f"implied-by:{a}"})
    kept = [(c, t) for c, t in kept if t not in remove]

    rank = {c: i for i, c in enumerate(order)}
    kept.sort(key=lambda ct: rank.get(ct[0], len(order)))  # stable: keeps priority within category
    tags = [t for _, t in kept]
    max_tags = cap.get("max_tags") or 0
    if max_tags and len(tags) > max_tags:
        dropped += [{"tag": t, "reason": "max-tags"} for t in tags[max_tags:]]
        tags = tags[:max_tags]
    return Normalized(", ".join(tags), tags, classification, dropped)
