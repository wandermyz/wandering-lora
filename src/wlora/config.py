"""Dataset configuration (dataset.yaml)."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "dataset.yaml"


def default_config_text() -> str:
    return resources.files("wlora.defaults").joinpath("dataset.yaml").read_text()


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "taxonomy":
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Dataset:
    root: Path
    cfg: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> "Dataset":
        root = Path(root).expanduser().resolve()
        if root.is_file():
            root = root.parent
        path = root / CONFIG_NAME
        if not path.exists():
            raise SystemExit(f"{path} not found (run `wlora init {root}` first)")
        user = yaml.safe_load(path.read_text()) or {}
        cfg = _merge(yaml.safe_load(default_config_text()), user)
        ds = cls(root, cfg)
        ds.validate()
        return ds

    def _path(self, key: str) -> Path:
        p = Path(self.cfg["paths"][key]).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def images_dir(self) -> Path:
        return self._path("images")

    @property
    def exports_dir(self) -> Path:
        return self._path("exports")

    @property
    def state_dir(self) -> Path:
        return self._path("state")

    @property
    def subject(self) -> dict:
        return self.cfg["subject"]

    @property
    def trigger(self) -> str:
        return self.subject["trigger_word"].strip()

    @property
    def taxonomy(self) -> dict[str, dict]:
        return self.cfg["taxonomy"]

    def validate(self) -> None:
        if not self.trigger:
            raise SystemExit("subject.trigger_word must not be empty")
        for name, dim in self.taxonomy.items():
            kind = dim.get("type", "enum")
            if kind not in ("enum", "bool"):
                raise SystemExit(f"taxonomy.{name}: unknown type {kind!r}")
            if kind == "enum" and not dim.get("values"):
                raise SystemExit(f"taxonomy.{name}: enum needs values")
