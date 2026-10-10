import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from wlora import pipeline
from wlora.config import Dataset, default_config_text
from wlora.normalize import Vocabulary, coerce_classification, normalize
from wlora.prompt import extract_json
from wlora.sidecar import Sidecar, sha256_text


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    (tmp_path / "dataset.yaml").write_text(default_config_text())
    (tmp_path / "images").mkdir()
    return Dataset.load(tmp_path)


def norm(ds, cls, tags):
    return normalize(ds, Vocabulary.load(ds), coerce_classification(ds, cls), tags)


def test_trigger_first_once_and_ordered(ds):
    n = norm(ds, {"composition": "full_body", "pose": "sitting", "environment": "indoors",
                  "humans_visible": False, "eyes_open": True},
             {"lighting": ["natural light"], "pose": ["Sitting"], "gaze": ["looking at camera"],
              "setting": ["siggraph", "hardwood_floor"], "composition": ["full body"]})
    assert n.tags == ["siggraph", "cat", "no humans", "animal focus", "full body", "sitting",
                      "looking at viewer", "indoors", "wooden floor", "sunlight"]


def test_identity_and_quality_tags_dropped(ds):
    n = norm(ds, {}, {"appearance": ["fluffy fur", "long hair", "wet fur"], "other": ["cute", "masterpiece"],
                      "expression": ["green eyes", "closed eyes"]})
    assert "wet fur" in n.tags and "closed eyes" in n.tags
    for t in ("fluffy fur", "long hair", "cute", "masterpiece", "green eyes"):
        assert t not in n.tags
    reasons = {d["tag"]: d["reason"] for d in n.dropped}
    assert reasons["long hair"] == "identity" and reasons["cute"] == "blocklist"


def test_identity_keep_mode(ds):
    ds.cfg["identity"]["mode"] = "keep"
    assert "long hair" in norm(ds, {}, {"appearance": ["long hair"]}).tags


def test_contradictions_taxonomy_wins(ds):
    n = norm(ds, {"environment": "indoors", "eyes_open": False, "humans_visible": True},
             {"setting": ["outdoors"], "gaze": ["looking at viewer"], "objects": ["hand"]})
    assert "indoors" in n.tags and "outdoors" not in n.tags
    assert "closed eyes" in n.tags and "looking at viewer" not in n.tags
    assert "hand" in n.tags and "no humans" not in n.tags


def test_implied_tag_removed(ds):
    n = norm(ds, {"pose": "lying"}, {"pose": ["on back"]})
    assert "on back" in n.tags and "lying" not in n.tags


def test_invalid_classification_becomes_none(ds):
    c = coerce_classification(ds, {"pose": "levitating", "face_visible": "yes", "view": "Three-Quarter"})
    assert c["pose"] is None and c["face_visible"] is True and c["view"] == "three_quarter"


def test_extract_json_fenced_and_truncated():
    assert extract_json('```json\n{"a": {"b": [1]}}\n```') == {"a": {"b": [1]}}
    assert extract_json('<think>x</think> here: {"a": {"b": "}"}') == {"a": {"b": "}"}}
    assert extract_json('```json\n{"a": {"b": [1]}\n```') == {"a": {"b": [1]}}


def _img(ds, name="a.jpg"):
    p = ds.images_dir / name
    Image.new("RGB", (8, 8)).save(p)
    return p


RESULT = {"caption": "siggraph, cat, sitting", "tags": ["siggraph", "cat", "sitting"], "description": "d",
          "classification": {}, "raw": {"output": "{}"}, "generation": {}}


def test_sidecar_status_and_protection(ds):
    img = _img(ds)
    sc = Sidecar(img)
    assert sc.state()[0] == "none"
    pipeline.write_result(ds, img, RESULT, "primary")
    st, data, txt = sc.state()
    assert st == "generated" and txt == RESULT["caption"] and data["caption_sha256"] == sha256_text(txt)
    assert pipeline.decide(st, force=False) == "skip"

    sc.txt_path.write_text("siggraph, cat, manually fixed")  # human edit outside any tool
    assert sc.state()[0] == "edited"
    assert pipeline.decide("edited", force=True) == "candidate"
    pipeline.write_result(ds, img, {**RESULT, "caption": "siggraph, cat, new"}, "primary")
    st, data, txt = sc.state()
    assert txt == "siggraph, cat, manually fixed"  # never overwritten
    assert data["candidate"]["caption"] == "siggraph, cat, new"


def test_needs_regen_replaces_and_keeps_previous(ds):
    img = _img(ds)
    pipeline.write_result(ds, img, RESULT, "primary")
    data = json.loads(Sidecar(img).json_path.read_text())
    data["status"] = "needs_regen"
    data["custom_key"] = 1
    Sidecar(img).json_path.write_text(json.dumps(data))
    assert pipeline.decide(Sidecar(img).state()[0], False) == "primary"
    pipeline.write_result(ds, img, {**RESULT, "caption": "siggraph, cat, lying"}, "primary")
    st, data, txt = Sidecar(img).state()
    assert st == "generated" and txt == "siggraph, cat, lying"
    assert data["previous"]["caption"] == "siggraph, cat, sitting" and data["custom_key"] == 1


def test_export_only_approved(ds, tmp_path):
    a, b = _img(ds, "a.jpg"), _img(ds, "b.jpg")
    for img in (a, b):
        pipeline.write_result(ds, img, RESULT, "primary")
    pipeline.mark(ds, ["a"], "approved")
    pipeline.export(ds, "x", ["approved"], dry_run=False)
    out = ds.exports_dir / "x"
    assert sorted(p.name for p in out.iterdir()) == ["a.jpg", "a.txt", "manifest.jsonl"]
    assert (out / "a.txt").read_text().strip() == RESULT["caption"]


def test_ingest_heic_like_and_unique_stems(ds, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (8, 8)).save(src / "x.png")
    Image.new("RGB", (8, 8)).save(src / "x.jpg")
    import os, time
    for p in src.iterdir():
        os.utime(p, (time.time() - 60, time.time() - 60))
    pipeline.ingest(ds, src, "copy", dry_run=False)
    names = sorted(p.name for p in ds.images_dir.iterdir())
    assert len(names) == 2 and len({Path(n).stem for n in names}) == 2


def test_trailing_newline_counts_as_edit_like_gallery(ds):
    img = _img(ds)
    pipeline.write_result(ds, img, RESULT, "primary")
    Sidecar(img).txt_path.write_text(RESULT["caption"] + "\n")
    assert Sidecar(img).state()[0] == "edited"
    pipeline.mark(ds, ["a"], "approved")
    assert Sidecar(img).state()[0] == "approved"
    assert Sidecar(img).txt_path.read_text() == RESULT["caption"] + "\n"  # mark never rewrites .txt


def test_external_edit_revokes_approval(ds):
    img = _img(ds)
    pipeline.write_result(ds, img, RESULT, "primary")
    pipeline.mark(ds, ["a"], "approved")
    assert Sidecar(img).state()[0] == "approved"
    Sidecar(img).txt_path.write_text("siggraph, cat, changed")
    assert Sidecar(img).state()[0] == "edited"


def test_raw_extensions_exclude_video():
    from wlora.imageio import RAW_EXTS
    assert ".cr3" in RAW_EXTS and ".mov" not in RAW_EXTS


def test_ingest_skips_video(ds, tmp_path, capsys):
    import os, time
    src = tmp_path / "src"
    src.mkdir()
    (src / "clip.MOV").write_bytes(b"x")
    os.utime(src / "clip.MOV", (time.time() - 60, time.time() - 60))
    pipeline.ingest(ds, src, "move", dry_run=False)
    assert (src / "clip.MOV").exists() and not any(ds.images_dir.iterdir())
