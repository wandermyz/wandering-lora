"""Builds the structured-captioning prompt from the dataset config."""
from __future__ import annotations

import json
import re

from .config import Dataset

TAG_CATEGORIES = {
    "composition": "framing, e.g. close-up, upper body, full body, wide shot, cropped",
    "view": "camera angle, e.g. from side, from above, from below, from behind",
    "pose": "body pose, e.g. sitting, lying, on back, on side, curled up, standing, stretching",
    "action": "what the subject is doing, e.g. sleeping, eating, licking, yawning, playing",
    "expression": "visible expression, e.g. open mouth, tongue out, closed eyes, half-closed eyes",
    "gaze": "e.g. looking at viewer, looking up, looking away, looking to the side",
    "appearance": "only temporary or unusual appearance, e.g. wet fur, wearing a cone",
    "accessories": "worn items, e.g. collar, bell, bowtie, harness, clothes",
    "objects": "notable objects the subject interacts with or that are prominent, e.g. toy, bowl, blanket, box",
    "setting": "location, e.g. bed, couch, rug, windowsill, garden, carpet, wooden floor",
    "background": "e.g. simple background, blurry background, white wall, plant",
    "lighting": "e.g. sunlight, dim lighting, backlighting, shadow",
    "meta": "visible overlays or photo artifacts, e.g. timestamp, watermark, text, fisheye, motion blur",
}


def json_skeleton(ds: Dataset) -> dict:
    cls = {}
    for name, dim in ds.taxonomy.items():
        if dim.get("type", "enum") == "bool":
            cls[name] = "true|false"
        else:
            cls[name] = "|".join(dim["values"])
    return {
        "description": "<2-4 sentences>",
        "classification": cls,
        "tags": {k: ["..."] for k in TAG_CATEGORIES},
    }


def build_prompt(ds: Dataset) -> str:
    s = ds.subject
    dims = []
    for name, dim in ds.taxonomy.items():
        if dim.get("type", "enum") == "bool":
            dims.append(f"- {name} (true/false): {dim.get('description', '')}")
        else:
            vals = ", ".join(
                f"{v}" + (f" ({d['description']})" if d and d.get("description") else "")
                for v, d in dim["values"].items()
            )
            desc = f" — {dim['description']}" if dim.get("description") else ""
            dims.append(f"- {name}{desc}: one of {vals}")
    cats = "\n".join(f"- {k}: {v}" for k, v in TAG_CATEGORIES.items())
    features = s.get("features")
    return f"""You are labeling a photo for an image-generation training dataset.
The photo shows a specific {s['type']}. {f"Its permanent appearance is: {features}. " if features else ""}Do NOT describe these permanent features in tags; focus on what varies between photos.

Return ONLY a JSON object with this shape:
{json.dumps(json_skeleton(ds), indent=1)}

Classification dimensions (choose exactly one listed value; use "other"/"unknown" if unsure):
{chr(10).join(dims)}

Tag categories (short lowercase Danbooru-style tags, 1-3 words each, 0-4 tags per category, empty list if nothing applies):
{cats}

Rules:
- Only tag what is clearly visible. Do not guess.
- No quality or subjective words (cute, beautiful, cozy, adorable, high quality).
- Do not name the {s['type']} species or breed in tags.
- description: plain factual English, mention pose, camera angle, setting and lighting."""


def extract_json(text: str) -> dict:
    """Parse the first JSON object in model output (tolerates ``` fences / chatter)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"```(?:json)?", "", t)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    # Small models sometimes stop before the final braces: close what is open.
    if not in_str and 0 < depth <= 3:
        tail = t[start:].rstrip().rstrip(",")
        return json.loads(tail + "}" * depth)
    raise ValueError("unterminated JSON object in model output")
