# Sidecar format (contract between wandering-lora and wandering-gallery)

For every image `<stem>.<ext>` in a dataset's `images/` directory:

| File | Owner | Purpose |
|---|---|---|
| `<stem>.<ext>` | ingest only | the image (JPEG/PNG/WebP). Never modified after ingest. |
| `<stem>.txt` | shared | **the** training caption, single line, UTF-8, no trailing newline required. Source of truth for caption text. |
| `<stem>.caption.json` | shared | source of truth for everything else: status, classification, raw output, provenance, candidate. |

Stems are unique within a directory (ingest guarantees it), so `<stem>` maps to exactly one image.

## `<stem>.caption.json`

```jsonc
{
  "schema": "wandering-lora/caption@1",
  "image": "IMG_5232.jpg",            // file name of the image, for sanity checks
  "image_sha256": "…",                // hash of image bytes at processing time
  "status": "generated",              // generated | edited | approved | needs_regen
  "caption_sha256": "…",              // sha256 of the UTF-8 .txt content as last written by a tool
  "caption": "siggraph, cat, …",      // copy of .txt content at last tool write (informational)
  "tags": ["siggraph", "cat", …],     // normalized tag list (== caption split on ", ")
  "description": "A long-haired …",   // natural-language description from the model
  "classification": {                 // keys/values defined by the dataset taxonomy
    "composition": "full_body",
    "view": "side",
    "pose": "lying",
    "environment": "indoors",
    "face_visible": true
  },
  "subject": { "name": "Siggraph", "type": "cat", "trigger_word": "siggraph" },
  "raw": { "output": "<verbatim model output>", "dropped_tags": ["…"] },
  "generation": {
    "model": "Qwen/Qwen3.5-4B",
    "revision": "<hf commit>",
    "settings": { "max_pixels": 1048576, "temperature": 0, "…": "…" },
    "pipeline_version": "0.1.0",
    "created_at": "2026-10-10T05:30:00Z"
  },
  "candidate": null,                  // or { caption, tags, description, classification, raw, generation }
  "previous": null,                   // last replaced primary result {caption, status, replaced_at}
  "review": { "updated_at": "…", "by": "gallery" },
  "updated_at": "2026-10-10T05:30:00Z"
}
```

Unknown keys must be preserved by any writer (read-modify-write the whole object).

## Status semantics

- `generated` — produced by the pipeline, untouched by a human.
- `edited` — a human changed the caption but has not approved it.
- `approved` — a human approved the current `.txt`.
- `needs_regen` — a human asked for regeneration; the next `wlora run` replaces the primary result.

Effective status: if `status` is `generated` or `approved` but the sha256 of the raw `.txt` bytes
differs from `caption_sha256`, the .txt was edited by some other tool → treat as `edited`
(an approval only covers the exact text that was approved). Unknown statuses → `edited`.
A `.txt` without json → `edited`.

Human-protected = effective status `edited` or `approved`.

## Writer rules

Pipeline (`wlora run`):
- No sidecar, or `status=generated` and not human-protected → write `.txt` + json (primary).
- `status=needs_regen` → write primary, move old caption into `previous`, status → `generated`.
- Human-protected → never touch `.txt`; with `--force` store the result in `candidate` only.

Gallery:
- Edit caption → write `.txt`, set `caption`, `tags`, `caption_sha256`, status `edited` (or `approved` if approved in the same action).
- Approve → status `approved` (and sync `caption_sha256` with the current .txt).
- Mark for regeneration → status `needs_regen`.
- Accept candidate → candidate becomes primary (writes `.txt`), status `edited`, candidate → null.
- Discard candidate → candidate → null.

Both sides:
- Atomic writes: write `.<name>.tmp-<pid>-<rand>` in the same dir, fsync, rename. Write `.txt` first, then json.
- Optimistic concurrency: the gallery sends the `updated_at` + `caption_sha256` it loaded; the server
  re-reads both files and returns HTTP 409 if they changed. The pipeline re-reads the json right
  before writing and re-evaluates the rules if it changed since inference started.
- Timestamps are ISO-8601 UTC.
