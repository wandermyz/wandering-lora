# wandering-lora

Local AI caption + classification pipeline that prepares image datasets for
LoRA training (tuned for WAI-Illustrious / SDXL tag-style captions).
Everything runs locally; no images leave the machine.

- Vision model: **Qwen3.5-4B** (default) via Hugging Face transformers, structured JSON output.
- Output: `<stem>.txt` (training caption) + `<stem>.caption.json` (classification,
  description, raw model output, provenance, review status) next to every image.
- Review: in [wandering-gallery](https://github.com/wandermyz/wandering-gallery)
  (Caption section of the info panel: edit / approve / regenerate / accept candidate).
- LoRA training itself is out of scope.

The sidecar contract shared with the gallery is in [docs/sidecar-format.md](docs/sidecar-format.md).

## Install

Requires an NVIDIA GPU (tested: RTX 5070 Ti 16 GB, driver CUDA 13.x) and [uv](https://docs.astral.sh/uv/).
The environment is isolated in `.venv` (PyTorch cu130 wheels); ComfyUI and system CUDA are untouched.

```sh
git clone git@github.com:wandermyz/wandering-lora.git && cd wandering-lora
UV_PYTHON_PREFERENCE=only-managed uv sync   # managed Python ships headers needed by Triton kernels
.venv/bin/hf download Qwen/Qwen3.5-4B       # ~9 GB; weights go to ~/.cache/huggingface
# optional, for `wlora compare`:
.venv/bin/hf download Qwen/Qwen3.5-9B fancyfeast/llama-joycaption-beta-one-hf-llava SmilingWolf/wd-eva02-large-tagger-v3
```

Add `.venv/bin` to `PATH` or call `.venv/bin/wlora`. After downloading, everything works
offline (`HF_HUB_OFFLINE=1`).

## Workflow

```sh
wlora init ~/Datasets/siggraph --trigger siggraph --type cat
$EDITOR ~/Datasets/siggraph/dataset.yaml     # subject features, identity tags, taxonomy
wlora ingest ~/Datasets/siggraph ~/Shared    # move images in; HEIC/raw -> JPEG, videos skipped
wlora run ~/Datasets/siggraph --limit 20     # prototype on a few images
wlora run ~/Datasets/siggraph                # everything not yet captioned (resumable)
wlora status ~/Datasets/siggraph -v
# review in wandering-gallery (library pointing at ~/Datasets/siggraph/images, writable: true)
wlora run ~/Datasets/siggraph                # picks up images marked "Regenerate" in the gallery
wlora export ~/Datasets/siggraph --name v1   # approved images + .txt -> exports/v1/
```

Other commands: `wlora compare` (side-by-side HTML of several models on sample images),
`wlora prompt` (show the model prompt), `wlora models`, `wlora mark <ds> approved|needs_regen <images…>`.

### Dataset layout

```
~/Datasets/siggraph/
  dataset.yaml            # config (see src/wlora/defaults/dataset.yaml for all options)
  images/                 # images + sidecars; a gallery library
    IMG_5232.jpg
    IMG_5232.txt          # "siggraph, cat, no humans, animal focus, full body, from above, on back, ..."
    IMG_5232.caption.json
  exports/<name>/         # training-ready copies: image + .txt + manifest.jsonl
  .wlora/                 # wlora.log, compare reports
```

### Input formats

JPEG/PNG/WebP are moved as-is. HEIC/HEIF/AVIF are converted to JPEG (q95, orientation
applied, EXIF kept). Camera raws (CR3, CR2, NEF, ARW, DNG, RAF, ORF, RW2, …) are converted
via LibRaw (`rawpy`): by default the camera-rendered full-size embedded JPEG is used as-is
(byte-exact, camera colours, EXIF kept); `ingest.raw.source: develop` demosaics instead.
Like HEIC, the original raw is deleted after conversion with `ingest.mode: move`
(kept with `copy`). Videos and other files are skipped.

### Re-runs and protection of human work

`wlora run` only processes images that have no caption, are marked `needs_regen`, or whose
`.txt` is missing. Captions that a human edited or approved are never overwritten:
`--force` re-generates them into a `candidate` that can be accepted or discarded in the
gallery. `--force` on untouched `generated` captions replaces them (old caption kept in
`previous`). The pipeline re-reads sidecars right before writing, and the gallery rejects
stale saves with HTTP 409, so there is no silent last-writer-wins.

## Captions

Order (configurable, `caption.order`):

```
trigger, class, fixed (no humans, animal focus), composition, view, pose, action,
expression, gaze, appearance, accessories, objects, environment, setting, background, lighting
```

- The trigger word comes from config and is inserted exactly once; the model never sees or invents it.
- Taxonomy answers (e.g. `composition: full_body`, `eyes_open: false`) emit canonical tags
  and win over contradicting free-form tags.
- Permanent identity features (`subject.features`) are given to the model as context and
  matching tags (`identity.drop_tags`) are removed so the trigger word absorbs them.
  `identity.mode: keep` disables this.
- Vocabulary normalization (synonyms, blocklist of quality/subjective words, exclusive
  groups, implied tags) lives in `src/wlora/defaults/vocabulary.yaml`; extend per dataset
  with `caption.vocabulary: my-vocab.yaml`.
- Every dropped tag is recorded with its reason in `raw.dropped_tags`.

## Classification

Defined per dataset in `taxonomy:` — enums (one value) or booleans, each optionally mapped
to a caption tag. The default is cat-oriented (composition, view, pose, environment,
face/eyes/tail visibility, occlusion, humans visible) but nothing in the code is
cat-specific; change it for a person, object or character.

## Performance (RTX 5070 Ti)

Qwen3.5-4B bf16, images downscaled to ≤1 MP: ~9 GiB peak VRAM, ~12–13 s per image
(~15 min per 70 images). Do not run at the same time as heavy ComfyUI jobs.
