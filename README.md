# Realtime 3D Canvas → AI Image Editing

A browser-based 3D canvas where you import, position, rotate, and scale 3D objects,
then watch an AI re-render the viewport **in ~realtime** as you edit. Generation runs
on a **self-hosted FLUX.2 Klein 9B** model (distilled, 4-step) loaded in-process on a
local H100 — no remote API, no submit-then-poll latency floor.

Every time you move the scene, the frontend captures the WebGL viewport and sends it to
the backend as a *reference image*; FLUX.2 Klein edits it according to your prompt and
the result appears side-by-side.

## Requirements

- **GPU:** 1× NVIDIA H100 80GB (or any ~40GB+ CUDA GPU). Runs bf16, no quantization —
  the ~9B flow model plus the Qwen3 text encoder are ~34GB resident in VRAM.
- **Python:** 3.10+ with a CUDA build of PyTorch.
- A Hugging Face account that has accepted the FLUX.2 license (weights are gated).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

(`torch` must be a CUDA build matching your driver. This repo was developed on
torch 2.7 + CUDA 12.8. `diffusers` is pulled from `main` because FLUX.2 support is recent.)

### 2. Authenticate + accept the license (one-time)

The weights are gated. Visit and click **"Agree and access"**:
<https://huggingface.co/black-forest-labs/FLUX.2-klein-9B>

Then log in (the 9B model is under a **non-commercial** license — fine for a hackathon):

```bash
hf auth login
```

### 3. Start the server

```bash
python server.py
```

On first run this downloads the weights (~34GB) and loads them into VRAM, then runs a
warmup inference. When you see `Ready — model warm in VRAM.` it's good to go. Confirm
the GPU is resident with `nvidia-smi`.

**Echo mode** (no GPU / no weights — mirrors the capture back, for UI testing):

```bash
KLEIN_ECHO=1 python server.py
```

### 4. Open the frontend

The server serves the UI itself, so just visit it in a browser:

- On the box: <http://localhost:3000/>
- Remote box: **http://&lt;box&gt;:3000/** (the page calls the API on the same origin, so
  it works with no config — just make sure port 3000 is reachable).

Prefer to open the file directly? Open `index.html` from disk (it falls back to
`http://localhost:3000`), handy with an SSH port-forward
(`ssh -L 3000:localhost:3000 user@box`). Point it anywhere with
`?api=http://<box>:3000` or `localStorage.apiBase`.

Import a model (`.glb .gltf .obj .fbx .stl`), then move/rotate/scale it — generation
auto-triggers ~300ms after you stop. The **Generate** button still works for manual runs.

## How realtime works

- **Auto-regenerate:** any edit schedules a debounced generate; only the latest frame is
  sent. In-flight requests are cancelled (`AbortController`) so the loop never floods the GPU.
- **Distilled 4-step model:** `num_inference_steps=4`, kept warm in VRAM, GPU access
  serialized by a single lock (one H100 = one inference at a time), inference run in a
  worker thread so the event loop stays responsive.
- **Cleaner reference frame:** the grid is hidden during capture, the frame is letterboxed
  to the output resolution, JPEG-encoded to shrink the upload, and the prompt is prefixed
  with *"same composition and camera angle as the reference image, photorealistic, "* to
  lock structure. The seed is fixed by default for frame-to-frame coherence.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KLEIN_MODEL_ID` | `black-forest-labs/FLUX.2-klein-9B` | HF repo id to load. |
| `KLEIN_ECHO` | unset | `1` forces echo mode (mirror input back, no GPU). Auto-enabled if CUDA is unavailable. |
| `KLEIN_CPU_OFFLOAD` | unset | `1` enables model CPU offload (fallback if VRAM is tight; higher latency). |
| `KLEIN_NO_WARMUP` | unset | `1` skips the startup warmup inference. |
| `PORT` | `3000` | Server port. |

## API

### `GET /`
Health check → `{"status":"ok","mode":"klein|echo","model":"...","busy":false}`

### `POST /generate`

**Request:**
```json
{
  "image_b64": "<base64 JPEG/PNG of the 3D viewport>",
  "prompt": "a futuristic city",
  "steps": 4,
  "seed": 42,
  "width": 512,
  "height": 512
}
```

**Response:**
```json
{
  "image_b64": "<base64 PNG>",
  "mime_type": "image/png"
}
```

`strength` and `guidance` are accepted for backward compatibility but the distilled Klein
model does instruction editing rather than strength-based img2img and ignores guidance, so
the frontend no longer exposes a Strength control.

## Notes / possible next steps

- The `Flux2KleinKVPipeline` + `FLUX.2-klein-9b-kv` variant adds KV-cached reference
  conditioning for faster repeated edits — a natural speedup for this realtime loop.
- Caching prompt embeddings across frames (the prompt is usually static during a drag)
  would cut the Qwen3 text-encoder cost per frame.
