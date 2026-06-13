# Session Log — 2026-06-13

A build-day session that turned this repo from a remote-API image-edit demo into a
self-hosted, realtime 3D→AI canvas with webcam input and photo→3D object creation,
all running on a single H100.

> Note: API keys / tokens used during the session (Hugging Face, FAL) are intentionally
> **omitted** from this log — this is a public repo.

## Starting point
A browser 3D canvas (`index.html`, Three.js) that captured the WebGL viewport and sent it
to a FastAPI proxy (`server.py`) which forwarded to **Fireworks FLUX Kontext** over an async
submit-then-poll flow (~3s floor → not realtime).

## What we built (in order)

### 1. Self-hosted realtime FLUX.2 Klein  — `960fdd0`, `6f3d25e`
- Replaced the remote Fireworks call with an **in-process FLUX.2 Klein 9B** (distilled, 4-step)
  loaded once and kept warm in VRAM. Verified live docs first: `Flux2KleinPipeline`,
  `black-forest-labs/FLUX.2-klein-9B`, single-reference editing via `image=`, bf16, `.to("cuda")`.
- `pipeline.py` (warm loader) + `server.py` rewrite: decode b64 → PIL → inference in a worker
  thread serialized by one `asyncio.Lock` → re-encode. Echo fallback kept.
- Frontend realtime: wired the dead debounce into `markChanged` (auto-regen ~300ms after you
  stop), `AbortController` cancellation, grid-hidden JPEG capture, composition-prefix prompt,
  locked seed, honest param wiring (Strength hidden — distilled Klein ignores guidance).
- **Measured ~0.6s warm round-trip on the H100**; GPU lock serializes concurrent requests.

### 2. Serve the UI from the backend + port 3000  — `64b237a`
- Server serves `index.html` at `/` (health moved to `/healthz`); frontend defaults its API
  base to the page origin → a remote box just needs `http://<box>:3000/`. Default port 3000.

### 3. Optional remote providers (FAL, Fireworks)  — `7742dc0`
- `providers.py`: async FAL (`fal-ai/flux-2/klein/9b/edit`, sync endpoint) + Fireworks (FLUX.1
  Kontext, submit/poll). Selectable via `IMAGE_GEN_PROVIDER`; local Klein stays default.
- **FAL verified live**; chose it over Hunyuan3D etc. partly on license.

### 4. Bundled demo scene + Krea prompt  — `500fb19`
- Kenney "Furniture Kit" (CC0) glbs (chair/sofa/plant/rug) served at `/assets`; one-click
  "demo scene" loader arranges them; default prompt mirrors the Krea reference.

### 5. Sharper output  — `02cbb26`
- Default output 512→**1024** (FLUX.2's native ~1MP). ~1.6s/frame; much crisper.

### 6. Webcam input (Krea-style realtime edit)  — `6974527`
- `getUserMedia` mirrored preview + a **continuous generation loop** at 512 for fps; robust
  lifecycle (re-entrancy guard + generation token, track cleanup, error backoff, frame-ready
  skip). Removed the "Generating…" overlay (previous frame holds → reads as a feed).
- Per-button default prompts (demo = interior; webcam = "Make it claymation").
- Server returns **JPEG** (~10× smaller than PNG) — big win over the Cloudflare tunnel used to
  share the app for remote camera access (HTTPS required for `getUserMedia`).
- An **adversarial multi-agent review** of the diff caught a camera-leak/abort-storm on rapid
  toggle and a no-backoff error loop — both fixed before commit.

### 7. Image → 3D objects  — `1a73439` (stub), `891ba92` (live)
- Researched iPhone/ARKit capture (needs a native app; outputs USDZ) → pivoted to a
  **self-hosted image-to-3D model**.
- Prototyped the flow first with a stub: job API (`POST /generate-3d` → poll `/jobs/{id}` →
  `GET …/model.glb`) + a "Create 3D from image" button (webcam frame or photo upload →
  auto-import the GLB).
- Stood up **microsoft/TRELLIS** (`TRELLIS-image-large`, MIT) as a Docker sidecar (`trellis/`).
  The install fought back — both fixes are documented in `trellis/build.sh`:
  1. `setup.sh` skips all CUDA extensions when no GPU is visible, and `docker build` has none
     → install them in a *running* `--gpus` container, then `docker commit`.
  2. `cicc` segfaults compiling **nvdiffrast** for `sm_90` under CUDA 11.8 → build it for
     **`sm_80+PTX`** (the H100 JITs the PTX at runtime).
- **Live: photo → TRELLIS → textured GLB (~15s, ~2.3k verts) imported into the scene.**

### 8. USDZ import  — `5ae025f`, `891ba92`
- Added `.usdz` to the importer. three.js's `USDZLoader` can't read **binary USDC** (what most
  `.usdz` are), so the server converts to GLB with **`usd2gltf`** (`/convert-usdz`); the frontend
  routes `.usdz` uploads through it. Verified on Apple's USDC teapot sample.

## Architecture (end state)
```
Browser (index.html, Three.js)
  ├─ realtime edit loop ─► POST /generate  ─► FLUX.2 Klein (in-process, H100)  [or FAL/Fireworks]
  ├─ Create-3D button ──► POST /generate-3d ─► TRELLIS sidecar (Docker :8100) ─► GLB
  └─ .usdz upload ──────► POST /convert-usdz ─► usd2gltf ─► GLB
server.py (FastAPI :3000) serves the UI, the APIs, /assets, and /healthz
```

## Repo layout
- `server.py` — FastAPI: realtime `/generate`, image→3D job API, `/convert-usdz`, static UI/assets.
- `pipeline.py` — warm in-process FLUX.2 Klein loader.
- `providers.py` — FAL / Fireworks remote backends.
- `index.html` — the whole frontend (Three.js canvas, realtime loop, webcam, create-3D, USDZ).
- `trellis/` — `Dockerfile`, `build.sh`, `sidecar.py` for the TRELLIS image→3D service.
- `assets/` — bundled CC0 demo furniture.

## Commit history
```
891ba92  TRELLIS image-to-3D live + server-side USDZ import
502f43d  Add TRELLIS image-to-3D sidecar (Dockerfile + service); wire server to it
5ae025f  Support importing USDZ files
1a73439  Prototype image-to-3D flow (job API + Create-3D UI, stubbed reconstruction)
6974527  Add webcam input source + realtime polish
02cbb26  Default to 1024×1024 output for sharper results
500fb19  Add bundled demo scene (CC0 furniture) + Krea default prompt
7742dc0  Add optional remote model providers (FAL, Fireworks)
64b237a  Serve UI from backend; default to port 3000
6f3d25e  Realtime auto-regen with cancellation + clean scene capture
960fdd0  Self-host FLUX.2 Klein 9B in-process; drop remote Fireworks
1046dd7  initial
```

## Hardware / models
- 1× NVIDIA H100 80GB. FLUX.2 Klein ~35GB resident; TRELLIS ~6GB — both co-resident.
- Models: `black-forest-labs/FLUX.2-klein-9B` (image edit), `microsoft/TRELLIS-image-large`
  (image→3D), `usd2gltf` (USDZ→GLB), Kenney Furniture Kit (CC0 demo assets).
