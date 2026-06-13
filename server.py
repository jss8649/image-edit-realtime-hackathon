"""
Realtime image-editing proxy for the 3D canvas frontend.

The frontend captures its WebGL viewport and POSTs it to /generate. By default we
run a single-reference FLUX.2 Klein edit (distilled, 4 steps) IN-PROCESS on the
local H100 — loaded once and kept warm in VRAM, no remote API, no ~3s floor.

The model can instead be hosted REMOTELY via a provider (set IMAGE_GEN_PROVIDER):
    klein      - local in-process FLUX.2 Klein 9B (default; needs a GPU)
    fal        - fal-ai/flux-2/klein/9b/edit (same model, hosted)   [needs FAL_KEY]
    fireworks  - FLUX.1 Kontext via Fireworks   [needs FIREWORKS_API_KEY]
    echo       - mirror the capture back (no GPU, UI smoke test)
Remote providers run on the event loop (no GPU lock); the local model serializes
GPU access with one asyncio.Lock and runs inference in a worker thread.

Env vars:
    IMAGE_GEN_PROVIDER - klein (default) | fal | fireworks | echo
    KLEIN_MODEL_ID     - local HF repo id (default black-forest-labs/FLUX.2-klein-9B)
    KLEIN_ECHO         - "1" to force echo mode
    KLEIN_CPU_OFFLOAD  - "1" to use model CPU offload (fallback if VRAM is tight)
    KLEIN_NO_WARMUP    - "1" to skip the startup warmup inference
    FAL_KEY / FAL_MODEL                 - FAL provider config
    FIREWORKS_API_KEY / FIREWORKS_MODEL - Fireworks provider config
    PORT               - server port (default 3000)

Usage:
    pip install -r requirements.txt
    huggingface-cli login          # local weights are gated — accept the license
    python server.py               # or: IMAGE_GEN_PROVIDER=fal FAL_KEY=... python server.py
"""

import base64
import io
import logging
import os
import pathlib
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from functools import partial

from anyio import to_thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import pipeline
import providers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

PORT = int(os.environ.get("PORT", "3000"))

# Resolve the active backend.
PROVIDER = os.environ.get("IMAGE_GEN_PROVIDER", "klein").lower()
_FORCE_ECHO = os.environ.get("KLEIN_ECHO", "").lower() in ("1", "true", "yes")

if _FORCE_ECHO or PROVIDER == "echo":
    MODE = "echo"
elif PROVIDER in ("fal", "fireworks"):
    MODE = PROVIDER
elif PROVIDER == "klein":
    MODE = "klein" if pipeline.is_available() else "echo"  # echo fallback when no GPU
else:
    log.warning("Unknown IMAGE_GEN_PROVIDER=%r — falling back to echo", PROVIDER)
    MODE = "echo"

# One H100 → one inference at a time.
import asyncio
gpu_lock = asyncio.Lock()


# ── Request / Response models (unchanged contract with the frontend) ──

class GenerateRequest(BaseModel):
    image_b64: str
    prompt: str = ""
    strength: float = 0.75          # retained for compatibility; not used by Klein
    steps: int = 4
    seed: int = 42
    width: int = 512
    height: int = 512
    guidance: float = 1.0           # passthrough; distilled Klein ignores guidance


class GenerateResponse(BaseModel):
    image_b64: str
    mime_type: str = "image/jpeg"


class Generate3DRequest(BaseModel):
    image_b64: str


# ── base64 <-> PIL helpers ──

def _decode_image(image_b64: str) -> Image.Image:
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[-1]
    try:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Could not decode input image: {exc}")


def _encode_image(img: Image.Image) -> str:
    # JPEG keeps the response small — important over a tunnel and for the webcam loop.
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ── Lifespan: load + warm the model once ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODE == "klein":
        log.info("Loading FLUX.2 Klein into VRAM (this can take a while on first run)…")
        await to_thread.run_sync(pipeline.load)
        if os.environ.get("KLEIN_NO_WARMUP", "").lower() not in ("1", "true", "yes"):
            await to_thread.run_sync(pipeline.warmup)
        log.info("Ready — model warm in VRAM.")
    elif MODE in ("fal", "fireworks"):
        log.info("Using REMOTE provider '%s' — no local model loaded.", MODE)
    else:
        log.warning("Running in ECHO mode (input mirrored back, no generation).")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve bundled demo 3D assets at /assets (the frontend's "Demo scene" loads these).
_ASSETS_DIR = pathlib.Path(__file__).resolve().parent / "assets"
if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")


_INDEX_HTML = pathlib.Path(__file__).resolve().parent / "index.html"


@app.get("/")
async def index():
    # Serve the UI from the backend so a remote box just needs http://<box>:PORT/
    return FileResponse(_INDEX_HTML)


@app.get("/healthz")
async def health():
    model = {
        "klein": pipeline.MODEL_ID,
        "fal": providers.FAL_MODEL,
        "fireworks": providers.FIREWORKS_MODEL,
    }.get(MODE, "—")
    return {"status": "ok", "mode": MODE, "model": model, "busy": gpu_lock.locked()}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    img = _decode_image(req.image_b64)

    # Echo mode — no GPU, just mirror the capture back (UI smoke test).
    if MODE == "echo":
        return GenerateResponse(image_b64=_encode_image(img))

    # Local in-process Klein — serialize GPU access, run off the event loop.
    if MODE == "klein":
        work = partial(
            pipeline.generate, img, req.prompt,
            steps=req.steps, seed=req.seed, guidance=req.guidance,
            width=req.width, height=req.height,
        )
        async with gpu_lock:
            try:
                out = await to_thread.run_sync(work)
            except Exception as exc:
                log.exception("Generation failed")
                raise HTTPException(500, f"Generation failed: {exc}")
        return GenerateResponse(image_b64=_encode_image(out))

    # Remote provider (fal / fireworks) — async HTTP, no GPU lock.
    generate_fn = providers.GENERATORS[MODE]
    try:
        out = await generate_fn(
            img, req.prompt, steps=req.steps, seed=req.seed,
            width=req.width, height=req.height,
        )
    except Exception as exc:
        log.exception("%s generation failed", MODE)
        raise HTTPException(502, f"{MODE} generation failed: {exc}")
    return GenerateResponse(image_b64=_encode_image(out))


# ── Image → 3D object job API ───────────────────────────────────────────────
# POST /generate-3d  -> {job_id}
# GET  /jobs/{id}     -> {status: queued|running|done|error}
# GET  /jobs/{id}/model.glb -> the generated GLB
#
# Reconstruction is currently STUBBED (returns a sample asset after a short delay).
# To go live, replace _reconstruct_3d's body with a call to the TRELLIS sidecar
# (POST the image to the TRELLIS service, write its GLB to out_path) — nothing else
# in this file or the frontend needs to change.

_JOBS: dict = {}
_JOBS_DIR = pathlib.Path(tempfile.gettempdir()) / "klein_3d_jobs"
_JOBS_DIR.mkdir(exist_ok=True)
_job_sema = asyncio.Semaphore(1)          # one reconstruction at a time (shares the GPU)
_STUB_ASSETS = ["pottedPlant.glb", "loungeChair.glb", "loungeSofa.glb"]
_stub_counter = {"n": 0}


def _reconstruct_3d(image: Image.Image, out_path: str):
    """Image -> textured GLB at out_path. STUB: simulate latency + return a sample asset.

    Real impl (TRELLIS sidecar): POST `image` to the TRELLIS service and stream its
    GLB bytes to out_path. Keep this function blocking — it runs in a worker thread.
    """
    import time
    time.sleep(3)  # simulate reconstruction latency
    pick = _STUB_ASSETS[_stub_counter["n"] % len(_STUB_ASSETS)]
    _stub_counter["n"] += 1
    shutil.copyfile(_ASSETS_DIR / pick, out_path)


async def _run_3d_job(job_id: str, image: Image.Image):
    async with _job_sema:
        _JOBS[job_id]["status"] = "running"
        out = str(_JOBS_DIR / f"{job_id}.glb")
        try:
            await to_thread.run_sync(partial(_reconstruct_3d, image, out))
            _JOBS[job_id].update(status="done", glb=out)
            log.info("3D job %s done -> %s", job_id, out)
        except Exception as exc:
            log.exception("3D reconstruction failed")
            _JOBS[job_id].update(status="error", error=str(exc))


@app.post("/generate-3d")
async def generate_3d(req: Generate3DRequest):
    img = _decode_image(req.image_b64)
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "queued", "glb": None, "error": None}
    asyncio.create_task(_run_3d_job(job_id, img))
    log.info("3D job %s queued", job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return {"job_id": job_id, "status": job["status"], "error": job["error"]}


@app.get("/jobs/{job_id}/model.glb")
async def job_model(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] != "done":
        raise HTTPException(409, f"job not ready (status={job['status']})")
    return FileResponse(job["glb"], media_type="model/gltf-binary", filename="object.glb")


if __name__ == "__main__":
    import uvicorn
    log.info("Starting server on port %d (provider=%s)", PORT, MODE)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
