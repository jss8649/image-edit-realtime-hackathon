"""
Realtime image-editing proxy backed by an IN-PROCESS FLUX.2 Klein 9B model.

The 3D frontend captures its WebGL viewport and POSTs it to /generate. We decode
it to a PIL image, run a single-reference FLUX.2 Klein edit (distilled, 4 steps)
on the local H100, and return the result as base64. The model is loaded once at
startup and kept warm in VRAM — no remote API, no submit-then-poll, no ~3s floor.

GPU access is serialized with a single asyncio.Lock (one H100 = one inference at a
time) and the blocking inference runs in a worker thread so the event loop stays
responsive for cancellations.

Env vars:
    KLEIN_MODEL_ID    - HF repo id (default black-forest-labs/FLUX.2-klein-9B)
    KLEIN_ECHO        - "1" to force echo mode (mirror input back, no GPU)
    KLEIN_CPU_OFFLOAD - "1" to use model CPU offload (fallback if VRAM is tight)
    KLEIN_NO_WARMUP   - "1" to skip the startup warmup inference
    PORT              - server port (default 8000)

Usage:
    pip install -r requirements.txt
    huggingface-cli login          # weights are gated — accept the license first
    python server.py
"""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from functools import partial

from anyio import to_thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

PORT = int(os.environ.get("PORT", "8000"))
ECHO_MODE = (
    os.environ.get("KLEIN_ECHO", "").lower() in ("1", "true", "yes")
    or not pipeline.is_available()
)

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
    mime_type: str = "image/png"


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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Lifespan: load + warm the model once ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    if ECHO_MODE:
        if not pipeline.is_available():
            log.warning("CUDA not available — running in ECHO mode (input mirrored back).")
        else:
            log.warning("KLEIN_ECHO set — running in ECHO mode (input mirrored back).")
    else:
        log.info("Loading FLUX.2 Klein into VRAM (this can take a while on first run)…")
        await to_thread.run_sync(pipeline.load)
        if os.environ.get("KLEIN_NO_WARMUP", "").lower() not in ("1", "true", "yes"):
            await to_thread.run_sync(pipeline.warmup)
        log.info("Ready — model warm in VRAM.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "ok", "mode": "echo" if ECHO_MODE else "klein",
            "model": pipeline.MODEL_ID, "busy": gpu_lock.locked()}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    img = _decode_image(req.image_b64)

    # Echo mode — no GPU, just mirror the capture back (UI smoke test).
    if ECHO_MODE:
        return GenerateResponse(image_b64=_encode_image(img))

    work = partial(
        pipeline.generate,
        img,
        req.prompt,
        steps=req.steps,
        seed=req.seed,
        guidance=req.guidance,
        width=req.width,
        height=req.height,
    )

    # Serialize GPU access; run the blocking inference off the event loop.
    async with gpu_lock:
        try:
            out = await to_thread.run_sync(work)
        except Exception as exc:
            log.exception("Generation failed")
            raise HTTPException(500, f"Generation failed: {exc}")

    return GenerateResponse(image_b64=_encode_image(out))


if __name__ == "__main__":
    import uvicorn
    if ECHO_MODE:
        log.warning("Starting in ECHO mode on port %d", PORT)
    else:
        log.info("Starting Klein server on port %d (model=%s)", PORT, pipeline.MODEL_ID)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
