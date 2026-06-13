"""
TRELLIS image-to-3D sidecar.

Loads microsoft/TRELLIS-image-large once and serves:
    GET  /health        -> readiness
    POST /reconstruct   -> {image_b64} → textured GLB bytes (model/gltf-binary)

Runs in its own CUDA env (Docker), isolated from the main Klein server. The main
server's _reconstruct_3d() POSTs here when TRELLIS_URL is set.
"""
import base64
import io
import logging
import os
import threading

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import torch
from fastapi import FastAPI, Response, HTTPException
from PIL import Image
from pydantic import BaseModel

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("trellis-sidecar")

MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "microsoft/TRELLIS-image-large")
SIMPLIFY = float(os.environ.get("TRELLIS_SIMPLIFY", "0.95"))
TEXTURE_SIZE = int(os.environ.get("TRELLIS_TEXTURE_SIZE", "1024"))

app = FastAPI()
_pipe = None
_lock = threading.Lock()   # one reconstruction at a time on the GPU


class ReconRequest(BaseModel):
    image_b64: str
    seed: int = 1


@app.on_event("startup")
def _load():
    global _pipe
    log.info("Loading %s …", MODEL_ID)
    _pipe = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
    _pipe.cuda()
    if torch.cuda.is_available():
        log.info("TRELLIS ready — VRAM %.1f GB", torch.cuda.memory_allocated() / 1e9)


@app.get("/health")
def health():
    return {"status": "ok", "ready": _pipe is not None, "model": MODEL_ID}


@app.post("/reconstruct")
def reconstruct(req: ReconRequest):
    if _pipe is None:
        raise HTTPException(503, "model not loaded yet")
    try:
        raw = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"bad image: {exc}")

    # Serialize GPU access; rembg background removal runs inside run(preprocess_image=True).
    with _lock:
        log.info("reconstructing (seed=%d) from %dx%d image", req.seed, image.width, image.height)
        outputs = _pipe.run(
            image,
            seed=req.seed,
            formats=["mesh", "gaussian"],
            preprocess_image=True,
        )
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0], outputs["mesh"][0],
            simplify=SIMPLIFY, texture_size=TEXTURE_SIZE,
        )
        data = glb.export(file_type="glb")

    if isinstance(data, str):
        data = data.encode()
    log.info("done — GLB %d bytes", len(data))
    return Response(content=bytes(data), media_type="model/gltf-binary")
