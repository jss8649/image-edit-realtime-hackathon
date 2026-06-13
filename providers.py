"""
Optional REMOTE image-edit providers — alternatives to the local in-process Klein.

Select one with IMAGE_GEN_PROVIDER=fal|fireworks (default is the local "klein"
model in pipeline.py). These call a hosted API over HTTP, so no local GPU is
needed when a remote provider is active.

Both take the captured viewport as a PIL image plus the edit prompt and return a
PIL image, so server.py can dispatch uniformly.

Providers / env:
    fal        — fal-ai/flux-2/klein/9b/edit (FLUX.2 Klein, the same model we host
                 locally). Synchronous endpoint, no polling.
                   FAL_KEY     (required)   auth: "Authorization: Key <FAL_KEY>"
                   FAL_MODEL   (default fal-ai/flux-2/klein/9b/edit)
    fireworks  — FLUX.1 Kontext. Async submit-then-poll (Fireworks has no FLUX.2
                 edit endpoint as of 2026-06).
                   FIREWORKS_API_KEY (required, fw_...)   auth: Bearer
                   FIREWORKS_MODEL   (default flux-kontext-pro; or flux-kontext-max)

API shapes verified against fal.ai / docs.fireworks.ai (June 2026).
"""

import asyncio
import base64
import io
import logging
import os

import httpx
from PIL import Image

log = logging.getLogger("providers")

REQUEST_TIMEOUT = 120

# ── FAL config ──
FAL_KEY = os.environ.get("FAL_KEY", "")
FAL_MODEL = os.environ.get("FAL_MODEL", "fal-ai/flux-2/klein/9b/edit")

# ── Fireworks config ──
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", os.environ.get("IMAGE_GEN_API_KEY", ""))
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL", "flux-kontext-pro")
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models"
FIREWORKS_POLL_INTERVAL = 1.0
FIREWORKS_POLL_MAX = 120  # ~120s ceiling


# ── shared helpers ──

def _jpeg_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


async def _url_to_pil(client: httpx.AsyncClient, url: str) -> Image.Image:
    r = await client.get(url)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


async def _result_to_pil(client: httpx.AsyncClient, value: str) -> Image.Image:
    """Normalize a 'url-or-base64' image reference into a PIL image."""
    if value.startswith("http://") or value.startswith("https://"):
        return await _url_to_pil(client, value)
    return _b64_to_pil(value)


def _size_to_aspect(w: int, h: int) -> str:
    """Map width/height to the closest Fireworks/BFL aspect ratio (21:9 … 9:21)."""
    ratio = (w or 1) / (h or 1)
    options = [
        (21 / 9, "21:9"), (16 / 9, "16:9"), (3 / 2, "3:2"), (4 / 3, "4:3"),
        (1 / 1, "1:1"), (3 / 4, "3:4"), (2 / 3, "2:3"), (9 / 16, "9:16"), (9 / 21, "9:21"),
    ]
    return min(options, key=lambda o: abs(o[0] - ratio))[1]


# ── FAL: fal-ai/flux-2/klein/9b/edit (synchronous, no polling) ──

async def fal_generate(img: Image.Image, prompt: str, steps: int = 4,
                       seed=None, width=None, height=None) -> Image.Image:
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is not set")

    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    payload = {
        "prompt": prompt or "",
        "image_urls": [f"data:image/jpeg;base64,{_jpeg_b64(img)}"],
        "num_inference_steps": max(4, min(int(steps or 4), 8)),
        "num_images": 1,
        "output_format": "png",
        "sync_mode": True,  # return the image inline as a data URI (no CDN round-trip)
    }
    if seed is not None and int(seed) >= 0:
        payload["seed"] = int(seed)
    if width and height:
        payload["image_size"] = {"width": int(width), "height": int(height)}

    url = f"https://fal.run/{FAL_MODEL}"
    log.info("FAL request -> %s (steps=%s seed=%s)", FAL_MODEL, payload["num_inference_steps"], seed)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"FAL error {r.status_code}: {r.text[:300]}")
        data = r.json()
        images = data.get("images") or []
        if not images or not images[0].get("url"):
            raise RuntimeError(f"FAL returned no image: keys={list(data.keys())}")
        return await _result_to_pil(client, images[0]["url"])


# ── Fireworks: FLUX.1 Kontext (async submit + poll) ──

async def fireworks_generate(img: Image.Image, prompt: str, steps: int = 4,
                             seed=None, width=None, height=None) -> Image.Image:
    if not FIREWORKS_API_KEY:
        raise RuntimeError("FIREWORKS_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    base = f"{FIREWORKS_BASE}/{FIREWORKS_MODEL}"
    payload = {
        "prompt": prompt or "high quality image",
        "input_image": _jpeg_b64(img),     # bare base64 (no data: prefix)
        "output_format": "png",
        "safety_tolerance": 2,             # max allowed for image-to-image
        "prompt_upsampling": False,
        "aspect_ratio": _size_to_aspect(width, height),
    }
    if seed is not None and int(seed) >= 0:
        payload["seed"] = int(seed)

    log.info("Fireworks submit -> %s (seed=%s)", FIREWORKS_MODEL, seed)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(base, headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"Fireworks submit error {r.status_code}: {r.text[:300]}")
        request_id = r.json().get("request_id")
        if not request_id:
            raise RuntimeError(f"Fireworks: no request_id in response: {r.text[:200]}")

        poll_url = f"{base}/get_result"
        for i in range(FIREWORKS_POLL_MAX):
            pr = await client.post(poll_url, headers=headers, json={"id": request_id})
            if pr.status_code >= 400:
                raise RuntimeError(f"Fireworks poll error {pr.status_code}: {pr.text[:300]}")
            d = pr.json()
            status = d.get("status", "")

            if status == "Ready":
                result = d.get("result") or {}
                sample = result.get("sample") if isinstance(result, dict) else None
                if not sample:
                    raise RuntimeError(f"Fireworks Ready but no result.sample: {result}")
                return await _result_to_pil(client, sample)

            if status in ("Error", "Content Moderated", "Request Moderated"):
                raise RuntimeError(f"Fireworks failed: {status} {d.get('details') or ''}")

            # "Pending" — and "Task not found" can occur briefly before the task
            # registers — keep polling.
            await asyncio.sleep(FIREWORKS_POLL_INTERVAL)

        raise RuntimeError("Fireworks timed out (polling limit reached)")


# Dispatch table for server.py
GENERATORS = {
    "fal": fal_generate,
    "fireworks": fireworks_generate,
}
