"""
Image generation proxy with Fireworks AI support.

Accepts a canvas capture + prompt from the frontend, forwards to an
upstream image-gen API, normalizes the response.

Env vars:
    IMAGE_GEN_URL      - upstream API endpoint (if unset, runs in echo mode)
    IMAGE_GEN_API_KEY  - bearer token for upstream
    IMAGE_GEN_PROVIDER - "fireworks" for Fireworks-specific handling, or
                         "generic" (default) for pass-through
    PORT               - server port (default 8000)

Fireworks shortcut — set only the API key:
    IMAGE_GEN_API_KEY=fw_xxx IMAGE_GEN_PROVIDER=fireworks python server.py

Usage:
    pip install fastapi uvicorn httpx
    python server.py
"""

import asyncio
import base64
import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("proxy")

IMAGE_GEN_URL = os.environ.get("IMAGE_GEN_URL", "")
IMAGE_GEN_API_KEY = os.environ.get("IMAGE_GEN_API_KEY", "")
IMAGE_GEN_PROVIDER = os.environ.get("IMAGE_GEN_PROVIDER", "generic").lower()
PORT = int(os.environ.get("PORT", "8000"))
UPSTREAM_TIMEOUT = 120

# Auto-detect Fireworks from API key prefix
if IMAGE_GEN_API_KEY.startswith("fw_") and IMAGE_GEN_PROVIDER == "generic":
    IMAGE_GEN_PROVIDER = "fireworks"

# Fireworks defaults
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models"
FIREWORKS_KONTEXT_MODEL = "flux-kontext-pro"
FIREWORKS_POLL_INTERVAL = 4  # seconds between polls
FIREWORKS_POLL_MAX = 30  # max polls (~120s)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    image_b64: str
    prompt: str = ""
    strength: float = 0.75
    steps: int = 20
    seed: int = 42
    width: int = 512
    height: int = 512


class GenerateResponse(BaseModel):
    image_b64: str
    mime_type: str = "image/png"


# ── Helpers ──

def _detect_mime(b64_data: str) -> str:
    try:
        header = base64.b64decode(b64_data[:32] + "==")
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if header[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"
    except Exception:
        pass
    return "image/png"


def _extract_image(data: dict) -> Optional[str]:
    """Extract base64 image from common response shapes."""
    for key in ("image_b64", "image"):
        val = data.get(key)
        if isinstance(val, str) and len(val) > 100:
            if val.startswith("data:"):
                val = val.split(",", 1)[-1]
            return val

    images = data.get("images")
    if isinstance(images, list) and images:
        val = images[0]
        if isinstance(val, str) and len(val) > 100:
            return val

    # Fireworks: { base64: ["..."] }
    base64_arr = data.get("base64")
    if isinstance(base64_arr, list) and base64_arr:
        val = base64_arr[0]
        if isinstance(val, str):
            return val

    # OpenAI: { data: [{ b64_json: "..." }] }
    data_arr = data.get("data")
    if isinstance(data_arr, list) and data_arr:
        item = data_arr[0]
        if isinstance(item, dict):
            val = item.get("b64_json") or item.get("b64")
            if isinstance(val, str):
                return val

    # Stability AI: { artifacts: [{ base64: "..." }] }
    artifacts = data.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        item = artifacts[0]
        if isinstance(item, dict):
            val = item.get("base64")
            if isinstance(val, str):
                return val

    # Replicate: { output: "..." }
    output = data.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        return output[0]

    # Fireworks Kontext get_result: { result: { sample: "url" } }
    result = data.get("result")
    if isinstance(result, dict):
        sample = result.get("sample")
        if isinstance(sample, str):
            return sample
        return _extract_image(result)

    return None


async def _fetch_url_as_b64(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "image/png").split(";")[0]
        return base64.b64encode(resp.content).decode(), mime


def _size_to_aspect(w: int, h: int) -> str:
    """Map width/height to the closest Fireworks aspect ratio."""
    ratio = w / h
    options = [
        (21 / 9, "21:9"), (16 / 9, "16:9"), (3 / 2, "3:2"), (5 / 4, "5:4"),
        (4 / 3, "4:3"), (1 / 1, "1:1"), (3 / 4, "3:4"), (4 / 5, "4:5"),
        (2 / 3, "2:3"), (9 / 16, "9:16"), (9 / 21, "9:21"),
    ]
    return min(options, key=lambda x: abs(x[0] - ratio))[1]


# ── Fireworks Provider ──

async def _fireworks_generate(req: GenerateRequest, client: httpx.AsyncClient) -> GenerateResponse:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {IMAGE_GEN_API_KEY}",
    }

    # Use Kontext for image-to-image editing
    model = FIREWORKS_KONTEXT_MODEL
    url = f"{FIREWORKS_BASE}/{model}"

    payload = {
        "prompt": req.prompt or "high quality image",
        "input_image": req.image_b64,
        "seed": req.seed if req.seed > 0 else 42,
        "aspect_ratio": _size_to_aspect(req.width, req.height),
        "output_format": "png",
    }

    log.info("Fireworks Kontext request to %s (prompt=%r)", url, req.prompt)

    resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code >= 400:
        log.error("Fireworks error %d: %s", resp.status_code, resp.text[:500])
        raise HTTPException(resp.status_code, f"Fireworks error: {resp.text[:500]}")

    data = resp.json()

    # Kontext is async — returns { request_id: "..." }
    request_id = data.get("request_id")
    if request_id:
        log.info("Fireworks async request_id=%s — polling for result", request_id)
        return await _fireworks_poll(request_id, model, headers, client)

    # Synchronous response (e.g. schnell) — extract directly
    image_val = _extract_image(data)
    if not image_val:
        raise HTTPException(502, f"Could not extract image from Fireworks response: {list(data.keys())}")

    if image_val.startswith("http"):
        image_b64, mime = await _fetch_url_as_b64(image_val)
        return GenerateResponse(image_b64=image_b64, mime_type=mime)

    return GenerateResponse(image_b64=image_val, mime_type=_detect_mime(image_val))


async def _fireworks_poll(
    request_id: str, model: str, headers: dict, client: httpx.AsyncClient
) -> GenerateResponse:
    """Poll Fireworks get_result until the image is ready."""
    poll_url = f"{FIREWORKS_BASE}/{model}/get_result"

    # Wait before the first poll — task needs time to register
    await asyncio.sleep(3)

    for i in range(FIREWORKS_POLL_MAX):
        resp = await client.post(poll_url, json={"id": request_id}, headers=headers)
        if resp.status_code == 429:
            # Rate limited — back off and retry
            retry_after = int(resp.headers.get("retry-after", "10"))
            log.warning("Poll %d: rate limited, waiting %ds", i + 1, retry_after)
            await asyncio.sleep(retry_after)
            continue
        if resp.status_code >= 400:
            raise HTTPException(502, f"Fireworks poll error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        status = data.get("status", "")
        log.info("Poll %d: status=%s", i + 1, status)

        if status in ("Pending", "Task not found"):
            # "Task not found" early on means the task hasn't registered yet
            await asyncio.sleep(FIREWORKS_POLL_INTERVAL)
            continue

        if status == "Ready":
            result = data.get("result")
            log.info("Result type=%s, preview=%s", type(result).__name__,
                     str(result)[:200] if result else "None")

            image_val = _extract_image(data)

            # result might be a direct base64 string or URL
            if not image_val and isinstance(result, str):
                image_val = result
            # result might be a list of URLs or base64 strings
            if not image_val and isinstance(result, list) and result:
                image_val = result[0] if isinstance(result[0], str) else None

            if not image_val:
                raise HTTPException(502, f"Image ready but could not extract. result type={type(result).__name__}, keys={list(data.keys())}")

            if image_val.startswith("http"):
                image_b64, mime = await _fetch_url_as_b64(image_val)
                return GenerateResponse(image_b64=image_b64, mime_type=mime)

            return GenerateResponse(image_b64=image_val, mime_type=_detect_mime(image_val))

        if status in ("Error", "Content Moderated", "Request Moderated"):
            details = data.get("details") or data.get("error_message") or status
            raise HTTPException(502, f"Fireworks generation failed: {details}")

    raise HTTPException(504, "Fireworks generation timed out (polling limit reached)")


# ── Generic Provider ──

async def _generic_generate(req: GenerateRequest, client: httpx.AsyncClient) -> GenerateResponse:
    headers = {"Content-Type": "application/json"}
    if IMAGE_GEN_API_KEY:
        headers["Authorization"] = f"Bearer {IMAGE_GEN_API_KEY}"

    payload = {
        "image_b64": req.image_b64,
        "image": req.image_b64,
        "prompt": req.prompt,
        "strength": req.strength,
        "steps": req.steps,
        "seed": req.seed,
        "width": req.width,
        "height": req.height,
    }

    resp = await client.post(IMAGE_GEN_URL, json=payload, headers=headers)

    if resp.status_code >= 400:
        log.error("Upstream error %d: %s", resp.status_code, resp.text[:500])
        raise HTTPException(resp.status_code, f"Upstream error: {resp.text[:500]}")

    try:
        upstream_data = resp.json()
    except Exception:
        raise HTTPException(502, "Upstream returned non-JSON response")

    image_val = _extract_image(upstream_data)
    if not image_val:
        raise HTTPException(502, "Could not extract image from upstream response")

    if image_val.startswith("http://") or image_val.startswith("https://"):
        image_b64, mime = await _fetch_url_as_b64(image_val)
        return GenerateResponse(image_b64=image_b64, mime_type=mime)

    return GenerateResponse(image_b64=image_val, mime_type=_detect_mime(image_val))


# ── Endpoint ──

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    log.info(
        "Request: prompt=%r strength=%.2f steps=%d seed=%d size=%dx%d image=%d chars",
        req.prompt, req.strength, req.steps, req.seed,
        req.width, req.height, len(req.image_b64),
    )

    # Echo mode
    if not IMAGE_GEN_URL and IMAGE_GEN_PROVIDER != "fireworks":
        log.warning("IMAGE_GEN_URL not set — running in echo mode")
        return GenerateResponse(
            image_b64=req.image_b64,
            mime_type=_detect_mime(req.image_b64),
        )

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        try:
            if IMAGE_GEN_PROVIDER == "fireworks":
                return await _fireworks_generate(req, client)
            else:
                return await _generic_generate(req, client)
        except HTTPException:
            raise
        except httpx.TimeoutException:
            raise HTTPException(504, "Upstream timed out")
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Upstream connection error: {exc}")


if __name__ == "__main__":
    if IMAGE_GEN_PROVIDER == "fireworks":
        log.info("Provider: Fireworks AI (model=%s)", FIREWORKS_KONTEXT_MODEL)
    elif not IMAGE_GEN_URL:
        log.warning("IMAGE_GEN_URL not set — server will run in echo mode")
    else:
        log.info("Provider: generic, proxying to %s", IMAGE_GEN_URL)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
