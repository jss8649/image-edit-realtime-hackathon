# Realtime 3D Canvas → Image Generation

A browser-based 3D canvas where you can import, position, rotate, and scale 3D objects, then send the viewport capture to an AI image generation backend.

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi uvicorn httpx
```

### 2. Start the proxy server

**Fireworks AI** (auto-detected from `fw_` key prefix):

```bash
IMAGE_GEN_API_KEY=fw_your-key python server.py
```

**Echo mode** (no API key — mirrors the canvas capture back for testing):

```bash
python server.py
```

**Generic upstream API:**

```bash
IMAGE_GEN_URL=https://api.example.com/generate \
IMAGE_GEN_API_KEY=sk-your-key \
python server.py
```

### 3. Open the frontend

Open `index.html` in your browser. The frontend connects to `http://localhost:8000/generate`.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `IMAGE_GEN_API_KEY` | No | Bearer token. Keys starting with `fw_` auto-select the Fireworks provider. |
| `IMAGE_GEN_PROVIDER` | No | `fireworks` or `generic` (default). Auto-detected from key prefix. |
| `IMAGE_GEN_URL` | No | Upstream endpoint for the generic provider. If unset, runs in echo mode. |
| `PORT` | No | Server port (default `8000`). |

## Supported Providers

- **Fireworks AI** — uses FLUX Kontext Pro for image-to-image editing (async submit + poll)
- **Generic** — forwards to any API and normalizes responses from OpenAI, Stability AI, Replicate, or similar formats

## API

### `POST /generate`

**Request:**
```json
{
  "image_b64": "<base64 PNG of the 3D viewport>",
  "prompt": "a futuristic city",
  "strength": 0.75,
  "steps": 20,
  "seed": 42,
  "width": 512,
  "height": 512
}
```

**Response:**
```json
{
  "image_b64": "<base64 image data>",
  "mime_type": "image/png"
}
```
