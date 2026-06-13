"""
In-process FLUX.2 Klein 9B (distilled, 4-step) image-editing pipeline.

The model is loaded ONCE at startup and kept warm in VRAM. `generate()` performs
single-reference image editing: the captured 3D viewport is passed as the
reference image (`image=`), and `prompt` describes the desired look. The model is
step-distilled to 4 inference steps for (near) realtime use.

Target hardware: 1x NVIDIA H100 80GB, bf16, no quantization — the ~9B flow model
plus the Qwen3 text encoder are ~34GB in bf16, so everything stays resident on the
GPU (`.to("cuda")`). CPU offload is available as a fallback via KLEIN_CPU_OFFLOAD.

Refs:
  - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B  (distilled 4-step)
  - https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2
"""

import logging
import os

import torch
from PIL import Image

log = logging.getLogger("pipeline")

MODEL_ID = os.environ.get("KLEIN_MODEL_ID", "black-forest-labs/FLUX.2-klein-9B")
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Reference settings for the distilled Klein model (per model card / docs).
# It is step-distilled, so guidance_scale is effectively ignored — kept as a
# pass-through knob only.
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0

# Snap requested output dims to something the VAE/patchifier is happy with.
_DIM_MULTIPLE = 32
_DIM_MIN, _DIM_MAX = 256, 2048

_pipe = None


def is_available() -> bool:
    return torch.cuda.is_available()


def _snap(x: int) -> int:
    x = int(round(x / _DIM_MULTIPLE) * _DIM_MULTIPLE)
    return max(_DIM_MIN, min(_DIM_MAX, x))


def load():
    """Load the pipeline once and keep it warm. Returns the cached pipeline."""
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import Flux2KleinPipeline  # imported lazily so echo mode needs no GPU stack

    log.info("Loading %s (dtype=%s)…", MODEL_ID, DTYPE)
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)

    if os.environ.get("KLEIN_CPU_OFFLOAD", "").lower() in ("1", "true", "yes"):
        log.warning("KLEIN_CPU_OFFLOAD set — enabling model CPU offload (higher latency)")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)

    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        log.info("Klein loaded — VRAM allocated %.1f GB / reserved %.1f GB",
                 torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)
    return _pipe


def warmup(width: int = 512, height: int = 512):
    """Run one throwaway inference so the first real request isn't slow."""
    log.info("Warming up the pipeline…")
    dummy = Image.new("RGB", (width, height), (30, 30, 50))
    generate(dummy, "warmup", steps=DEFAULT_STEPS, seed=0,
             guidance=DEFAULT_GUIDANCE, width=width, height=height)
    log.info("Warmup complete.")


def generate(image: Image.Image, prompt: str, steps: int = DEFAULT_STEPS,
             seed: int = 42, guidance: float = DEFAULT_GUIDANCE,
             width=None, height=None) -> Image.Image:
    """Single-reference image edit. `image` is the captured viewport (reference),
    `prompt` is the edit instruction. Returns a PIL image.

    NOTE: synchronous / blocking — call from a threadpool so the event loop stays
    responsive, and serialize calls (one H100 = one inference at a time).
    """
    pipe = load()

    if image.mode != "RGB":
        image = image.convert("RGB")

    w = _snap(width or image.width)
    h = _snap(height or image.height)

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    log.info("generate: steps=%d seed=%d guidance=%.2f out=%dx%d ref=%dx%d prompt=%r",
             int(steps), int(seed), float(guidance), w, h, image.width, image.height, prompt)

    result = pipe(
        prompt=prompt or "",
        image=image,                      # single reference image for editing
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),   # ignored by the distilled model, passed for completeness
        width=w,
        height=h,
        generator=generator,
    )
    return result.images[0]
