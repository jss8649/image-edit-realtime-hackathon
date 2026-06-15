# Realtime Canvas — single-container app: FastAPI + in-process FLUX.2 Klein 9B,
# serves the UI, and converts USDZ via headless Blender. (No TRELLIS sidecar — the
# image→3D endpoint falls back to a stub when TRELLIS_URL is unset.)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

# git: pip-install diffusers from main. The rest are Blender's headless X11 deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates xz-utils \
      libgl1 libglib2.0-0 libsm6 libice6 libxxf86vm1 libxfixes3 libxi6 \
      libxkbcommon0 libxrender1 libxrandr2 libxinerama1 libxcursor1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (torch is already in the base image and satisfies requirements).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Headless Blender — the robust USDZ→GLB converter.
RUN curl -sSL -o /tmp/blender.tar.xz \
      https://download.blender.org/release/Blender4.2/blender-4.2.21-linux-x64.tar.xz \
 && tar xf /tmp/blender.tar.xz -C /opt && rm /tmp/blender.tar.xz \
 && mv /opt/blender-4.2.21-linux-x64 /opt/blender
ENV BLENDER_BIN=/opt/blender/blender

# App code + bundled demo assets.
COPY pipeline.py providers.py server.py usdz_convert.py index.html ./
COPY assets ./assets

EXPOSE 3000
CMD ["python", "server.py"]
