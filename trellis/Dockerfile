# TRELLIS image-to-3D sidecar — isolated CUDA env (torch 2.4 / CUDA 11.8, TRELLIS's
# tested config) so it doesn't clash with the main Klein server (torch 2.7 / CUDA 12.8).
# H100 is sm_90; CUDA 11.8 supports Hopper.
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ATTN_BACKEND=xformers \
    SPCONV_ALGO=native \
    TORCH_CUDA_ARCH_LIST="9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build libgl1 libglib2.0-0 libegl1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
WORKDIR /workspace/TRELLIS

# Build TRELLIS's CUDA extensions into the base env (no --new-env). Use xformers
# (prebuilt) instead of flash-attn to avoid a very long compile.
RUN . ./setup.sh --basic --xformers --spconv --nvdiffrast --diffoctreerast --vox2seq --mipgaussian --kaolin

RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY sidecar.py /workspace/TRELLIS/sidecar.py

EXPOSE 8100
CMD ["python", "-m", "uvicorn", "sidecar:app", "--host", "0.0.0.0", "--port", "8100"]
