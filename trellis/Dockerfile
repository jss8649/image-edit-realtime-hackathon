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

# Base Python deps only. The CUDA extensions (spconv/xformers/kaolin/nvdiffrast/
# diffoctreerast/mipgaussian) require a GPU, which `docker build` doesn't expose —
# so build.sh installs them in a running GPU container and commits the result.
RUN . ./setup.sh --basic

RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY sidecar.py /workspace/TRELLIS/sidecar.py

EXPOSE 8100
CMD ["python", "-m", "uvicorn", "sidecar:app", "--host", "0.0.0.0", "--port", "8100"]
