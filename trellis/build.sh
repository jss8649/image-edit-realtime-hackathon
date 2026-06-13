#!/usr/bin/env bash
# Build the TRELLIS sidecar image.
#
# Why this isn't a plain `docker build`: TRELLIS's setup.sh gates all CUDA-extension
# installs behind a GPU check, and `docker build` has no GPU — so the extensions get
# skipped ("Unsupported platform: cpu"). We therefore build a base image, then install
# the extensions inside a *running* GPU container and commit the result.
#
# Also note: under CUDA 11.8, nvcc's cicc segfaults compiling nvdiffrast for sm_90, so
# nvdiffrast is built for sm_80+PTX (the H100 driver JITs the PTX at runtime). The other
# extensions build fine for sm_90.
#
# Usage:  sudo ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=trellis-sidecar:latest
KAOLIN_URL=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html

echo "[1/4] build base image (clone TRELLIS, base deps, fastapi, sidecar)…"
docker build -t trellis-sidecar:base .

echo "[2/4] start a GPU container to install the CUDA extensions…"
docker rm -f trellis-build 2>/dev/null || true
docker run -d --gpus all --name trellis-build trellis-sidecar:base sleep infinity

echo "[3/4] install extensions (with the GPU visible)…"
docker exec trellis-build bash -c '
  set -e
  cd /workspace/TRELLIS
  export TORCH_CUDA_ARCH_LIST=9.0
  # wheels + sm_90 source builds (these compile fine on cuda 11.8)
  . ./setup.sh --xformers --spconv --diffoctreerast --mipgaussian || true
  pip install kaolin -f '"$KAOLIN_URL"'
  # nvdiffrast: build for sm_80+PTX to dodge the cuda-11.8 sm_90 cicc crash
  rm -rf /tmp/nvd && git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvd
  TORCH_CUDA_ARCH_LIST="8.0+PTX" pip install --no-build-isolation /tmp/nvd
  # sanity
  python -c "from trellis.pipelines import TrellisImageTo3DPipeline; from trellis.utils import postprocessing_utils; print(\"FULL_IMPORT_OK\")"
'

echo "[4/4] commit -> $IMAGE"
docker commit --change='CMD ["python","-m","uvicorn","sidecar:app","--host","0.0.0.0","--port","8100"]' \
  trellis-build "$IMAGE"
docker rm -f trellis-build

echo "Done. Run with:"
echo "  docker run -d --name trellis --gpus all -p 8100:8100 \\"
echo "    -v ~/.cache/huggingface:/root/.cache/huggingface $IMAGE"
