#!/usr/bin/env bash
# First-start installer for the public official PyTorch image.
# Used when GHCR cannot be pulled without a registry credential.
# No secrets here.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/opt/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}"
mkdir -p /app "$HF_HOME"
if ! command -v git >/dev/null || ! command -v ffmpeg >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends git ffmpeg >/dev/null
fi
if [ ! -f /app/handler.py ]; then
  git clone --depth 1 https://github.com/SamAlencar/acestep-runpod.git /tmp/acestep-src
  cp /tmp/acestep-src/handler.py /tmp/acestep-src/convert_official.py /tmp/acestep-src/requirements.txt /app/
fi
# Marker so a warm worker skips pip
if [ ! -f /opt/.audiogen-deps-ok ]; then
  python -c "import sys,torch; print(sys.version, torch.__version__, getattr(torch.version,'cuda',None), flush=True)"
  pip install --no-cache-dir --upgrade pip
  pip install --no-cache-dir torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
  pip install --no-cache-dir -r /app/requirements.txt
  pip install --no-cache-dir --no-deps --ignore-requires-python "stable-audio-tools==0.0.20"
  python -c "import torch,torchaudio,diffusers,runpod,stable_audio_tools; print('deps-ok', torch.__version__, torch.version.cuda, flush=True)"
  touch /opt/.audiogen-deps-ok
fi
cd /app
exec python -u /app/handler.py
