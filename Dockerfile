# ACE-Step 1.5 (MIT) + Stable Audio Open Small RunPod Serverless worker (CUDA).
# Follows the same pattern as ghcr.io/samalencar/omnivoice-runpod.
# Keep the base image torch CUDA build intact — never force-reinstall torch.
# Do not bake secrets. HF_TOKEN belongs on the endpoint environment, not in the image.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
LABEL org.opencontainers.image.source="https://github.com/SamAlencar/acestep-runpod"
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    HF_HUB_CACHE=/opt/hf-cache \
    PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /runpod-volume/hf-cache /opt/hf-cache
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && python -c "import sys,torch; print(sys.version, torch.__version__, torch.version.cuda)" \
 && pip install --no-cache-dir torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps --ignore-requires-python "stable-audio-tools==0.0.20" \
 && python -c "import torch,torchaudio,diffusers,runpod,stable_audio_tools; print('ok', torch.__version__, torch.version.cuda)"
COPY handler.py convert_official.py /app/
CMD ["python", "-u", "/app/handler.py"]
