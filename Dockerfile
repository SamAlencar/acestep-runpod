# ACE-Step 1.5 (MIT) RunPod Serverless worker (CUDA).
# Follows the same pattern as ghcr.io/samalencar/omnivoice-runpod.
# Keep the base image torch CUDA build intact — never force-reinstall torch.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
LABEL org.opencontainers.image.source="https://github.com/SamAlencar/acestep-runpod"
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/hf-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf-cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /runpod-volume/hf-cache /opt/hf-cache
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && python -c "import torch; print(torch.__version__, torch.version.cuda)"
COPY handler.py convert_official.py /app/
CMD ["python", "-u", "/app/handler.py"]
