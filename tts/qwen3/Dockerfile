# Qwen3-TTS (Apache-2.0) RunPod Serverless worker. WORKDIR must be /app (not /workspace).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
LABEL org.opencontainers.image.source="https://github.com/SamAlencar/acestep-runpod"
ENV DEBIAN_FRONTEND=noninteractive HF_HOME=/opt/hf-cache HF_HUB_CACHE=/opt/hf-cache PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox libsox-dev git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
 && pip install --no-cache-dir runpod soundfile numpy "transformers==4.57.3" "accelerate==1.12.0" librosa sox onnxruntime einops huggingface_hub \
 && pip install --no-cache-dir --no-deps qwen-tts==0.1.1 \
 && python -c "import torch,torchaudio,runpod; from qwen_tts import Qwen3TTSModel; print('ok', torch.__version__, torch.version.cuda)"
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
