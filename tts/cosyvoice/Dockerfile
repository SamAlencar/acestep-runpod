# Fun-CosyVoice3-0.5B-2512 (Apache-2.0) RunPod Serverless worker. WORKDIR must be /app.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
LABEL org.opencontainers.image.source="https://github.com/SamAlencar/acestep-runpod"
ENV DEBIAN_FRONTEND=noninteractive HF_HOME=/opt/hf-cache HF_HUB_CACHE=/opt/hf-cache PYTHONUNBUFFERED=1 PYTHONPATH=/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox libsox-dev git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN git clone --recursive --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir runpod "torchaudio==2.3.1" --extra-index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir "setuptools<70" wheel tiktoken more-itertools numba \
 && pip install --no-cache-dir --no-build-isolation openai-whisper==20231117 \
 && pip install --no-cache-dir conformer==0.3.2 diffusers==0.29.0 hydra-core==1.3.2 HyperPyYAML==1.2.3 inflect==7.3.1 librosa==0.10.2 lightning==2.2.4 matplotlib==3.7.5 modelscope==1.20.0 networkx==3.1 numpy==1.26.4 omegaconf==2.3.0 onnx==1.16.0 onnxruntime==1.18.0 protobuf==4.25 pyarrow==18.1.0 pydantic==2.7.0 pyworld==0.3.4 rich==13.7.1 soundfile==0.12.1 transformers==4.51.3 x-transformers==2.11.24 wetext==0.0.4 wget==3.2 gdown==5.1.0 huggingface_hub \
 && cd /app/CosyVoice && python -c "import torch; from cosyvoice.cli.cosyvoice import AutoModel; print('ok', torch.__version__, torch.version.cuda)"
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
