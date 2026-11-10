"""RunPod serverless worker — ACE-Step 1.5 (MIT).

Input: {
  model?, prompt, lyrics?, duration_s?, seed?, num_inference_steps?,
  bpm?, keyscale?, timesignature?, vocal_language?, budget_usd?, action?
}
Output: {
  wav_b64, sample_rate, duration_s, channels, model, infer_s, load_s,
  wall_s, estimated_cost_usd, warnings[], gpu_name?
}

`model` selects the generator. Default: acestep-v15-turbo (official 2B MIT).
`stable-audio-open-small` is reserved and not enabled yet (needs HF terms + token).
"""
from __future__ import annotations

import base64
import io
import os
import time
import traceback

import numpy as np
import runpod
import soundfile as sf
import torch

PIPELINES: dict = {}
LOAD_S: dict = {}

# Conservative default (~A4000 $0.17/h). Override per endpoint.
COST_PER_GPU_SEC = float(os.environ.get("AUDIOGEN_COST_PER_GPU_SEC", "0.00005"))
MAX_PROMPT_CHARS = 2000
MIN_DURATION = 10.0
MAX_DURATION = 60.0
DEFAULT_MODEL = "acestep-v15-turbo"

# Official, ungated, MIT weights. Turbo 2B is converted once and cached.
MODEL_SPECS = {
    "acestep-v15-turbo": {
        "kind": "acestep-official-convert",
        "repo": "ACE-Step/Ace-Step1.5",
        "dit_config": "acestep-v15-turbo",
        "allow_patterns": ["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*"],
    },
    "acestep-v15-xl-turbo": {
        "kind": "acestep-diffusers",
        "repo": "ACE-Step/acestep-v15-xl-turbo-diffusers",
    },
    # Reserved for later. Do not download or run until terms are accepted.
    "stable-audio-open-small": {
        "kind": "stable-audio-open-small",
        "enabled": False,
        "repo": "stabilityai/stable-audio-open-small",
    },
}


def cache_root() -> str:
    vol = "/runpod-volume"
    if os.path.isdir(vol) and os.access(vol, os.W_OK):
        root = os.path.join(vol, "hf-cache")
    else:
        root = "/opt/hf-cache"
    os.makedirs(root, exist_ok=True)
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ[key] = root
    return root


def _dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"


def load_pipeline(model_id: str):
    if model_id in PIPELINES:
        return PIPELINES[model_id], 0.0, LOAD_S.get(model_id, 0.0)

    spec = MODEL_SPECS.get(model_id)
    if spec is None:
        raise ValueError(f"unknown model {model_id!r}; known: {sorted(MODEL_SPECS)}")
    if spec.get("enabled") is False or spec["kind"] == "stable-audio-open-small":
        raise RuntimeError(
            "model stable-audio-open-small is reserved but not enabled: "
            "accept the Hugging Face terms, register at https://stability.ai/community-license, "
            "and set HF_TOKEN on the endpoint first"
        )

    cache = cache_root()
    t0 = time.time()
    from diffusers import AceStepPipeline

    if spec["kind"] == "acestep-diffusers":
        local = os.path.join(cache, "pipes", model_id)
        if not os.path.isfile(os.path.join(local, "model_index.json")):
            pipe = AceStepPipeline.from_pretrained(spec["repo"], torch_dtype=_dtype())
            os.makedirs(local, exist_ok=True)
            pipe.save_pretrained(local, safe_serialization=True)
        else:
            pipe = AceStepPipeline.from_pretrained(local, torch_dtype=_dtype())
    elif spec["kind"] == "acestep-official-convert":
        converted = os.path.join(cache, "pipes", model_id)
        if not os.path.isfile(os.path.join(converted, "model_index.json")):
            from huggingface_hub import snapshot_download
            from convert_official import convert_official

            src = snapshot_download(
                repo_id=spec["repo"],
                allow_patterns=spec["allow_patterns"],
                cache_dir=cache,
            )
            convert_official(src, spec["dit_config"], converted, "bf16" if _dtype() is torch.bfloat16 else "fp16")
        pipe = AceStepPipeline.from_pretrained(converted, torch_dtype=_dtype())
    else:
        raise ValueError(f"unsupported kind {spec['kind']}")

    if torch.cuda.is_available():
        try:
            pipe = pipe.to("cuda")
        except torch.cuda.OutOfMemoryError:
            print("[acestep] OOM on full GPU load, falling back to CPU offload", flush=True)
            pipe.to("cpu")
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            try:
                pipe.vae.enable_tiling()
            except Exception as e:
                print(f"[acestep] vae tiling skipped: {e}", flush=True)
    else:
        print("[acestep] CUDA not available; running on CPU (slow)", flush=True)

    load_s = time.time() - t0
    PIPELINES[model_id] = pipe
    LOAD_S[model_id] = load_s
    print(f"[acestep] {model_id} ready in {load_s:.1f}s on {_gpu_name()}", flush=True)
    return pipe, load_s, load_s


def _to_wav_bytes(audio, sample_rate: int):
    wav = audio[0] if isinstance(audio, (list, tuple)) else audio
    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    wav = np.asarray(wav).astype(np.float32)
    # (channels, samples) or (samples, channels) or (samples,)
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 2 and wav.shape[0] < wav.shape[1]:
        # (C, S) -> (S, C)
        wav = wav.T
    elif wav.ndim == 1:
        pass
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    channels = 1 if wav.ndim == 1 else wav.shape[1]
    n = wav.shape[0]
    return buf.getvalue(), channels, n, peak


def handler(event):
    t_wall = time.time()
    job = event.get("input") or {}
    warnings = []
    if job.get("action") == "ping":
        return {"ok": True, "gpu_name": _gpu_name(), "models": sorted(MODEL_SPECS), "wall_s": round(time.time() - t_wall, 2)}

    model_id = (job.get("model") or os.environ.get("AUDIOGEN_DEFAULT_MODEL") or DEFAULT_MODEL).strip()
    prompt = (job.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt required"}
    if len(prompt) > MAX_PROMPT_CHARS:
        return {"error": f"prompt too long ({len(prompt)} > {MAX_PROMPT_CHARS})"}

    duration_s = float(job.get("duration_s") or 24)
    if duration_s < MIN_DURATION or duration_s > MAX_DURATION:
        return {"error": f"duration_s must be between {MIN_DURATION} and {MAX_DURATION}"}

    budget = job.get("budget_usd")
    if budget is not None and float(budget) <= 0:
        return {"error": "budget_usd must be > 0"}

    lyrics = job.get("lyrics")
    if lyrics is None:
        lyrics = ""
    steps = int(job.get("num_inference_steps") or 8)
    seed = job.get("seed")
    vocal_language = job.get("vocal_language") or "en"

    try:
        pipe, this_load_s, first_load_s = load_pipeline(model_id)
        sample_rate = int(getattr(pipe, "sample_rate", 48000) or 48000)
        kwargs = {
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": duration_s,
            "num_inference_steps": steps,
            "vocal_language": vocal_language,
        }
        if job.get("bpm") is not None:
            kwargs["bpm"] = int(job["bpm"])
        if job.get("keyscale"):
            kwargs["keyscale"] = job["keyscale"]
        if job.get("timesignature"):
            kwargs["timesignature"] = str(job["timesignature"])
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs["generator"] = torch.Generator(device=device).manual_seed(int(seed))

        t1 = time.time()
        with torch.inference_mode():
            out = pipe(**kwargs)
        infer_s = time.time() - t1

        audios = getattr(out, "audios", out[0] if isinstance(out, (list, tuple)) else out)
        wav_bytes, channels, n_samples, peak = _to_wav_bytes(audios, sample_rate)
        elapsed = time.time() - t_wall
        cost = round(elapsed * COST_PER_GPU_SEC, 5)
        if budget is not None and cost > float(budget):
            warnings.append(f"estimated_cost_usd {cost} exceeded budget_usd {budget}")
        if peak >= 0.999:
            warnings.append(f"peak {peak:.3f} near clipping")
        return {
            "wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": sample_rate,
            "duration_s": round(n_samples / sample_rate, 3),
            "channels": channels,
            "model": model_id,
            "seed": None if seed is None else int(seed),
            "infer_s": round(infer_s, 2),
            "load_s": round(this_load_s, 2),
            "first_load_s": round(first_load_s, 2),
            "wall_s": round(elapsed, 2),
            "peak": round(peak, 4),
            "gpu_name": _gpu_name(),
            "estimated_cost_usd": cost,
            "warnings": warnings,
        }
    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()[-2000:],
            "wall_s": round(time.time() - t_wall, 2),
            "gpu_name": _gpu_name(),
            "model": model_id,
        }


runpod.serverless.start({"handler": handler})
