"""RunPod worker: Qwen3-TTS CustomVoice and Base voice-clone (both Apache-2.0).

Modes:
  - custom_voice: generate_custom_voice (speaker + optional instruct)
  - voice_clone:  generate_voice_clone (ref_audio_b64 + ref_text)
The whole `text` is passed in a single generate call (no chunking).
HF_TOKEN is read from the environment only. It is never logged or returned.
"""
from __future__ import annotations

import base64, io, os, tempfile, time, traceback

import numpy as np
import runpod
import soundfile as sf
import torch

CUSTOM_MODEL = os.environ.get("QWEN_CUSTOM_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
CLONE_MODEL = os.environ.get("QWEN_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
_M: dict = {}


def _get_model(model_id: str):
    # Only one model in VRAM at a time (16 GB cards).
    if _M.get("id") != model_id:
        for k in ("m", "id", "load_s"):
            _M.pop(k, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from qwen_tts import Qwen3TTSModel
        t = time.time()
        _M["m"] = Qwen3TTSModel.from_pretrained(
            model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        _M["load_s"] = time.time() - t
        _M["id"] = model_id
    return _M["m"]


def _set_seed(seed):
    if seed is None:
        return
    s = int(seed)
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _wav_b64(w, sr):
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.squeeze()
        if w.ndim > 1:
            w = w.mean(axis=0 if w.shape[0] < w.shape[-1] else 1)
    buf = io.BytesIO()
    sf.write(buf, w, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode(), float(len(w) / sr)


def handler(job):
    inp = job.get("input") or {}
    try:
        mode = inp.get("mode") or (
            "voice_clone" if inp.get("ref_audio_b64") or inp.get("ref_audio") else "custom_voice"
        )
        seed = inp.get("seed")
        _set_seed(seed)
        t1 = time.time()
        if mode == "voice_clone":
            mid = inp.get("model") or CLONE_MODEL
            m = _get_model(mid)
            raw = inp.get("ref_audio_b64")
            if not raw:
                raise ValueError("voice_clone requires ref_audio_b64")
            if not inp.get("ref_text"):
                raise ValueError("voice_clone requires ref_text")
            path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            with open(path, "wb") as f:
                f.write(base64.b64decode(raw))
            wavs, sr = m.generate_voice_clone(
                text=inp["text"],
                language=inp.get("language", "Portuguese"),
                ref_audio=path,
                ref_text=inp["ref_text"],
                do_sample=True,
            )
        else:
            mid = inp.get("model") or CUSTOM_MODEL
            m = _get_model(mid)
            wavs, sr = m.generate_custom_voice(
                text=inp["text"],
                language=inp.get("language", "Portuguese"),
                speaker=inp.get("speaker", "Aiden"),
                instruct=inp.get("instruct") or None,
                do_sample=True,
            )
        b64, dur = _wav_b64(wavs[0], sr)
        return {
            "wav_b64": b64,
            "sample_rate": int(sr),
            "duration_s": dur,
            "infer_s": time.time() - t1,
            "load_s": _M.get("load_s"),
            "model": mid,
            "mode": mode,
            "seed": None if seed is None else int(seed),
            "gpu_name": torch.cuda.get_device_name(0),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}


runpod.serverless.start({"handler": handler})
