"""RunPod worker: Qwen3-TTS CustomVoice (Apache-2.0). Whole text in one generate call (no chunking)."""
import base64, io, os, time, traceback
import numpy as np, runpod, soundfile as sf, torch

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
_M = {}

def model():
    if "m" not in _M:
        from qwen_tts import Qwen3TTSModel
        t = time.time()
        _M["m"] = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        _M["load_s"] = time.time() - t
    return _M["m"]

def handler(job):
    inp = job.get("input") or {}
    try:
        t0 = time.time(); m = model(); t1 = time.time()
        wavs, sr = m.generate_custom_voice(
            text=inp["text"], language=inp.get("language", "Portuguese"),
            speaker=inp.get("speaker", "Aiden"), instruct=inp.get("instruct") or None)
        w = np.asarray(wavs[0], dtype=np.float32)
        buf = io.BytesIO(); sf.write(buf, w, sr, format="WAV", subtype="PCM_16")
        return {"wav_b64": base64.b64encode(buf.getvalue()).decode(), "sample_rate": sr,
                "duration_s": len(w) / sr, "infer_s": time.time() - t1, "load_s": _M.get("load_s"),
                "model": MODEL_ID, "gpu_name": torch.cuda.get_device_name(0),
                "speakers": list(m.get_supported_speakers() or [])}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}

runpod.serverless.start({"handler": handler})
