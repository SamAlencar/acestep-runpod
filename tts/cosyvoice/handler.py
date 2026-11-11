"""RunPod worker: Fun-CosyVoice3-0.5B-2512 (Apache-2.0). Zero-shot from a provided (synthetic) prompt wav.
Whole text passed in one inference call (no client-side chunking)."""
import base64, os, sys, tempfile, time, traceback
import runpod, torch
os.chdir("/app/CosyVoice")
MODEL_ID = os.environ.get("COSY_MODEL", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
_M = {}

def model():
    if "m" not in _M:
        from huggingface_hub import snapshot_download
        from cosyvoice.cli.cosyvoice import AutoModel
        t = time.time()
        d = snapshot_download(MODEL_ID, local_dir="/opt/models/cosyvoice3")
        _M["m"] = AutoModel(model_dir=d)
        _M["load_s"] = time.time() - t
    return _M["m"]

def handler(job):
    inp = job.get("input") or {}
    try:
        import torchaudio
        m = model(); t1 = time.time()
        p = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        open(p, "wb").write(base64.b64decode(inp["prompt_wav_b64"]))
        prompt_text = "You are a helpful assistant.<|endofprompt|>" + inp.get("prompt_text", "")
        mode = inp.get("mode", "zero_shot")
        if mode == "cross_lingual":
            gen = m.inference_cross_lingual("You are a helpful assistant.<|endofprompt|>" + inp["text"], p, stream=False, speed=float(inp.get("speed", 1.0)))
        else:
            gen = m.inference_zero_shot(inp["text"], prompt_text, p, stream=False, speed=float(inp.get("speed", 1.0)))
        parts = [o["tts_speech"] for o in gen]
        w = torch.cat(parts, dim=1)
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        torchaudio.save(out, w.cpu(), m.sample_rate)
        return {"wav_b64": base64.b64encode(open(out, "rb").read()).decode(), "sample_rate": m.sample_rate,
                "duration_s": w.shape[1] / m.sample_rate, "segments_internal": len(parts),
                "infer_s": time.time() - t1, "load_s": _M.get("load_s"), "model": MODEL_ID,
                "gpu_name": torch.cuda.get_device_name(0)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}

runpod.serverless.start({"handler": handler})
