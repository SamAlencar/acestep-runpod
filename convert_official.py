"""Convert official ACE-Step 1.5 checkpoints into a Diffusers AceStepPipeline layout.

Adapted from huggingface/diffusers scripts/convert_ace_step_to_diffusers.py.
Audio tokenizer / detokenizer modules are included only when those keys exist
(the official 2B turbo checkpoint may omit them; text-to-music still works).
"""
from __future__ import annotations

import json
import os
import shutil

import torch
from safetensors.torch import load_file


_ATTN_KEY_RENAMES = [
    (".q_proj.", ".to_q."),
    (".k_proj.", ".to_k."),
    (".v_proj.", ".to_v."),
    (".o_proj.", ".to_out.0."),
    (".q_norm.", ".norm_q."),
    (".k_norm.", ".norm_k."),
]


def _rename_attn_keys(key: str) -> str:
    for old, new in _ATTN_KEY_RENAMES:
        key = key.replace(old, new)
    return key


def convert_official(checkpoint_dir: str, dit_config: str, output_dir: str, dtype_str: str = "bf16") -> str:
    if os.path.isdir(os.path.join(output_dir, "model_index.json")) or os.path.isfile(
        os.path.join(output_dir, "model_index.json")
    ):
        return output_dir

    dit_dir = os.path.join(checkpoint_dir, dit_config)
    vae_dir = os.path.join(checkpoint_dir, "vae")
    text_encoder_dir = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
    single_model_path = os.path.join(dit_dir, "model.safetensors")
    sharded_index_path = os.path.join(dit_dir, "model.safetensors.index.json")
    config_path = os.path.join(dit_dir, "config.json")

    if os.path.exists(single_model_path):
        dit_weight_files = [single_model_path]
    elif os.path.exists(sharded_index_path):
        with open(sharded_index_path) as f:
            shard_index = json.load(f)
        dit_weight_files = [os.path.join(dit_dir, s) for s in sorted(set(shard_index["weight_map"].values()))]
    else:
        raise FileNotFoundError(f"DiT weights not found in {dit_dir}")

    for path, name in [(config_path, "config"), (vae_dir, "VAE"), (text_encoder_dir, "text encoder")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at: {path}")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    target_dtype = dtype_map[dtype_str]

    with open(config_path) as f:
        original_config = json.load(f)

    state_dict = {}
    for p in dit_weight_files:
        print(f"[convert] loading {os.path.basename(p)}", flush=True)
        state_dict.update(load_file(p))

    transformer_sd, condition_encoder_sd = {}, {}
    audio_tokenizer_sd, audio_token_detokenizer_sd, other_sd = {}, {}, {}
    for key, value in state_dict.items():
        if key.startswith("decoder."):
            new_key = key[len("decoder.") :]
            new_key = new_key.replace("proj_in.1.", "proj_in_conv.")
            new_key = new_key.replace("proj_out.1.", "proj_out_conv.")
            transformer_sd[_rename_attn_keys(new_key)] = value.to(target_dtype)
        elif key.startswith("encoder."):
            condition_encoder_sd[_rename_attn_keys(key[len("encoder.") :])] = value.to(target_dtype)
        elif key == "null_condition_emb":
            condition_encoder_sd["null_condition_emb"] = value.to(target_dtype)
        elif key.startswith("tokenizer."):
            audio_tokenizer_sd[_rename_attn_keys(key[len("tokenizer.") :])] = value.to(target_dtype)
        elif key.startswith("detokenizer."):
            audio_token_detokenizer_sd[_rename_attn_keys(key[len("detokenizer.") :])] = value.to(target_dtype)
        else:
            other_sd[key] = value.to(target_dtype)

    encoder_hidden_size = original_config.get("encoder_hidden_size", original_config["hidden_size"])
    encoder_intermediate_size = original_config.get("encoder_intermediate_size", original_config["intermediate_size"])
    encoder_num_attention_heads = original_config.get(
        "encoder_num_attention_heads", original_config["num_attention_heads"]
    )
    encoder_num_key_value_heads = original_config.get(
        "encoder_num_key_value_heads", original_config["num_key_value_heads"]
    )

    transformer_config = {
        "hidden_size": original_config["hidden_size"],
        "intermediate_size": original_config["intermediate_size"],
        "num_hidden_layers": original_config["num_hidden_layers"],
        "num_attention_heads": original_config["num_attention_heads"],
        "num_key_value_heads": original_config["num_key_value_heads"],
        "head_dim": original_config["head_dim"],
        "in_channels": original_config["in_channels"],
        "audio_acoustic_hidden_dim": original_config["audio_acoustic_hidden_dim"],
        "patch_size": original_config["patch_size"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "sliding_window": original_config["sliding_window"],
        "layer_types": original_config["layer_types"],
        "encoder_hidden_size": encoder_hidden_size,
        "is_turbo": bool(original_config.get("is_turbo", False)),
        "model_version": original_config.get("model_version"),
    }
    condition_encoder_config = {
        "hidden_size": encoder_hidden_size,
        "intermediate_size": encoder_intermediate_size,
        "text_hidden_dim": original_config["text_hidden_dim"],
        "timbre_hidden_dim": original_config["timbre_hidden_dim"],
        "num_lyric_encoder_hidden_layers": original_config["num_lyric_encoder_hidden_layers"],
        "num_timbre_encoder_hidden_layers": original_config["num_timbre_encoder_hidden_layers"],
        "num_attention_heads": encoder_num_attention_heads,
        "num_key_value_heads": encoder_num_key_value_heads,
        "head_dim": original_config["head_dim"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "sliding_window": original_config["sliding_window"],
    }

    silence_latent_src = os.path.join(dit_dir, "silence_latent.pt")
    if os.path.exists(silence_latent_src):
        silence_raw = torch.load(silence_latent_src, weights_only=True, map_location="cpu")
        condition_encoder_sd["silence_latent"] = silence_raw.transpose(1, 2).to(target_dtype).contiguous()
        print(f"[convert] baked silence_latent {tuple(condition_encoder_sd['silence_latent'].shape)}", flush=True)

    from transformers import AutoModel, AutoTokenizer
    from diffusers import (
        AceStepConditionEncoder,
        AceStepPipeline,
        AceStepTransformer1DModel,
        AutoencoderOobleck,
        FlowMatchEulerDiscreteScheduler,
    )

    print("[convert] constructing modules", flush=True)
    transformer = AceStepTransformer1DModel(**transformer_config).to(target_dtype)
    transformer.load_state_dict(transformer_sd, strict=True)
    condition_encoder = AceStepConditionEncoder(**condition_encoder_config).to(target_dtype)
    condition_encoder.load_state_dict(condition_encoder_sd, strict=True)

    audio_tokenizer = audio_token_detokenizer = None
    if audio_tokenizer_sd and audio_token_detokenizer_sd:
        from diffusers.pipelines.ace_step import AceStepAudioTokenDetokenizer, AceStepAudioTokenizer

        n_pool = original_config.get("num_attention_pooler_hidden_layers", 2)
        tok_cfg = {
            "hidden_size": encoder_hidden_size,
            "intermediate_size": encoder_intermediate_size,
            "audio_acoustic_hidden_dim": original_config["audio_acoustic_hidden_dim"],
            "pool_window_size": original_config.get("pool_window_size", 5),
            "fsq_dim": original_config.get("fsq_dim", encoder_hidden_size),
            "fsq_input_levels": original_config.get("fsq_input_levels", [8, 8, 8, 5, 5, 5]),
            "fsq_input_num_quantizers": original_config.get("fsq_input_num_quantizers", 1),
            "num_attention_pooler_hidden_layers": n_pool,
            "num_attention_heads": encoder_num_attention_heads,
            "num_key_value_heads": encoder_num_key_value_heads,
            "head_dim": original_config["head_dim"],
            "rope_theta": original_config["rope_theta"],
            "attention_bias": original_config["attention_bias"],
            "attention_dropout": original_config["attention_dropout"],
            "rms_norm_eps": original_config["rms_norm_eps"],
            "sliding_window": original_config["sliding_window"],
            "layer_types": original_config["layer_types"][:n_pool],
        }
        detok_cfg = {k: v for k, v in tok_cfg.items() if k not in ("fsq_dim", "fsq_input_levels", "fsq_input_num_quantizers")}
        audio_tokenizer = AceStepAudioTokenizer(**tok_cfg).to(target_dtype)
        audio_tokenizer.load_state_dict(audio_tokenizer_sd, strict=True)
        audio_token_detokenizer = AceStepAudioTokenDetokenizer(**detok_cfg).to(target_dtype)
        audio_token_detokenizer.load_state_dict(audio_token_detokenizer_sd, strict=True)
    elif audio_tokenizer_sd or audio_token_detokenizer_sd:
        print("[convert] incomplete tokenizer/detokenizer keys; skipping both", flush=True)

    vae = AutoencoderOobleck.from_pretrained(vae_dir).to(target_dtype)
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, torch_dtype=target_dtype)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0)

    pipe = AceStepPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        condition_encoder=condition_encoder,
        scheduler=scheduler,
        audio_tokenizer=audio_tokenizer,
        audio_token_detokenizer=audio_token_detokenizer,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[convert] saving -> {output_dir}", flush=True)
    pipe.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    if os.path.exists(silence_latent_src):
        shutil.copy2(silence_latent_src, os.path.join(output_dir, "silence_latent.pt"))
    if other_sd:
        print(f"[convert] dropped {len(other_sd)} unmatched keys: {list(other_sd)[:8]}", flush=True)
    return output_dir
