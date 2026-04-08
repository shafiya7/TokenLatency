# from __future__ import annotations

# import gc
# import json
# import time
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# import torch


# def ensure_dir(path: str | Path) -> Path:
#     path = Path(path)
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def timestamp_slug() -> str:
#     return datetime.now().strftime("%Y%m%d_%H%M%S")


# def now_ms() -> float:
#     return time.perf_counter_ns() / 1_000_000.0


# def sync_device(device: torch.device) -> None:
#     if device.type == "cuda":
#         torch.cuda.synchronize(device)
#     elif device.type == "mps":
#         try:
#             torch.mps.synchronize()
#         except Exception:
#             pass


# def cleanup_torch(model: Any | None = None, device: torch.device | None = None) -> None:
#     if model is not None:
#         del model
#     gc.collect()
#     if device is not None:
#         if device.type == "cuda":
#             torch.cuda.empty_cache()
#         elif device.type == "mps":
#             try:
#                 torch.mps.empty_cache()
#             except Exception:
#                 pass


# def resolve_device(device_preference: str) -> torch.device:
#     pref = device_preference.lower().strip()
#     if pref == "cpu":
#         return torch.device("cpu")
#     if pref == "cuda":
#         if not torch.cuda.is_available():
#             raise RuntimeError("CUDA requested but not available.")
#         return torch.device("cuda")
#     if pref == "mps":
#         if not torch.backends.mps.is_available():
#             raise RuntimeError("MPS requested but not available.")
#         return torch.device("mps")

#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     if torch.backends.mps.is_available():
#         return torch.device("mps")
#     return torch.device("cpu")


# def resolve_dtype(dtype_name: str, device: torch.device):
#     name = dtype_name.lower().strip()
#     if name == "float32":
#         return torch.float32
#     if name == "float16":
#         return torch.float16
#     if name == "bfloat16":
#         return torch.bfloat16
#     if device.type == "cuda":
#         return torch.float16
#     if device.type == "mps":
#         return torch.float16
#     return torch.float32


# def json_dump(data: Any, path: str | Path) -> Path:
#     path = Path(path)
#     path.write_text(json.dumps(data, indent=2), encoding="utf-8")
#     return path


# def truncate_prompt_by_tokens(tokenizer, text: str, target_tokens: int) -> str:
#     token_ids = tokenizer.encode(text, add_special_tokens=False)
#     if len(token_ids) <= target_tokens:
#         return text
#     return tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)


# def build_long_prompt() -> str:
#     return (
#         "Explain TTFT, steady-state token latency, KV-cache reuse, prompt processing, "
#         "attention scaling with sequence length, framework overhead, and memory bandwidth limits. "
#     ) * 40


# def estimate_kv_cache_bytes_per_token(config: Any, dtype_bytes: int) -> int | None:
#     num_layers = getattr(config, "num_hidden_layers", None)
#     num_kv_heads = getattr(config, "num_key_value_heads", None)
#     num_attn_heads = getattr(config, "num_attention_heads", None)
#     hidden_size = getattr(config, "hidden_size", None)

#     if num_kv_heads is None:
#         num_kv_heads = num_attn_heads
#     if hidden_size is None or num_attn_heads is None:
#         return None
#     head_dim = hidden_size // num_attn_heads
#     if None in (num_layers, num_kv_heads, head_dim):
#         return None

#     return int(2 * num_layers * num_kv_heads * head_dim * dtype_bytes)


# def estimate_kv_cache_rw_bytes_per_decode_step(
#     prompt_tokens: int,
#     generated_so_far: int,
#     kv_cache_bytes_per_token: int | None,
# ) -> dict[str, int | None]:
#     if kv_cache_bytes_per_token is None:
#         return {"kv_read_bytes": None, "kv_write_bytes": None, "kv_total_bytes": None}
#     total_cached_tokens = prompt_tokens + generated_so_far
#     kv_read_bytes = total_cached_tokens * kv_cache_bytes_per_token
#     kv_write_bytes = kv_cache_bytes_per_token
#     return {
#         "kv_read_bytes": int(kv_read_bytes),
#         "kv_write_bytes": int(kv_write_bytes),
#         "kv_total_bytes": int(kv_read_bytes + kv_write_bytes),
#     }


# def estimate_transfer_time_ms(num_bytes: int | None, bandwidth_gbps: float | None) -> float | None:
#     if num_bytes is None or bandwidth_gbps is None or bandwidth_gbps <= 0:
#         return None
#     return float(num_bytes / (bandwidth_gbps * 1e9) * 1e3)


from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_ms() -> float:
    return time.perf_counter_ns() / 1_000_000.0


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def cleanup_torch(model: Any | None = None, device: torch.device | None = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if device is not None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


def resolve_device(device_preference: str) -> torch.device:
    pref = device_preference.lower().strip()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_name: str, device: torch.device):
    name = dtype_name.lower().strip()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def json_dump(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def truncate_prompt_by_tokens(tokenizer, text: str, target_tokens: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= target_tokens:
        return text
    return tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)


def build_long_prompt() -> str:
    return (
        "Explain TTFT, steady-state token latency, KV-cache reuse, prompt processing, "
        "attention scaling with sequence length, framework overhead, and memory bandwidth limits. "
    ) * 40


def estimate_kv_cache_bytes_per_token(config: Any, dtype_bytes: int) -> int | None:
    num_layers = getattr(config, "num_hidden_layers", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    num_attn_heads = getattr(config, "num_attention_heads", None)
    hidden_size = getattr(config, "hidden_size", None)

    if num_kv_heads is None:
        num_kv_heads = num_attn_heads
    if hidden_size is None or num_attn_heads is None:
        return None
    head_dim = hidden_size // num_attn_heads
    if None in (num_layers, num_kv_heads, head_dim):
        return None

    return int(2 * num_layers * num_kv_heads * head_dim * dtype_bytes)


def estimate_kv_cache_rw_bytes_per_decode_step(
    prompt_tokens: int,
    generated_so_far: int,
    kv_cache_bytes_per_token: int | None,
) -> dict[str, int | None]:
    if kv_cache_bytes_per_token is None:
        return {"kv_read_bytes": None, "kv_write_bytes": None, "kv_total_bytes": None}
    total_cached_tokens = prompt_tokens + generated_so_far
    kv_read_bytes = total_cached_tokens * kv_cache_bytes_per_token
    kv_write_bytes = kv_cache_bytes_per_token
    return {
        "kv_read_bytes": int(kv_read_bytes),
        "kv_write_bytes": int(kv_write_bytes),
        "kv_total_bytes": int(kv_read_bytes + kv_write_bytes),
    }


def estimate_transfer_time_ms(num_bytes: int | None, bandwidth_gbps: float | None) -> float | None:
    if num_bytes is None or bandwidth_gbps is None or bandwidth_gbps <= 0:
        return None
    return float(num_bytes / (bandwidth_gbps * 1e9) * 1e3)



def estimate_effective_bandwidth_gbps(device: torch.device, dtype: torch.dtype | None = None) -> float:
    """Heuristic bandwidth used for analytical KV-cache transfer estimates.

    This is not a hardware-counter measurement. It exists so Goal 2 reports do not
    silently collapse to 0 ms when the user does not manually provide bandwidth.
    Values are conservative ballpark figures meant only for analytical modeling.
    """
    if device.type == "cuda":
        if dtype == torch.float32:
            return 500.0
        return 700.0
    if device.type == "mps":
        return 120.0
    return 50.0
