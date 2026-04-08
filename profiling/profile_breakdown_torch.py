from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain why TTFT is different from steady-state token latency."
MAX_NEW_TOKENS = 8
OUT_PATH = Path("results/torch_profiler_summary.txt")


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        with torch.inference_mode():
            for step in range(MAX_NEW_TOKENS):
                with record_function(f"decode_step_{step}"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)], dim=1
                    )

    summary = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(summary, encoding="utf-8")
    print(f"Saved profiler summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
