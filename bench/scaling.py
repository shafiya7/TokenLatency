from __future__ import annotations

from dataclasses import asdict
from typing import Any

from transformers import AutoTokenizer

from bench.config import BenchmarkConfig, ScalingConfig
from bench.harness import HFLatencyHarness
from bench.utils import build_long_prompt, cleanup_torch, ensure_dir, json_dump, timestamp_slug, truncate_prompt_by_tokens


def run_scaling(cfg: ScalingConfig) -> dict[str, Any]:
    base_prompt = build_long_prompt()
    rows = []

    for model_name in cfg.model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        try:
            for precision in cfg.precisions:
                for prompt_length in cfg.prompt_lengths:
                    prompt = truncate_prompt_by_tokens(tokenizer, base_prompt, prompt_length)
                    bench_cfg = BenchmarkConfig(
                        model_name=model_name,
                        prompt=prompt,
                        max_new_tokens=cfg.max_new_tokens,
                        warmup_runs=cfg.warmup_runs,
                        measured_trials=cfg.measured_trials,
                        device_preference=cfg.device_preference,
                        torch_dtype=precision,
                        save_dir=cfg.save_dir,
                        system_name=cfg.system_name,
                        estimated_bandwidth_gbps=cfg.estimated_bandwidth_gbps,
                        use_kv_cache=cfg.use_kv_cache,
                    )
                    harness = HFLatencyHarness(bench_cfg)
                    try:
                        result = harness.run()
                    finally:
                        harness.close()
                    rows.append({
                        "model_name": model_name,
                        "prompt_length_tokens": prompt_length,
                        "precision": precision,
                        "ttft_mean_ms": result["summary"]["filtered"]["ttft_ms"]["mean"],
                        "per_token_mean_ms": result["summary"]["filtered"]["avg_per_token_ms"]["mean"],
                        "e2e_mean_ms": result["summary"]["filtered"]["e2e_ms"]["mean"],
                        "kv_cache_bytes_per_token": result["trials"][0].get("kv_cache_bytes_per_token"),
                        "result_file": result["saved_to"],
                        "runtime": result["runtime"],
                    })
        finally:
            cleanup_torch(None, None)

    payload = {"config": asdict(cfg), "rows": rows}
    save_dir = ensure_dir(cfg.save_dir)
    save_path = save_dir / f"scaling_{timestamp_slug()}.json"
    json_dump(payload, save_path)
    payload["saved_to"] = str(save_path)
    return payload
