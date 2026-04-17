from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.config import BenchmarkConfig
from bench.latency_breakdown import StageTimer
from bench.utils import (
    cleanup_torch,
    ensure_dir,
    estimate_kv_cache_bytes_per_token,
    estimate_kv_cache_rw_bytes_per_decode_step,
    estimate_transfer_time_ms,
    estimate_effective_bandwidth_gbps,
    json_dump,
    now_ms,
    resolve_device,
    resolve_dtype,
    sync_device,
    timestamp_slug,
)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p95": float("nan"),
        }
    vals = sorted(values)
    n = len(vals)
    mid = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return {
        "count": n,
        "mean": sum(vals) / n,
        "median": mid,
        "min": vals[0],
        "max": vals[-1],
        "p95": percentile(vals, 0.95),
    }


def iqr_bounds(values: list[float], multiplier: float = 1.5) -> tuple[float, float] | None:
    if len(values) < 4:
        return None
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    return (q1 - multiplier * iqr, q3 + multiplier * iqr)


def iqr_filter(values: list[float], multiplier: float = 1.5) -> list[float]:
    bounds = iqr_bounds(values, multiplier)
    if bounds is None:
        return values[:]
    lo, hi = bounds
    return [x for x in values if lo <= x <= hi]


def _avg_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set().union(*[row.keys() for row in rows])
    return {k: mean([row.get(k, 0.0) for row in rows]) for k in keys}


def _goal2_breakdown_mapping() -> dict[str, str]:
    return {
        "embedding_lookup_ms": "Measured by forward hook on model token embedding module (embed_tokens).",
        "attention_qkv_ms": "Measured by forward hooks on q_proj, k_proj, and v_proj linear layers.",
        "attention_softmax_ms": "Derived as: attention_total_ms - attention_qkv_ms - attention_projection_ms.",
        "attention_projection_ms": "Measured by forward hook on self-attention output projection (o_proj).",
        "attention_total_ms": "Measured by forward hook on self-attention module.",
        "kv_cache_rw_estimated_ms": "Estimated from KV bytes transferred per decode step divided by configured bandwidth. Not directly isolated by profiler.",
        "mlp_projection_ms": "Measured by forward hooks on gate_proj, up_proj, and down_proj.",
        "mlp_total_ms": "Measured by forward hook on MLP module.",
        "layernorm_total_ms": "Measured by forward hooks on input/post-attention/final norm modules.",
        "residual_ms": "Derived as: decoder_layer_total_ms - attention_total_ms - mlp_total_ms - layernorm_total_ms.",
        "lm_head_ms": "Measured by forward hook on LM head.",
        "sampling_decoding_ms": "Measured around token selection step after logits are produced.",
        "framework_overhead_ms": "Residual time not covered by non-overlapping component measurements.",
        "total_token_ms": "Wall-clock time for one decode step including model forward, sampling, and uncategorized overhead.",
    }


def _finalize_breakdown(
    raw: dict[str, float],
    total_step_ms: float,
    sampling_ms: float,
    kv_rw_estimated_ms: float,
) -> dict[str, float]:
    out = {k: float(v) for k, v in raw.items()}

    qkv_ms = max(0.0, out.get("attention_qkv_ms", 0.0))
    attention_proj_ms = max(0.0, out.get("attention_projection_ms", 0.0))
    attention_total_ms = max(0.0, out.get("attention_total_ms", 0.0))
    mlp_total_ms = max(0.0, out.get("mlp_total_ms", 0.0))
    layernorm_ms = max(0.0, out.get("layernorm_total_ms", 0.0))
    decoder_layer_total_ms = max(0.0, out.get("decoder_layer_total_ms", 0.0))
    embedding_ms = max(0.0, out.get("embedding_lookup_ms", 0.0))
    lm_head_ms = max(0.0, out.get("lm_head_ms", 0.0))

    softmax_ms = max(0.0, attention_total_ms - qkv_ms - attention_proj_ms)
    residual_ms = max(0.0, decoder_layer_total_ms - attention_total_ms - mlp_total_ms - layernorm_ms)

    out["attention_softmax_ms"] = softmax_ms
    out["residual_ms"] = residual_ms
    out["kv_cache_rw_estimated_ms"] = float(kv_rw_estimated_ms)
    out["sampling_decoding_ms"] = float(sampling_ms)

    non_overlapping_total = (
        embedding_ms
        + qkv_ms
        + softmax_ms
        + attention_proj_ms
        + kv_rw_estimated_ms
        + mlp_total_ms
        + layernorm_ms
        + residual_ms
        + lm_head_ms
        + sampling_ms
    )
    out["framework_overhead_ms"] = max(0.0, total_step_ms - non_overlapping_total)
    out["total_token_ms"] = float(total_step_ms)
    return out


@dataclass(slots=True)
class TrialResult:
    trial_index: int
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    avg_per_token_ms: float
    e2e_ms: float
    per_token_latencies_ms: list[float]
    first_step_breakdown_ms: dict[str, float]
    steady_state_breakdown_avg_ms: dict[str, float]
    per_token_breakdown_ms: list[dict[str, float]]
    kv_cache_bytes_per_token: int | None
    kv_cache_rw_per_step: list[dict[str, int | float | None]]
    output_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HFLatencyHarness:
    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device_preference)
        self.dtype = resolve_dtype(cfg.torch_dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=False,
            trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {
            "trust_remote_code": cfg.trust_remote_code,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "torch_dtype": self.dtype,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        except TypeError:
            model_kwargs.pop("torch_dtype", None)
            model_kwargs["dtype"] = self.dtype
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self.dtype_bytes = torch.tensor([], dtype=self.dtype).element_size()
        self.kv_cache_bytes_per_token = estimate_kv_cache_bytes_per_token(self.model.config, self.dtype_bytes)
        self.effective_bandwidth_gbps = (
            float(cfg.estimated_bandwidth_gbps)
            if cfg.estimated_bandwidth_gbps is not None
            else estimate_effective_bandwidth_gbps(self.device, self.dtype)
        )

    def close(self) -> None:
        cleanup_torch(self.model, self.device)

    def _pick_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / self.cfg.temperature
        if self.cfg.top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=min(self.cfg.top_k, logits.shape[-1]), dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return topk_idx.gather(-1, sampled)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def _run_trial(self, prompt: str, trial_index: int) -> TrialResult:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        generated_ids: list[int] = []
        per_token_latencies_ms: list[float] = []
        steady_state_breakdowns: list[dict[str, float]] = []
        per_token_breakdown_ms: list[dict[str, float]] = []
        kv_cache_rw_per_step: list[dict[str, int | float | None]] = []

        with StageTimer(self.model, self.device) as timer:
            sync_device(self.device)
            t0 = now_ms()
            before = timer.snapshot()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=self.cfg.use_kv_cache,
                return_dict=True,
            )
            sync_device(self.device)
            logits = outputs.logits[:, -1, :]
            t_sample0 = now_ms()
            next_token = self._pick_next_token(logits)
            sync_device(self.device)
            t_first_done = now_ms()

            first_raw = StageTimer.diff(timer.snapshot(), before)
            ttft_ms = t_first_done - t0

            kv0 = estimate_kv_cache_rw_bytes_per_decode_step(
                prompt_tokens=int(input_ids.shape[1]),
                generated_so_far=0,
                kv_cache_bytes_per_token=self.kv_cache_bytes_per_token if self.cfg.use_kv_cache else None,
            )
            kv0_transfer_ms = estimate_transfer_time_ms(kv0["kv_total_bytes"], self.effective_bandwidth_gbps) or 0.0
            kv0["estimated_transfer_ms"] = kv0_transfer_ms
            kv_cache_rw_per_step.append(kv0)

            first_step_breakdown = _finalize_breakdown(
                raw=first_raw,
                total_step_ms=ttft_ms,
                sampling_ms=t_first_done - t_sample0,
                kv_rw_estimated_ms=kv0_transfer_ms,
            )
            first_step_breakdown["token_index"] = 1.0
            first_step_breakdown["is_first_token"] = 1.0
            per_token_breakdown_ms.append(dict(first_step_breakdown))

            past_key_values = outputs.past_key_values if self.cfg.use_kv_cache else None
            generated_ids.append(int(next_token[0, 0].item()))

            current_input = next_token
            current_attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.device)], dim=1
            )

            for step_idx in range(self.cfg.max_new_tokens - 1):
                before = timer.snapshot()
                sync_device(self.device)
                step_start = now_ms()

                model_kwargs = {
                    "input_ids": current_input if self.cfg.use_kv_cache else torch.cat([input_ids, torch.tensor([generated_ids], device=self.device, dtype=input_ids.dtype)], dim=1),
                    "attention_mask": current_attention_mask,
                    "use_cache": self.cfg.use_kv_cache,
                    "return_dict": True,
                }
                if self.cfg.use_kv_cache:
                    model_kwargs["past_key_values"] = past_key_values

                outputs = self.model(**model_kwargs)
                sync_device(self.device)
                logits = outputs.logits[:, -1, :]
                t_sample = now_ms()
                next_token = self._pick_next_token(logits)
                sync_device(self.device)
                step_done = now_ms()

                raw_delta = StageTimer.diff(timer.snapshot(), before)
                kv_rw = estimate_kv_cache_rw_bytes_per_decode_step(
                    prompt_tokens=int(input_ids.shape[1]),
                    generated_so_far=step_idx + 1,
                    kv_cache_bytes_per_token=self.kv_cache_bytes_per_token if self.cfg.use_kv_cache else None,
                )
                kv_transfer_ms = estimate_transfer_time_ms(kv_rw["kv_total_bytes"], self.effective_bandwidth_gbps) or 0.0
                delta = _finalize_breakdown(
                    raw=raw_delta,
                    total_step_ms=step_done - step_start,
                    sampling_ms=step_done - t_sample,
                    kv_rw_estimated_ms=kv_transfer_ms,
                )
                delta["token_index"] = float(step_idx + 2)
                delta["is_first_token"] = 0.0
                steady_state_breakdowns.append(delta)
                per_token_breakdown_ms.append(dict(delta))

                kv_rw["estimated_transfer_ms"] = kv_transfer_ms
                kv_cache_rw_per_step.append(kv_rw)
                per_token_latencies_ms.append(step_done - step_start)

                if self.cfg.use_kv_cache:
                    past_key_values = outputs.past_key_values
                generated_ids.append(int(next_token[0, 0].item()))
                current_input = next_token
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), dtype=current_attention_mask.dtype, device=self.device)], dim=1
                )

        steady_avg = _avg_dict(steady_state_breakdowns)
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        e2e_ms = ttft_ms + sum(per_token_latencies_ms)

        return TrialResult(
            trial_index=trial_index,
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=len(generated_ids),
            ttft_ms=float(ttft_ms),
            avg_per_token_ms=float(mean(per_token_latencies_ms) if per_token_latencies_ms else 0.0),
            e2e_ms=float(e2e_ms),
            per_token_latencies_ms=[float(x) for x in per_token_latencies_ms],
            first_step_breakdown_ms={k: float(v) for k, v in first_step_breakdown.items()},
            steady_state_breakdown_avg_ms={k: float(v) for k, v in steady_avg.items()},
            per_token_breakdown_ms=[{k: float(v) for k, v in row.items()} for row in per_token_breakdown_ms],
            kv_cache_bytes_per_token=self.kv_cache_bytes_per_token if self.cfg.use_kv_cache else None,
            kv_cache_rw_per_step=kv_cache_rw_per_step,
            output_text=output_text,
        )

    def run(self) -> dict[str, Any]:
        for i in range(self.cfg.warmup_runs):
            self._run_trial(self.cfg.prompt, trial_index=-(i + 1))

        trials = [self._run_trial(self.cfg.prompt, trial_index=i) for i in range(1, self.cfg.measured_trials + 1)]

        ttft = [t.ttft_ms for t in trials]
        per_token = [t.avg_per_token_ms for t in trials]
        e2e = [t.e2e_ms for t in trials]

        filtered_ttft = iqr_filter(ttft, self.cfg.iqr_multiplier) if self.cfg.outlier_method == "iqr" else ttft[:]
        filtered_per_token = iqr_filter(per_token, self.cfg.iqr_multiplier) if self.cfg.outlier_method == "iqr" else per_token[:]
        filtered_e2e = iqr_filter(e2e, self.cfg.iqr_multiplier) if self.cfg.outlier_method == "iqr" else e2e[:]

        ttft_bounds = iqr_bounds(ttft, self.cfg.iqr_multiplier)
        per_token_bounds = iqr_bounds(per_token, self.cfg.iqr_multiplier)
        e2e_bounds = iqr_bounds(e2e, self.cfg.iqr_multiplier)

        result = {
            "result_schema_version": "2.0",
            "goal2_component_breakdown_schema": [
                "embedding_lookup_ms",
                "attention_qkv_ms",
                "attention_softmax_ms",
                "attention_projection_ms",
                "kv_cache_rw_estimated_ms",
                "mlp_total_ms",
                "layernorm_total_ms",
                "residual_ms",
                "lm_head_ms",
                "sampling_decoding_ms",
                "framework_overhead_ms",
                "total_token_ms",
            ],
            "measurement_metadata": {
                "timing_method": "forward hooks + synchronized wall-clock timing",
                "kv_cache_measurement_note": "KV-cache transfer time is estimated analytically from bytes moved and an effective bandwidth value; it is not directly isolated by a hardware profiler.",
                "effective_bandwidth_gbps": self.effective_bandwidth_gbps,
                "component_mapping": _goal2_breakdown_mapping(),
            },
            "config": asdict(self.cfg),
            "runtime": {
                "device": str(self.device),
                "dtype": str(self.dtype).replace("torch.", ""),
                "effective_bandwidth_gbps": self.effective_bandwidth_gbps,
            },
            "trials": [t.to_dict() for t in trials],
            "summary": {
                "raw": {
                    "ttft_ms": summarize(ttft),
                    "avg_per_token_ms": summarize(per_token),
                    "e2e_ms": summarize(e2e),
                },
                "filtered": {
                    "ttft_ms": summarize(filtered_ttft),
                    "avg_per_token_ms": summarize(filtered_per_token),
                    "e2e_ms": summarize(filtered_e2e),
                },
                "outlier_handling": {
                    "method": self.cfg.outlier_method,
                    "iqr_multiplier": self.cfg.iqr_multiplier,
                    "kept_trials": {
                        "ttft": len(filtered_ttft),
                        "avg_per_token": len(filtered_per_token),
                        "e2e": len(filtered_e2e),
                    },
                    "bounds": {
                        "ttft_ms": ttft_bounds,
                        "avg_per_token_ms": per_token_bounds,
                        "e2e_ms": e2e_bounds,
                    },
                },
            },
        }

        save_dir = ensure_dir(self.cfg.save_dir)
        suffix = "kv_on" if self.cfg.use_kv_cache else "kv_off"
        save_path = save_dir / f"single_benchmark_{suffix}_{timestamp_slug()}.json"
        json_dump(result, save_path)
        result["saved_to"] = str(save_path)
        return result