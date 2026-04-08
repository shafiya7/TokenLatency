from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class KVCacheSpec:
    """
    Simple analytical model for KV-cache footprint and bandwidth pressure
    during autoregressive decoding.
    """

    model_name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    key_dtype_bytes: float = 2.0
    value_dtype_bytes: float = 2.0

    @property
    def bytes_per_token_per_layer(self) -> float:
        key_bytes = self.num_kv_heads * self.head_dim * self.key_dtype_bytes
        value_bytes = self.num_kv_heads * self.head_dim * self.value_dtype_bytes
        return key_bytes + value_bytes

    @property
    def bytes_per_token_total(self) -> float:
        return self.bytes_per_token_per_layer * self.num_layers

    def total_cache_bytes(self, seq_len: int) -> float:
        return self.bytes_per_token_total * seq_len

    def total_cache_mb(self, seq_len: int) -> float:
        return self.total_cache_bytes(seq_len) / (1024 ** 2)

    def decode_read_bytes_one_step(self, seq_len: int) -> float:
        """
        Approximate bytes read from KV-cache for generating ONE new token.
        During one decode step, attention reads all previous K/V across all layers.
        """
        return self.total_cache_bytes(seq_len)

    def decode_read_mb_one_step(self, seq_len: int) -> float:
        return self.decode_read_bytes_one_step(seq_len) / (1024 ** 2)

    def estimated_transfer_ms(self, seq_len: int, bandwidth_gbps: float) -> float:
        """
        Estimate lower-bound transfer time from memory bandwidth.
        bandwidth_gbps is in GB/s.
        """
        if bandwidth_gbps <= 0:
            return 0.0
        gb = self.decode_read_bytes_one_step(seq_len) / (1024 ** 3)
        return (gb / bandwidth_gbps) * 1000.0

    def threshold_seq_len(self, cache_size_bytes: float) -> int:
        if self.bytes_per_token_total <= 0:
            return 0
        return int(cache_size_bytes // self.bytes_per_token_total)

    def table(self, seq_lengths: Iterable[int], bandwidth_gbps: float | None = None) -> list[dict]:
        rows = []
        for seq_len in seq_lengths:
            row = {
                "model_name": self.model_name,
                "seq_len": seq_len,
                "kv_bytes_per_token_total": self.bytes_per_token_total,
                "kv_total_bytes": self.total_cache_bytes(seq_len),
                "kv_total_mb": self.total_cache_mb(seq_len),
                "decode_read_mb_one_step": self.decode_read_mb_one_step(seq_len),
            }
            if bandwidth_gbps is not None:
                row["estimated_transfer_ms"] = self.estimated_transfer_ms(seq_len, bandwidth_gbps)
            rows.append(row)
        return rows


DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "q8": 1.0,
    "int4": 0.5,
    "q4": 0.5,
}


KNOWN_MODELS: dict[str, KVCacheSpec] = {
    # TinyLlama 1.1B
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": KVCacheSpec(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_layers=22,
        num_kv_heads=4,
        head_dim=64,
        key_dtype_bytes=2.0,
        value_dtype_bytes=2.0,
    ),

    # Meta Llama 3.2 1B
    "meta-llama/Llama-3.2-1B-Instruct": KVCacheSpec(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        num_layers=16,
        num_kv_heads=8,
        head_dim=64,
        key_dtype_bytes=2.0,
        value_dtype_bytes=2.0,
    ),

    # Meta Llama 3.2 3B
    "meta-llama/Llama-3.2-3B-Instruct": KVCacheSpec(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        key_dtype_bytes=2.0,
        value_dtype_bytes=2.0,
    ),
}


def build_spec(
    model_name: str,
    key_precision: str = "fp16",
    value_precision: str = "fp16",
) -> KVCacheSpec:
    if model_name not in KNOWN_MODELS:
        raise ValueError(f"Unsupported model for analytical KV model: {model_name}")

    base = KNOWN_MODELS[model_name]
    return KVCacheSpec(
        model_name=base.model_name,
        num_layers=base.num_layers,
        num_kv_heads=base.num_kv_heads,
        head_dim=base.head_dim,
        key_dtype_bytes=DTYPE_BYTES.get(key_precision, 2.0),
        value_dtype_bytes=DTYPE_BYTES.get(value_precision, 2.0),
    )


def compare_precisions(
    model_name: str,
    seq_lengths: Iterable[int],
    precisions: Iterable[str] = ("fp16", "int8", "int4"),
) -> list[dict]:
    rows = []
    for precision in precisions:
        spec = build_spec(model_name, key_precision=precision, value_precision=precision)
        for row in spec.table(seq_lengths):
            row["precision"] = precision
            rows.append(row)
    return rows


def cache_thresholds(model_name: str) -> list[dict]:
    """
    Show sequence lengths where KV-cache exceeds common cache capacities.
    """
    spec = build_spec(model_name)

    cache_sizes = {
        "256KB": 256 * 1024,
        "1MB": 1 * 1024 * 1024,
        "8MB": 8 * 1024 * 1024,
        "32MB": 32 * 1024 * 1024,
        "64MB": 64 * 1024 * 1024,
    }

    rows = []
    for label, size_bytes in cache_sizes.items():
        rows.append({
            "model_name": model_name,
            "cache_size": label,
            "threshold_seq_len": spec.threshold_seq_len(size_bytes),
        })
    return rows


if __name__ == "__main__":
    seqs = [16, 32, 64, 128, 256, 512]

    model = "meta-llama/Llama-3.2-3B-Instruct"
    spec = build_spec(model, key_precision="fp16", value_precision="fp16")

    print(f"\nKV Cache Model for: {model}")
    print(f"bytes_per_token_total = {spec.bytes_per_token_total:.0f} bytes\n")

    print("Per-sequence footprint:")
    for row in spec.table(seqs, bandwidth_gbps=100.0):
        print(row)

    print("\nPrecision comparison:")
    for row in compare_precisions(model, seqs):
        print(row)

    print("\nCache thresholds:")
    for row in cache_thresholds(model):
        print(row)