from dataclasses import dataclass, field


META_LLAMA_VARIANTS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


@dataclass(slots=True)
class BenchmarkConfig:
    model_name: str = META_LLAMA_VARIANTS[0]
    prompt: str = "Explain token-generation latency in a decoder-only LLaMA model."
    max_new_tokens: int = 24
    warmup_runs: int = 2
    measured_trials: int = 5
    outlier_method: str = "iqr"
    iqr_multiplier: float = 1.5
    device_preference: str = "auto"   # auto | cpu | cuda | mps
    torch_dtype: str = "auto"         # auto | float32 | float16 | bfloat16
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    save_dir: str = "results"
    system_name: str = "local_machine"
    estimated_bandwidth_gbps: float | None = None
    trust_remote_code: bool = False
    use_kv_cache: bool = True


@dataclass(slots=True)
class ScalingConfig:
    model_names: list[str] = field(default_factory=lambda: META_LLAMA_VARIANTS[:2])
    prompt_lengths: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    precisions: list[str] = field(default_factory=lambda: ["auto"])
    max_new_tokens: int = 24
    warmup_runs: int = 1
    measured_trials: int = 3
    device_preference: str = "auto"
    save_dir: str = "results"
    system_name: str = "local_machine"
    estimated_bandwidth_gbps: float | None = None
    use_kv_cache: bool = True
