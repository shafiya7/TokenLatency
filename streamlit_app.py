from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Make project imports work
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.config import BenchmarkConfig
from bench.harness import HFLatencyHarness
from bench.model_registry import list_models


st.set_page_config(
    page_title="LLaMA Token Latency Benchmark",
    page_icon="⚡",
    layout="wide",
)

DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def get_available_models() -> list[str]:
    try:
        models = list_models("configs/models.json")
        if models:
            return models
    except Exception:
        pass
    return DEFAULT_MODELS


def run_single_benchmark(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    warmup_runs: int,
    measured_trials: int,
    device: str,
    dtype: str,
    use_kv_cache: bool,
) -> dict[str, Any]:
    cfg = BenchmarkConfig(
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        measured_trials=measured_trials,
        device_preference=device,
        torch_dtype=dtype,
        use_kv_cache=use_kv_cache,
        save_dir="results/ui_runs",
        system_name="streamlit_app",
    )

    harness = HFLatencyHarness(cfg)
    try:
        result = harness.run()
    finally:
        harness.close()

    return result


def metric_value(result: dict[str, Any], metric_key: str) -> float:
    return float(result["summary"]["filtered"][metric_key]["mean"])


def get_first_output_text(result: dict[str, Any]) -> str:
    trials = result.get("trials", [])
    if not trials:
        return ""
    return trials[0].get("output_text", "")


def build_summary_row(result: dict[str, Any], run_label: str) -> dict[str, Any]:
    trials = result.get("trials", [])
    prompt_tokens = trials[0].get("prompt_tokens") if trials else None
    generated_tokens = trials[0].get("generated_tokens") if trials else None

    return {
        "Run": run_label,
        "Model": result["config"]["model_name"],
        "Device": result["runtime"]["device"],
        "Dtype": result["runtime"]["dtype"],
        "KV cache": result["config"].get("use_kv_cache", True),
        "TTFT (ms)": round(metric_value(result, "ttft_ms"), 3),
        "Per-token (ms)": round(metric_value(result, "avg_per_token_ms"), 3),
        "E2E (ms)": round(metric_value(result, "e2e_ms"), 3),
        "Prompt tokens": prompt_tokens,
        "Generated tokens": generated_tokens,
    }


def build_trials_df(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for trial in result.get("trials", []):
        rows.append(
            {
                "trial_index": trial.get("trial_index"),
                "prompt_tokens": trial.get("prompt_tokens"),
                "generated_tokens": trial.get("generated_tokens"),
                "ttft_ms": trial.get("ttft_ms"),
                "avg_per_token_ms": trial.get("avg_per_token_ms"),
                "e2e_ms": trial.get("e2e_ms"),
            }
        )
    return pd.DataFrame(rows)


def build_breakdown_df(result: dict[str, Any]) -> pd.DataFrame:
    trials = result.get("trials", [])
    if not trials:
        return pd.DataFrame()

    breakdown = trials[0].get("steady_state_breakdown_avg_ms", {})
    if not breakdown:
        return pd.DataFrame()

    rows = [{"Component": k, "Latency (ms)": v} for k, v in breakdown.items()]
    df = pd.DataFrame(rows).sort_values("Latency (ms)", ascending=False)
    return df


def render_model_card(title: str, result: dict[str, Any], run_label: str) -> None:
    st.subheader(title)

    c1, c2, c3 = st.columns(3)
    c1.metric("TTFT", f"{metric_value(result, 'ttft_ms'):.3f} ms")
    c2.metric("Per-token", f"{metric_value(result, 'avg_per_token_ms'):.3f} ms")
    c3.metric("E2E", f"{metric_value(result, 'e2e_ms'):.3f} ms")

    st.caption(
        f"Run: {run_label} | "
        f"Model: {result['config']['model_name']} | "
        f"Device: {result['runtime']['device']} | "
        f"Dtype: {result['runtime']['dtype']} | "
        f"KV cache: {result['config'].get('use_kv_cache', True)}"
    )

    with st.expander("Generated output", expanded=False):
        st.write(get_first_output_text(result) or "No output text found.")

    with st.expander("Trial details", expanded=False):
        trials_df = build_trials_df(result)
        if not trials_df.empty:
            st.dataframe(trials_df, use_container_width=True)
        else:
            st.info("No trial table available.")

    with st.expander("Steady-state token breakdown", expanded=False):
        breakdown_df = build_breakdown_df(result)
        if not breakdown_df.empty:
            st.dataframe(breakdown_df, use_container_width=True)
            st.bar_chart(
                breakdown_df.set_index("Component")["Latency (ms)"],
                use_container_width=True,
            )
        else:
            st.info("No breakdown data available.")


st.title("⚡ LLaMA Token Latency Benchmark Dashboard")
st.write("Compare TTFT, per-token latency, and end-to-end latency for two models with and without KV cache.")

available_models = get_available_models()

with st.sidebar:
    st.header("Benchmark Settings")

    prompt = st.text_area(
        "Prompt",
        value="Explain TTFT, per-token latency, and end-to-end latency in decoder-only LLaMA inference.",
        height=180,
    )

    model_a = st.selectbox(
        "Model A",
        options=available_models,
        index=0 if available_models else 0,
    )

    default_b_index = 1 if len(available_models) > 1 else 0
    model_b = st.selectbox(
        "Model B",
        options=available_models,
        index=default_b_index,
    )

    max_new_tokens = st.slider("Max new tokens", 8, 128, 24, 1)
    warmup_runs = st.slider("Warmup runs", 0, 5, 1, 1)
    measured_trials = st.slider("Measured trials", 1, 10, 3, 1)

    device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
    dtype = st.selectbox("Dtype", ["auto", "float16", "float32", "bfloat16"], index=0)

    run_button = st.button("Run comparison", type="primary", use_container_width=True)

if run_button:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    run_plan = [
        ("Model A | KV cache ON", model_a, True),
        ("Model A | KV cache OFF", model_a, False),
        ("Model B | KV cache ON", model_b, True),
        ("Model B | KV cache OFF", model_b, False),
    ]

    results: dict[str, dict[str, Any]] = {}
    progress = st.progress(0)
    status = st.empty()

    try:
        total_runs = len(run_plan)

        for i, (label, model_name, use_kv_cache) in enumerate(run_plan, start=1):
            status.info(f"Running {label}")
            results[label] = run_single_benchmark(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                warmup_runs=warmup_runs,
                measured_trials=measured_trials,
                device=device,
                dtype=dtype,
                use_kv_cache=use_kv_cache,
            )
            progress.progress(int(i / total_runs * 100))

        status.success("All benchmark runs complete.")

    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.divider()
    st.subheader("Summary Table")

    summary_rows = [
        build_summary_row(result, run_label)
        for run_label, result in results.items()
    ]
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    st.divider()
    st.subheader("Charts")

    chart_rows = []
    for run_label, result in results.items():
        model_name = result["config"]["model_name"]
        cache_label = "KV cache ON" if result["config"].get("use_kv_cache", True) else "KV cache OFF"

        chart_rows.extend(
            [
                {
                    "Run": run_label,
                    "Model": model_name,
                    "Cache": cache_label,
                    "Metric": "TTFT (ms)",
                    "Value": metric_value(result, "ttft_ms"),
                },
                {
                    "Run": run_label,
                    "Model": model_name,
                    "Cache": cache_label,
                    "Metric": "Per-token (ms)",
                    "Value": metric_value(result, "avg_per_token_ms"),
                },
                {
                    "Run": run_label,
                    "Model": model_name,
                    "Cache": cache_label,
                    "Metric": "E2E (ms)",
                    "Value": metric_value(result, "e2e_ms"),
                },
            ]
        )

    chart_df = pd.DataFrame(chart_rows)

    st.markdown("### TTFT")
    ttft_df = chart_df[chart_df["Metric"] == "TTFT (ms)"].pivot(
        index="Model", columns="Cache", values="Value"
    )
    st.bar_chart(ttft_df, use_container_width=True)

    st.markdown("### Per-token latency")
    per_token_df = chart_df[chart_df["Metric"] == "Per-token (ms)"].pivot(
        index="Model", columns="Cache", values="Value"
    )
    st.bar_chart(per_token_df, use_container_width=True)

    st.markdown("### End-to-end latency")
    e2e_df = chart_df[chart_df["Metric"] == "E2E (ms)"].pivot(
        index="Model", columns="Cache", values="Value"
    )
    st.bar_chart(e2e_df, use_container_width=True)

    st.divider()
    st.subheader("Cache vs No-cache improvement")

    improvement_rows = []
    for model_name in [model_a, model_b]:
        on_result = None
        off_result = None

        for _, result in results.items():
            if result["config"]["model_name"] == model_name:
                if result["config"].get("use_kv_cache", True):
                    on_result = result
                else:
                    off_result = result

        if on_result and off_result:
            ttft_on = metric_value(on_result, "ttft_ms")
            ttft_off = metric_value(off_result, "ttft_ms")
            per_on = metric_value(on_result, "avg_per_token_ms")
            per_off = metric_value(off_result, "avg_per_token_ms")
            e2e_on = metric_value(on_result, "e2e_ms")
            e2e_off = metric_value(off_result, "e2e_ms")

            improvement_rows.append(
                {
                    "Model": model_name,
                    "TTFT saved by cache (ms)": round(ttft_off - ttft_on, 3),
                    "Per-token saved by cache (ms)": round(per_off - per_on, 3),
                    "E2E saved by cache (ms)": round(e2e_off - e2e_on, 3),
                }
            )

    if improvement_rows:
        improvement_df = pd.DataFrame(improvement_rows)
        st.dataframe(improvement_df, use_container_width=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        render_model_card("Model A — KV cache ON", results["Model A | KV cache ON"], "Model A | KV cache ON")
        render_model_card("Model A — KV cache OFF", results["Model A | KV cache OFF"], "Model A | KV cache OFF")

    with c2:
        render_model_card("Model B — KV cache ON", results["Model B | KV cache ON"], "Model B | KV cache ON")
        render_model_card("Model B — KV cache OFF", results["Model B | KV cache OFF"], "Model B | KV cache OFF")

else:
    st.info("Choose two models, enter a prompt, and click 'Run comparison'.")