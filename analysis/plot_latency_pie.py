from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "results" / "breakdown_summary.csv"
OUT_FILE = ROOT / "analysis" / "plots" / "latency_pie.png"

# Anything below this percent gets merged into "Other"
MIN_PERCENT = 1.0


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Could not find: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    # Optional filters if you want one specific setup
    # df = df[df["model_name"] == "meta-llama/Llama-3.2-1B-Instruct"]
    # df = df[df["dtype"] == "float16"]
    # df = df[df["file"].str.contains("kv_on", na=False)]

    if df.empty:
        raise ValueError("No data found after filtering")

    component_cols = {
        "Attention": "steady_attention_total_ms",
        "MLP": "steady_mlp_total_ms",
        "KV Cache": "steady_kv_cache_rw_estimated_ms",
        "Embedding": "steady_embedding_lookup_ms",
        "LayerNorm": "steady_layernorm_total_ms",
        "Residual": "steady_residual_ms",
        "LM Head": "steady_lm_head_ms",
        "Sampling": "steady_sampling_decoding_ms",
        "Framework": "steady_framework_overhead_ms",
    }

    raw_values = {}
    for label, col in component_cols.items():
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").dropna().mean()
            if pd.notna(val) and val > 0:
                raw_values[label] = float(val)

    if not raw_values:
        raise ValueError("No valid latency values found in breakdown_summary.csv")

    total = sum(raw_values.values())
    kept = {}
    other_total = 0.0

    for label, value in raw_values.items():
        pct = (value / total) * 100
        if pct < MIN_PERCENT:
            other_total += value
        else:
            kept[label] = value

    if other_total > 0:
        kept["Other"] = other_total

    # sort biggest to smallest
    kept = dict(sorted(kept.items(), key=lambda x: x[1], reverse=True))

    labels = list(kept.keys())
    sizes = list(kept.values())

    def autopct_format(pct):
        return f"{pct:.1f}%" if pct >= MIN_PERCENT else ""

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))

    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        autopct=autopct_format,
        startangle=140,
        pctdistance=0.70,
        labeldistance=1.08,
    )

    plt.title("Latency Breakdown per Token (from breakdown_summary.csv)", fontsize=16)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved clean pie chart to: {OUT_FILE}")
    print("\nValues used (ms):")
    for k, v in kept.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()