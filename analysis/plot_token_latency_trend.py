import json
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("benchmarks/results")

def load_data():
    data = []
    for f in RESULTS_DIR.glob("*.json"):
        with open(f) as fp:
            j = json.load(fp)
            data.append(j)
    return data

def main():
    data = load_data()

    plt.figure()

    for d in data:
        if "per_token_latencies_ms" in d and "model_name" in d:
            latencies = d["per_token_latencies_ms"]
            token_idx = list(range(1, len(latencies) + 1))

            plt.plot(token_idx, latencies, label=d["model_name"])

    plt.xlabel("Generated Token Index")
    plt.ylabel("Latency per Token (ms)")
    plt.title("Per-Token Latency Stabilization (KV Cache Behavior)")

    plt.legend()
    plt.grid()

    plt.savefig("analysis/plots/09_token_latency_trend.png")
    print("Saved: 09_token_latency_trend.png")

if __name__ == "__main__":
    main()