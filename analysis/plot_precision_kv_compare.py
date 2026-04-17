from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_latest_matrix(results_dir: Path) -> dict:
    files = sorted(results_dir.glob('precision_kv_matrix_*.json'))
    if not files:
        raise FileNotFoundError('No precision_kv_matrix_*.json file found in results/.')
    latest = files[-1]
    return json.loads(latest.read_text(encoding='utf-8'))


def main() -> None:
    results_dir = Path('results')
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    payload = load_latest_matrix(results_dir)
    rows = payload['rows']

    precision_order = ['float32', 'float16', 'bfloat16', 'auto']
    rows = sorted(rows, key=lambda r: (precision_order.index(r['precision']) if r['precision'] in precision_order else 999, r['use_kv_cache']))

    labels = [f"{r['precision']}\n{'KV on' if r['use_kv_cache'] else 'KV off'}" for r in rows]

    metrics = [
        ('ttft_mean_ms', 'TTFT mean (ms)', 'precision_kv_ttft.png'),
        ('per_token_mean_ms', 'Per-token mean (ms)', 'precision_kv_per_token.png'),
        ('e2e_mean_ms', 'End-to-end mean (ms)', 'precision_kv_e2e.png'),
    ]

    for metric_key, ylabel, filename in metrics:
        values = [r.get(metric_key) for r in rows]
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values)
        plt.ylabel(ylabel)
        plt.title(ylabel + ' by precision and KV-cache setting')
        plt.tight_layout()
        out_path = plots_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f'Saved {out_path}')

    # markdown summary table
    md_lines = [
        '# Precision + KV Cache Comparison',
        '',
        '| Precision | KV Cache | TTFT Mean (ms) | Per-token Mean (ms) | E2E Mean (ms) |',
        '|---|---|---:|---:|---:|',
    ]
    for r in rows:
        md_lines.append(
            f"| {r['precision']} | {'on' if r['use_kv_cache'] else 'off'} | {r['ttft_mean_ms']:.3f} | {r['per_token_mean_ms']:.3f} | {r['e2e_mean_ms']:.3f} |"
        )

    md_lines.extend(['', '## Pairwise KV-cache benefit by precision', ''])
    for item in payload.get('pairwise_comparisons', []):
        speedup = item.get('per_token_speedup_with_kv')
        delta = item.get('per_token_delta_ms')
        ttft_delta = item.get('ttft_delta_ms')
        e2e_delta = item.get('e2e_delta_ms')
        md_lines.append(
            f"- {item['precision']}: with KV cache gave {speedup:.3f}x per-token speedup, {delta:.3f} ms lower per-token latency, {ttft_delta:.3f} ms TTFT delta, and {e2e_delta:.3f} ms E2E delta."
        )

    summary_path = Path('docs/precision_kv_compare_report.md')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    print(f'Saved {summary_path}')


if __name__ == '__main__':
    main()
