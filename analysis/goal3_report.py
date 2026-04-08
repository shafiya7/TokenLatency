from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _label(model_name: str) -> str:
    return model_name.split('/')[-1]


def main() -> None:
    results_dir = Path('results')
    scaling_files = sorted(results_dir.glob('scaling_*.json'))
    if not scaling_files:
        print('No scaling results found.')
        return

    latest = scaling_files[-1]
    data = json.loads(latest.read_text(encoding='utf-8'))
    rows = data['rows']

    by_combo = defaultdict(list)
    for row in rows:
        by_combo[(row['model_name'], row.get('precision', 'auto'))].append(row)

    lines = ['# Goal 3 Scaling Analysis', '']
    lines.append(f"**Source file:** `{latest}`")
    lines.append('')
    lines.append('## Experiment Coverage')
    lines.append('')
    lines.append(f"- Models: {len({r['model_name'] for r in rows})}")
    lines.append(f"- Precisions: {', '.join(sorted({r.get('precision', 'auto') for r in rows}))}")
    lines.append(f"- Prompt lengths: {', '.join(str(x) for x in sorted({r['prompt_length_tokens'] for r in rows}))}")
    lines.append('')
    lines.append('## Scaling Table')
    lines.append('')
    lines.append('| Model | Precision | Prompt Tokens | TTFT Mean (ms) | Per-token Mean (ms) | E2E Mean (ms) |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for row in sorted(rows, key=lambda r: (r['model_name'], r.get('precision', 'auto'), r['prompt_length_tokens'])):
        lines.append(
            f"| {_label(row['model_name'])} | {row.get('precision', 'auto')} | {row['prompt_length_tokens']} | {row['ttft_mean_ms']:.3f} | {row['per_token_mean_ms']:.3f} | {row['e2e_mean_ms']:.3f} |"
        )

    lines.extend(['', '## Dominant Trends', ''])
    for (model_name, precision), group in sorted(by_combo.items()):
        group = sorted(group, key=lambda r: r['prompt_length_tokens'])
        if len(group) < 2:
            continue
        first = group[0]
        last = group[-1]
        ttft_growth = last['ttft_mean_ms'] - first['ttft_mean_ms']
        token_growth = last['per_token_mean_ms'] - first['per_token_mean_ms']
        lines.append(
            f"- {_label(model_name)} [{precision}] from {first['prompt_length_tokens']}→{last['prompt_length_tokens']} tokens: TTFT changed by {ttft_growth:.3f} ms and per-token latency changed by {token_growth:.3f} ms."
        )

    lines.extend(['', '## Interpretation', ''])
    lines.append('- Sequence length typically affects TTFT first through the prompt pass, then increasingly affects steady-state decode as the KV-cache grows.')
    lines.append('- Model size changes both TTFT and per-token latency because larger variants have more layers and wider hidden states, increasing attention and MLP cost.')
    lines.append('- Precision differences matter most when the backend truly accelerates lower precision or reduces memory traffic; otherwise, gains may be modest.')
    lines.append('- The strongest bottleneck should be confirmed with the Goal 2 breakdown. If attention grows faster than MLP as context increases, that points to decode-time memory pressure and KV-cache traffic.')

    out = Path('docs/goal3_scaling_analysis.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f'Saved Goal 3 report to: {out}')


if __name__ == '__main__':
    main()
