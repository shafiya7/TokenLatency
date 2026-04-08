# from __future__ import annotations

# import json
# from collections import defaultdict
# from pathlib import Path

# import matplotlib.pyplot as plt


# def _safe_label(model_name: str) -> str:
#     return model_name.split('/')[-1]


# def _detect_inflections(rows: list[dict], metric: str) -> list[dict]:
#     rows = sorted(rows, key=lambda x: x['prompt_length_tokens'])
#     notes: list[dict] = []
#     previous_slope = None
#     for left, right in zip(rows, rows[1:]):
#         dx = right['prompt_length_tokens'] - left['prompt_length_tokens']
#         if dx <= 0:
#             continue
#         slope = (right[metric] - left[metric]) / dx
#         ratio = None if previous_slope in (None, 0) else slope / previous_slope
#         if previous_slope is not None and ratio is not None and ratio >= 1.5:
#             notes.append({
#                 'near_tokens': right['prompt_length_tokens'],
#                 'metric': metric,
#                 'previous_slope_ms_per_token': previous_slope,
#                 'current_slope_ms_per_token': slope,
#                 'ratio': ratio,
#             })
#         previous_slope = slope
#     return notes


# def _dominant_bottleneck_from_result(result_path: str) -> dict[str, float]:
#     data = json.loads(Path(result_path).read_text(encoding='utf-8'))
#     trial = data['trials'][0]
#     steady = trial.get('steady_state_breakdown_avg_ms', {})
#     return {
#         'attention_total_ms': steady.get('attention_total_ms', 0.0),
#         'mlp_total_ms': steady.get('mlp_total_ms', 0.0),
#         'layernorm_total_ms': steady.get('layernorm_total_ms', 0.0),
#         'sampling_decoding_ms': steady.get('sampling_decoding_ms', 0.0),
#         'framework_overhead_ms': steady.get('framework_overhead_ms', 0.0),
#         'kv_cache_rw_ms': steady.get('kv_cache_rw_ms', 0.0),
#     }


# def _write_goal3_summary(data: dict, summary_path: Path) -> None:
#     rows = data['rows']
#     by_combo = defaultdict(list)
#     for row in rows:
#         key = (row['model_name'], row.get('precision', 'auto'))
#         by_combo[key].append(row)

#     inflections = []
#     for key, group in by_combo.items():
#         notes = _detect_inflections(group, 'per_token_mean_ms')
#         for note in notes:
#             inflections.append((key, note))

#     lines = ['# Goal 3 Scaling Summary', '']
#     lines.append('## Sequence-Length Scaling Snapshot')
#     lines.append('')
#     lines.append('| Model | Precision | Prompt Tokens | TTFT Mean (ms) | Per-token Mean (ms) | E2E Mean (ms) | Dominant Bottleneck |')
#     lines.append('|---|---|---:|---:|---:|---:|---|')

#     for row in sorted(rows, key=lambda r: (r['model_name'], r.get('precision', 'auto'), r['prompt_length_tokens'])):
#         bottlenecks = _dominant_bottleneck_from_result(row['result_file'])
#         dominant = max(bottlenecks.items(), key=lambda kv: kv[1])[0].replace('_ms', '')
#         lines.append(
#             f"| {_safe_label(row['model_name'])} | {row.get('precision', 'auto')} | {row['prompt_length_tokens']} | {row['ttft_mean_ms']:.3f} | {row['per_token_mean_ms']:.3f} | {row['e2e_mean_ms']:.3f} | {dominant} |"
#         )

#     lines.extend(['', '## Inflection-Point Notes', ''])
#     if inflections:
#         for (model_name, precision), note in inflections:
#             lines.append(
#                 f"- {_safe_label(model_name)} [{precision}] shows a possible inflection near {note['near_tokens']} tokens: per-token slope rose from {note['previous_slope_ms_per_token']:.4f} to {note['current_slope_ms_per_token']:.4f} ms/token (x{note['ratio']:.2f})."
#             )
#     else:
#         lines.append('- No strong inflection points were detected by the slope-ratio heuristic on the current scaling sweep.')

#     lines.extend(['', '## Dominant Bottleneck Interpretation', ''])
#     lines.append('- Increasing prompt length primarily stresses attention and KV-cache traffic, so per-token latency often rises faster than TTFT after context becomes moderately large.')
#     lines.append('- Larger models increase both attention and MLP cost because there are more parameters, wider hidden states, and usually more layers.')
#     lines.append('- Reduced precision can improve latency when the backend and hardware meaningfully accelerate lower-precision kernels or reduce memory traffic.')
#     lines.append('- If framework overhead becomes visible at short prompts, that usually means launch/synchronization costs are non-trivial relative to the actual compute.')
#     summary_path.write_text("\n".join(lines) + "\n", encoding='utf-8')


# def main() -> None:
#     results_dir = Path('results')
#     scaling_files = sorted(results_dir.glob('scaling_*.json'))
#     if not scaling_files:
#         print('No scaling files found.')
#         return

#     latest = scaling_files[-1]
#     data = json.loads(latest.read_text(encoding='utf-8'))
#     rows = data['rows']

#     plots_dir = Path('plots')
#     plots_dir.mkdir(parents=True, exist_ok=True)

#     grouped = defaultdict(list)
#     for row in rows:
#         key = f"{_safe_label(row['model_name'])} [{row.get('precision', 'auto')}]"
#         grouped[key].append(row)

#     for metric in ['ttft_mean_ms', 'per_token_mean_ms', 'e2e_mean_ms']:
#         plt.figure(figsize=(8, 5))
#         for label, model_rows in grouped.items():
#             model_rows = sorted(model_rows, key=lambda x: x['prompt_length_tokens'])
#             xs = [r['prompt_length_tokens'] for r in model_rows]
#             ys = [r[metric] for r in model_rows]
#             plt.plot(xs, ys, marker='o', label=label)
#         plt.xlabel('Prompt length (tokens)')
#         plt.ylabel(metric.replace('_', ' '))
#         plt.title(metric.replace('_', ' ').title())
#         plt.legend()
#         out_path = plots_dir / f'{metric}.png'
#         plt.tight_layout()
#         plt.savefig(out_path, dpi=150)
#         plt.close()
#         print('Saved', out_path)

#     common_lengths = sorted({r['prompt_length_tokens'] for r in rows})
#     if common_lengths:
#         target_len = common_lengths[-1]
#         plt.figure(figsize=(8, 5))
#         model_rows = [r for r in rows if r['prompt_length_tokens'] == target_len]
#         model_rows = sorted(model_rows, key=lambda r: (_safe_label(r['model_name']), r.get('precision', 'auto')))
#         labels = [f"{_safe_label(r['model_name'])} [{r.get('precision', 'auto')}]" for r in model_rows]
#         values = [r['per_token_mean_ms'] for r in model_rows]
#         plt.bar(labels, values)
#         plt.xticks(rotation=25, ha='right')
#         plt.ylabel('per token mean ms')
#         plt.title(f'Model-size comparison at {target_len} prompt tokens')
#         out_path = plots_dir / 'model_size_comparison_per_token.png'
#         plt.tight_layout()
#         plt.savefig(out_path, dpi=150)
#         plt.close()
#         print('Saved', out_path)

#     precisions = sorted({r.get('precision', 'auto') for r in rows})
#     if len(precisions) > 1:
#         first_model = sorted({r['model_name'] for r in rows})[0]
#         precision_rows = [r for r in rows if r['model_name'] == first_model]
#         precision_rows = sorted(precision_rows, key=lambda r: (r.get('precision', 'auto'), r['prompt_length_tokens']))
#         by_precision = defaultdict(list)
#         for row in precision_rows:
#             by_precision[row.get('precision', 'auto')].append(row)
#         plt.figure(figsize=(8, 5))
#         for precision, group in by_precision.items():
#             group = sorted(group, key=lambda r: r['prompt_length_tokens'])
#             plt.plot([g['prompt_length_tokens'] for g in group], [g['per_token_mean_ms'] for g in group], marker='o', label=precision)
#         plt.xlabel('Prompt length (tokens)')
#         plt.ylabel('per token mean ms')
#         plt.title(f'Precision sweep for {_safe_label(first_model)}')
#         plt.legend()
#         out_path = plots_dir / 'precision_comparison_per_token.png'
#         plt.tight_layout()
#         plt.savefig(out_path, dpi=150)
#         plt.close()
#         print('Saved', out_path)

#     summary_path = Path('docs/goal3_scaling_report.md')
#     summary_path.parent.mkdir(parents=True, exist_ok=True)
#     _write_goal3_summary(data, summary_path)
#     print('Saved', summary_path)

#     inflection_notes = []
#     by_model_precision = defaultdict(list)
#     for row in rows:
#         key = (row['model_name'], row.get('precision', 'auto'))
#         by_model_precision[key].append(row)
#     for (model_name, precision), grp in by_model_precision.items():
#         notes = _detect_inflections(grp, 'per_token_mean_ms')
#         for note in notes:
#             inflection_notes.append(
#                 f"- {_safe_label(model_name)} [{precision}] possible inflection near {note['near_tokens']} tokens: slope rose from {note['previous_slope_ms_per_token']:.4f} to {note['current_slope_ms_per_token']:.4f} ms/token (x{note['ratio']:.2f})."
#             )
#     note_path = plots_dir / 'inflection_points.md'
#     note_path.write_text('# Scaling Inflection Notes\n\n' + ('\n'.join(inflection_notes) if inflection_notes else 'No strong inflection points detected by the simple slope heuristic.'), encoding='utf-8')
#     print('Saved', note_path)


# if __name__ == '__main__':
#     main()


from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _safe_label(model_name: str) -> str:
    return model_name.split('/')[-1]


def _detect_inflections(rows: list[dict], metric: str) -> list[dict]:
    rows = sorted(rows, key=lambda x: x['prompt_length_tokens'])
    notes: list[dict] = []
    previous_slope = None
    for left, right in zip(rows, rows[1:]):
        dx = right['prompt_length_tokens'] - left['prompt_length_tokens']
        if dx <= 0:
            continue
        slope = (right[metric] - left[metric]) / dx
        ratio = None if previous_slope in (None, 0) else slope / previous_slope
        if previous_slope is not None and ratio is not None and ratio >= 1.5:
            notes.append({
                'near_tokens': right['prompt_length_tokens'],
                'metric': metric,
                'previous_slope_ms_per_token': previous_slope,
                'current_slope_ms_per_token': slope,
                'ratio': ratio,
            })
        previous_slope = slope
    return notes


def _dominant_bottleneck_from_result(result_path: str) -> dict[str, float]:
    data = json.loads(Path(result_path).read_text(encoding='utf-8'))
    trial = data['trials'][0]
    steady = trial.get('steady_state_breakdown_avg_ms', {})
    return {
        'attention_total_ms': steady.get('attention_total_ms', 0.0),
        'mlp_total_ms': steady.get('mlp_total_ms', 0.0),
        'layernorm_total_ms': steady.get('layernorm_total_ms', 0.0),
        'sampling_decoding_ms': steady.get('sampling_decoding_ms', 0.0),
        'framework_overhead_ms': steady.get('framework_overhead_ms', 0.0),
        'kv_cache_rw_estimated_ms': steady.get('kv_cache_rw_estimated_ms', steady.get('kv_cache_rw_ms', 0.0)),
    }


def _write_goal3_summary(data: dict, summary_path: Path) -> None:
    rows = data['rows']
    by_combo = defaultdict(list)
    for row in rows:
        key = (row['model_name'], row.get('precision', 'auto'))
        by_combo[key].append(row)

    inflections = []
    for key, group in by_combo.items():
        notes = _detect_inflections(group, 'per_token_mean_ms')
        for note in notes:
            inflections.append((key, note))

    lines = ['# Goal 3 Scaling Summary', '']
    lines.append('## Sequence-Length Scaling Snapshot')
    lines.append('')
    lines.append('| Model | Precision | Prompt Tokens | TTFT Mean (ms) | Per-token Mean (ms) | E2E Mean (ms) | Dominant Bottleneck |')
    lines.append('|---|---|---:|---:|---:|---:|---|')

    for row in sorted(rows, key=lambda r: (r['model_name'], r.get('precision', 'auto'), r['prompt_length_tokens'])):
        bottlenecks = _dominant_bottleneck_from_result(row['result_file'])
        dominant = max(bottlenecks.items(), key=lambda kv: kv[1])[0].replace('_ms', '')
        lines.append(
            f"| {_safe_label(row['model_name'])} | {row.get('precision', 'auto')} | {row['prompt_length_tokens']} | {row['ttft_mean_ms']:.3f} | {row['per_token_mean_ms']:.3f} | {row['e2e_mean_ms']:.3f} | {dominant} |"
        )

    lines.extend(['', '## Inflection-Point Notes', ''])
    if inflections:
        for (model_name, precision), note in inflections:
            lines.append(
                f"- {_safe_label(model_name)} [{precision}] shows a possible inflection near {note['near_tokens']} tokens: per-token slope rose from {note['previous_slope_ms_per_token']:.4f} to {note['current_slope_ms_per_token']:.4f} ms/token (x{note['ratio']:.2f})."
            )
    else:
        lines.append('- No strong inflection points were detected by the slope-ratio heuristic on the current scaling sweep.')

    lines.extend(['', '## Dominant Bottleneck Interpretation', ''])
    lines.append('- Increasing prompt length primarily stresses attention and KV-cache traffic, so per-token latency often rises faster than TTFT after context becomes moderately large.')
    lines.append('- Larger models increase both attention and MLP cost because there are more parameters, wider hidden states, and usually more layers.')
    lines.append('- Reduced precision can improve latency when the backend and hardware meaningfully accelerate lower-precision kernels or reduce memory traffic.')
    lines.append('- If framework overhead becomes visible at short prompts, that usually means launch/synchronization costs are non-trivial relative to the actual compute.')
    summary_path.write_text("\n".join(lines) + "\n", encoding='utf-8')


def main() -> None:
    results_dir = Path('results')
    scaling_files = sorted(results_dir.glob('scaling_*.json'))
    if not scaling_files:
        print('No scaling files found.')
        return

    latest = scaling_files[-1]
    data = json.loads(latest.read_text(encoding='utf-8'))
    rows = data['rows']

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        key = f"{_safe_label(row['model_name'])} [{row.get('precision', 'auto')}]"
        grouped[key].append(row)

    for metric in ['ttft_mean_ms', 'per_token_mean_ms', 'e2e_mean_ms']:
        plt.figure(figsize=(8, 5))
        for label, model_rows in grouped.items():
            model_rows = sorted(model_rows, key=lambda x: x['prompt_length_tokens'])
            xs = [r['prompt_length_tokens'] for r in model_rows]
            ys = [r[metric] for r in model_rows]
            plt.plot(xs, ys, marker='o', label=label)
        plt.xlabel('Prompt length (tokens)')
        plt.ylabel(metric.replace('_', ' '))
        plt.title(metric.replace('_', ' ').title())
        plt.legend()
        out_path = plots_dir / f'{metric}.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print('Saved', out_path)

    common_lengths = sorted({r['prompt_length_tokens'] for r in rows})
    if common_lengths:
        target_len = common_lengths[-1]
        plt.figure(figsize=(8, 5))
        model_rows = [r for r in rows if r['prompt_length_tokens'] == target_len]
        model_rows = sorted(model_rows, key=lambda r: (_safe_label(r['model_name']), r.get('precision', 'auto')))
        labels = [f"{_safe_label(r['model_name'])} [{r.get('precision', 'auto')}]" for r in model_rows]
        values = [r['per_token_mean_ms'] for r in model_rows]
        plt.bar(labels, values)
        plt.xticks(rotation=25, ha='right')
        plt.ylabel('per token mean ms')
        plt.title(f'Model-size comparison at {target_len} prompt tokens')
        out_path = plots_dir / 'model_size_comparison_per_token.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print('Saved', out_path)

    precisions = sorted({r.get('precision', 'auto') for r in rows})
    if len(precisions) > 1:
        first_model = sorted({r['model_name'] for r in rows})[0]
        precision_rows = [r for r in rows if r['model_name'] == first_model]
        precision_rows = sorted(precision_rows, key=lambda r: (r.get('precision', 'auto'), r['prompt_length_tokens']))
        by_precision = defaultdict(list)
        for row in precision_rows:
            by_precision[row.get('precision', 'auto')].append(row)
        plt.figure(figsize=(8, 5))
        for precision, group in by_precision.items():
            group = sorted(group, key=lambda r: r['prompt_length_tokens'])
            plt.plot([g['prompt_length_tokens'] for g in group], [g['per_token_mean_ms'] for g in group], marker='o', label=precision)
        plt.xlabel('Prompt length (tokens)')
        plt.ylabel('per token mean ms')
        plt.title(f'Precision sweep for {_safe_label(first_model)}')
        plt.legend()
        out_path = plots_dir / 'precision_comparison_per_token.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print('Saved', out_path)

    summary_path = Path('docs/goal3_scaling_report.md')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_goal3_summary(data, summary_path)
    print('Saved', summary_path)

    inflection_notes = []
    by_model_precision = defaultdict(list)
    for row in rows:
        key = (row['model_name'], row.get('precision', 'auto'))
        by_model_precision[key].append(row)
    for (model_name, precision), grp in by_model_precision.items():
        notes = _detect_inflections(grp, 'per_token_mean_ms')
        for note in notes:
            inflection_notes.append(
                f"- {_safe_label(model_name)} [{precision}] possible inflection near {note['near_tokens']} tokens: slope rose from {note['previous_slope_ms_per_token']:.4f} to {note['current_slope_ms_per_token']:.4f} ms/token (x{note['ratio']:.2f})."
            )
    note_path = plots_dir / 'inflection_points.md'
    note_path.write_text('# Scaling Inflection Notes\n\n' + ('\n'.join(inflection_notes) if inflection_notes else 'No strong inflection points detected by the simple slope heuristic.'), encoding='utf-8')
    print('Saved', note_path)


if __name__ == '__main__':
    main()
