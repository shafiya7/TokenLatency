from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[0]
# When you copy this file into scripts/, it should resolve to repo root via parents[1].
# This local copy is just for download convenience.
if (PROJECT_ROOT / 'bench').exists():
    ROOT = PROJECT_ROOT
else:
    ROOT = PROJECT_ROOT.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.config import BenchmarkConfig  # type: ignore
from bench.harness import HFLatencyHarness  # type: ignore
from bench.utils import ensure_dir, json_dump, timestamp_slug  # type: ignore


def run_once(cfg: BenchmarkConfig) -> dict[str, Any]:
    harness = HFLatencyHarness(cfg)
    try:
        return harness.run()
    finally:
        harness.close()


def _summary_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    filtered = result.get('summary', {}).get('filtered', {})
    return {
        'ttft_mean_ms': filtered.get('ttft_ms', {}).get('mean'),
        'per_token_mean_ms': filtered.get('avg_per_token_ms', {}).get('mean'),
        'e2e_mean_ms': filtered.get('e2e_ms', {}).get('mean'),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run four combinations: fp32/fp16 x KV-cache on/off.'
    )
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument(
        '--prompt',
        default='Explain how precision and KV cache affect token generation latency in a decoder-only LLaMA model.',
    )
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--measured-trials', type=int, default=3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--system-name', default='local_machine')
    parser.add_argument('--estimated-bandwidth-gbps', type=float, default=None)
    parser.add_argument(
        '--precisions',
        nargs='*',
        default=['float32', 'float16'],
        help='Usually float32 float16. On CPU, float16 may be unsupported or slower.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []

    combos = []
    for precision in args.precisions:
        for use_kv_cache in (True, False):
            combos.append((precision, use_kv_cache))

    for precision, use_kv_cache in combos:
        cfg = BenchmarkConfig(
            model_name=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            warmup_runs=args.warmup_runs,
            measured_trials=args.measured_trials,
            device_preference=args.device,
            torch_dtype=precision,
            save_dir=args.save_dir,
            system_name=args.system_name,
            estimated_bandwidth_gbps=args.estimated_bandwidth_gbps,
            use_kv_cache=use_kv_cache,
        )
        result = run_once(cfg)
        metrics = _summary_metrics(result)
        row = {
            'model_name': args.model,
            'precision': precision,
            'use_kv_cache': use_kv_cache,
            'kv_cache_label': 'with_kv_cache' if use_kv_cache else 'without_kv_cache',
            'ttft_mean_ms': metrics['ttft_mean_ms'],
            'per_token_mean_ms': metrics['per_token_mean_ms'],
            'e2e_mean_ms': metrics['e2e_mean_ms'],
            'runtime_device': result.get('runtime', {}).get('device'),
            'runtime_dtype': result.get('runtime', {}).get('dtype'),
            'result_file': result.get('saved_to'),
        }
        experiment_rows.append(row)
        raw_results.append({
            'config': asdict(cfg),
            'result': result,
        })

    # pairwise speedups vs no-cache for same precision
    pairwise: list[dict[str, Any]] = []
    for precision in args.precisions:
        on = next((r for r in experiment_rows if r['precision'] == precision and r['use_kv_cache']), None)
        off = next((r for r in experiment_rows if r['precision'] == precision and not r['use_kv_cache']), None)
        if on and off:
            on_pt = on['per_token_mean_ms']
            off_pt = off['per_token_mean_ms']
            pairwise.append({
                'precision': precision,
                'per_token_speedup_with_kv': (off_pt / on_pt) if on_pt and off_pt else None,
                'per_token_delta_ms': (off_pt - on_pt) if on_pt is not None and off_pt is not None else None,
                'ttft_delta_ms': (off['ttft_mean_ms'] - on['ttft_mean_ms']) if on['ttft_mean_ms'] is not None and off['ttft_mean_ms'] is not None else None,
                'e2e_delta_ms': (off['e2e_mean_ms'] - on['e2e_mean_ms']) if on['e2e_mean_ms'] is not None and off['e2e_mean_ms'] is not None else None,
            })

    payload = {
        'experiment': 'precision_kv_matrix',
        'rows': experiment_rows,
        'pairwise_comparisons': pairwise,
        'raw_results': raw_results,
    }

    out_dir = ensure_dir(args.save_dir)
    out_path = out_dir / f'precision_kv_matrix_{timestamp_slug()}.json'
    json_dump(payload, out_path)

    print(json.dumps({
        'saved_to': str(out_path),
        'rows': experiment_rows,
        'pairwise_comparisons': pairwise,
    }, indent=2))


if __name__ == '__main__':
    main()
