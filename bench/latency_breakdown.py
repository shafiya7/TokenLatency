from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from bench.utils import now_ms, sync_device


@dataclass(slots=True)
class StageSnapshot:
    totals_ms: dict[str, float]


class StageTimer:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.totals_ms = defaultdict(float)
        self._starts = {}
        self._handles = []

    def _bucket(self, name: str, module: torch.nn.Module | None = None) -> str | None:
        class_name = module.__class__.__name__.lower() if module is not None else ""

        if "embed_tokens" in name:
            return "embedding_lookup_ms"
        if name.endswith("q_proj") or name.endswith("k_proj") or name.endswith("v_proj"):
            return "attention_qkv_ms"
        if name.endswith("o_proj"):
            return "attention_projection_ms"
        if name.endswith("self_attn"):
            return "attention_total_ms"
        if "decoderlayer" in class_name:
            return "decoder_layer_total_ms"
        if name.endswith("gate_proj") or name.endswith("up_proj") or name.endswith("down_proj"):
            return "mlp_projection_ms"
        if name.endswith("mlp"):
            return "mlp_total_ms"
        if name.endswith("input_layernorm") or name.endswith("post_attention_layernorm") or name.endswith("norm"):
            return "layernorm_total_ms"
        if name.endswith("lm_head"):
            return "lm_head_ms"
        return None

    def _pre(self, module_name: str):
        def hook(module, _args):
            sync_device(self.device)
            self._starts[id(module)] = (module_name, now_ms())
        return hook

    def _post(self, module_name: str):
        def hook(module, _args, _output):
            sync_device(self.device)
            item = self._starts.pop(id(module), None)
            if item is None:
                return
            _, t0 = item
            bucket = self._bucket(module_name, module)
            if bucket is not None:
                self.totals_ms[bucket] += (now_ms() - t0)
        return hook

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if self._bucket(name, module) is None:
                continue
            self._handles.append(module.register_forward_pre_hook(self._pre(name)))
            self._handles.append(module.register_forward_hook(self._post(name)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def snapshot(self) -> StageSnapshot:
        return StageSnapshot(dict(self.totals_ms))

    @staticmethod
    def diff(after: StageSnapshot, before: StageSnapshot) -> dict[str, float]:
        keys = set(before.totals_ms) | set(after.totals_ms)
        return {k: after.totals_ms.get(k, 0.0) - before.totals_ms.get(k, 0.0) for k in keys}

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.detach()
        return False
