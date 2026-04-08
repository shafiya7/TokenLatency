from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MODELS = {
    "models": [
        {"model_name": "meta-llama/Llama-3.2-1B-Instruct", "family": "meta-llama", "size": "1B"},
        {"model_name": "meta-llama/Llama-3.2-3B-Instruct", "family": "meta-llama", "size": "3B"},
        {"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "tinyllama", "size": "1.1B"},
    ]
}


def load_model_registry(path: str | Path = "configs/models.json") -> dict:
    model_path = Path(path)
    if not model_path.exists():
        return DEFAULT_MODELS
    return json.loads(model_path.read_text(encoding="utf-8"))


def list_models(path: str | Path = "configs/models.json") -> list[str]:
    data = load_model_registry(path)
    return [row["model_name"] for row in data["models"]]
