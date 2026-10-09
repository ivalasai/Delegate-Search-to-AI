#!/usr/bin/env python3
"""Single abstraction for language-model calls.

`call_model(config, system_prompt, user_prompt, call_seed)` is the only entry
point the simulation uses. Whether inference runs against local open weights
(transformers) or an OpenAI-compatible API is decided entirely by the model
config file (`inference_mode`), never by simulation code.

All AI behavior in the pipeline must come through this function; no formula in
the codebase may decide which items a model "prefers".
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("model_backend")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelResponse:
    text: str
    rendering: str  # "chat" | "completion"
    model_id: str
    version_pinned: str


def load_model_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))

    schema_path = PROJECT_ROOT / "src" / "schema" / "model_config.schema.json"
    try:
        import jsonschema

        jsonschema.validate(config, json.loads(schema_path.read_text(encoding="utf-8")))
    except ImportError:
        LOGGER.warning("jsonschema not installed; skipping model config validation")
    return config


def _call_ollama(
    config: dict[str, Any], system_prompt: str, user_prompt: str, call_seed: int
) -> ModelResponse:
    base_url = config.get("ollama_base_url", "http://localhost:11434")
    payload: dict[str, Any] = {
        "model": config["model_id"],
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "seed": call_seed,
        "options": {
            "temperature": config["temperature"],
            "num_predict": config.get("max_new_tokens", 256),
            **(
                {"num_ctx": int(config["num_ctx"])}
                if config.get("num_ctx") is not None
                else {}
            ),
        },
    }
    if "think" in config:
        payload["think"] = bool(config["think"])
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        sys.exit(
            f"ERROR: failed to call Ollama at {base_url} for model "
            f"{config['model_id']}: {exc}"
        )
    if payload.get("error"):
        sys.exit(
            f"ERROR: Ollama returned an error for model {config['model_id']}: "
            f"{payload['error']}"
        )
    text = payload.get("response") or ""
    if not str(text).strip() and payload.get("thinking"):
        text = str(payload["thinking"])
    return ModelResponse(
        text=text,
        rendering="chat",
        model_id=config["model_id"],
        version_pinned=config["version_pinned"],
    )


def call_model(
    config: dict[str, Any], system_prompt: str, user_prompt: str, call_seed: int
) -> ModelResponse:
    """Route a single model call according to the config's inference_mode."""
    if config["inference_mode"] == "local_ollama":
        return _call_ollama(config, system_prompt, user_prompt, call_seed)
    sys.exit(f"ERROR: unknown inference_mode: {config['inference_mode']}")
