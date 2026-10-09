#!/usr/bin/env python3
"""Embed extracted SBIR problem/solution statements with a local Ollama model.

Reads `data/corpus_raw/sbir_extracted.json` and embeds every problem statement
and solution statement via Ollama's local embeddings API. Outputs a single
cache file at `data/embeddings/sbir_v1_{embedding_model_name}.json`, keyed by
item id and retaining the source text for traceability.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("embed_corpus")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT / "data" / "corpus_raw"
EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "embeddings"
EMBEDDING_MODELS_CONFIG = PROJECT_ROOT / "configs" / "embedding_models.json"
def extracted_path_for_corpus(corpus_id: str) -> Path:
    if corpus_id == "sbir_v1":
        return CORPUS_ROOT / "sbir_extracted.json"
    return CORPUS_ROOT / f"{corpus_id}_extracted.json"


def load_extracted_items(corpus_id: str) -> dict[str, dict[str, str]]:
    """Load extracted SBIR problem/solution statements. Fails loudly if missing."""
    extracted_path = extracted_path_for_corpus(corpus_id)
    if not extracted_path.is_file():
        sys.exit(
            f"ERROR: extracted corpus not found: {extracted_path}\n"
            f"Run:\n"
            f"  python src/extract_sbir_corpus.py --corpus-id {corpus_id}\n"
            "There is no synthetic fallback."
        )

    payload = json.loads(extracted_path.read_text(encoding="utf-8"))
    items: dict[str, dict[str, str]] = {}
    for record in payload.get("items", []):
        for role, key in (("problem", "problem_statement"), ("solution", "solution_statement")):
            item_id = record[f"{role}_id"]
            text = str(record.get(key, "")).strip()
            if not text:
                sys.exit(f"ERROR: empty {role} statement for item {item_id}")
            items[item_id] = {
                "role": role,
                "text": text,
                "pair_id": record["pair_id"],
                "award_id": record["award_id"],
                "agency": record["agency"],
                "year": record["year"],
                "company": record["company"],
            }

    n_problems = sum(1 for item in items.values() if item["role"] == "problem")
    n_solutions = len(items) - n_problems
    if n_problems == 0 or n_solutions == 0:
        sys.exit(
            f"ERROR: extracted corpus '{corpus_id}' needs at least one problem and "
            f"one solution item (found {n_problems} problems, {n_solutions} solutions). "
            "There is no synthetic fallback."
        )
    LOGGER.info(
        "Loaded extracted corpus '%s': %d problems, %d solutions",
        corpus_id, n_problems, n_solutions,
    )
    return items


def load_embedding_model_spec(key: str) -> dict[str, str]:
    registry = json.loads(EMBEDDING_MODELS_CONFIG.read_text(encoding="utf-8"))
    if key not in registry:
        sys.exit(
            f"ERROR: unknown embedding model key '{key}'. "
            f"Available: {', '.join(sorted(registry))} "
            f"(see {EMBEDDING_MODELS_CONFIG})"
        )
    return registry[key]


def cache_path(corpus_id: str, model_name: str) -> Path:
    safe_name = model_name.replace(":", "_")
    return EMBEDDINGS_ROOT / f"{corpus_id}_{safe_name}.json"


def ollama_embed(text: str, model_name: str, base_url: str) -> list[float]:
    body = json.dumps({"model": model_name, "prompt": text}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        sys.exit(
            f"ERROR: failed to call Ollama embeddings API at {base_url} for "
            f"model '{model_name}': {exc}"
        )
    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        sys.exit(
            f"ERROR: Ollama embedding response for model '{model_name}' did not "
            "include a non-empty 'embedding' array."
        )
    try:
        return [float(x) for x in vector]
    except (TypeError, ValueError) as exc:
        sys.exit(f"ERROR: embedding vector contains non-numeric values: {exc}")


def embed_corpus(corpus_id: str, model_key: str, force: bool = False) -> Path:
    spec = load_embedding_model_spec(model_key)
    if spec.get("inference_mode") != "local_ollama":
        sys.exit(
            f"ERROR: embedding model '{model_key}' must use inference_mode="
            f"'local_ollama' for this Stage-1 pipeline."
        )
    model_name = spec["model_id"]
    base_url = spec.get("ollama_base_url", "http://localhost:11434")
    out_path = cache_path(corpus_id, model_name)
    if out_path.exists() and not force:
        LOGGER.info("Cache hit, skipping: %s (use --force to re-embed)", out_path)
        return out_path

    items = load_extracted_items(corpus_id)
    extracted_path = extracted_path_for_corpus(corpus_id)

    LOGGER.info(
        "Embedding with Ollama model %s @ %s",
        model_name, spec["version_pinned"],
    )

    item_ids = sorted(items)
    vectors = [ollama_embed(items[item_id]["text"], model_name, base_url) for item_id in item_ids]
    embedding_dim = len(vectors[0])
    if any(len(vec) != embedding_dim for vec in vectors):
        sys.exit("ERROR: inconsistent embedding dimensions returned by Ollama.")

    payload = {
        "corpus_id": corpus_id,
        "embedding_model": {
            "key": model_key,
            "model_id": model_name,
            "version_pinned": spec["version_pinned"],
            "inference_mode": spec["inference_mode"],
            "ollama_base_url": base_url,
        },
        "source_file": str(extracted_path.relative_to(PROJECT_ROOT)),
        "embedding_dim": embedding_dim,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": {
            item_id: {
                "role": items[item_id]["role"],
                "text": items[item_id]["text"],
                "pair_id": items[item_id]["pair_id"],
                "award_id": items[item_id]["award_id"],
                "agency": items[item_id]["agency"],
                "year": items[item_id]["year"],
                "company": items[item_id]["company"],
                "vector": vec,
            }
            for item_id, vec in zip(item_ids, vectors)
        },
    }

    EMBEDDINGS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Wrote %d embeddings (dim=%d) to %s",
        len(item_ids), payload["embedding_dim"], out_path,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a raw corpus with a pinned model.")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Key in configs/embedding_models.json (default: minilm)",
    )
    parser.add_argument("--force", action="store_true", help="Re-embed even if cached")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    embed_corpus(args.corpus_id, args.embedding_model, force=args.force)


if __name__ == "__main__":
    main()
