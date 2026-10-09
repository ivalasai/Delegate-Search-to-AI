#!/usr/bin/env python3
"""Extract problem/solution pairs from SBIR abstracts via phi3:mini (Ollama)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("extract_sbir")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "corpus_raw" / "sbir_sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "corpus_raw" / "sbir_extracted.json"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "extraction_prompt_v1.md"
OLLAMA_URL = "http://localhost:11434"
EXTRACTION_MODEL = "phi3:mini"
TEMPERATURE = 0.1
PROMPT_VERSION = "extraction_prompt_v1"

SYSTEM_PROMPT = (
    "You split SBIR award abstracts into two short research statements for a "
    "semantic-matching study. Be faithful to the source text; do not invent facts. "
    "Keep each statement concise (1–3 sentences). Return only valid JSON."
)

USER_TEMPLATE = """Split this SBIR award abstract into two statements.

Award ID: {award_id}
Company: {company}

Abstract:
{abstract}

Return ONLY a JSON object with exactly these two string fields:
{{"problem": "<the need, gap, or challenge being addressed>",
 "solution": "<the proposed technical approach or innovation>"}}

Do not include markdown, commentary, or extra keys."""


def load_sample(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def ollama_generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    base_url: str = OLLAMA_URL,
    temperature: float = TEMPERATURE,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama call failed for model '{model}' at {base_url}: {exc}"
        ) from exc
    if payload.get("error"):
        raise RuntimeError(f"Ollama error for model '{model}': {payload['error']}")
    return str(payload.get("response", ""))


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f]")


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unclosed JSON object")


def _load_json_object(text: str) -> dict[str, object]:
    cleaned = _CONTROL_CHAR_RE.sub(" ", text).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def parse_extraction_response(raw: str) -> dict[str, str]:
    text = raw.strip()
    block = _JSON_BLOCK_RE.search(text)
    if block:
        text = block.group(1).strip()
    errors: list[str] = []
    for candidate in (text, _first_json_object(text)):
        try:
            parsed = _load_json_object(candidate)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
    else:
        raise RuntimeError(
            f"Model did not return valid JSON. Raw response:\n{raw}"
        ) from None
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object, got {type(parsed).__name__}")
    problem = str(parsed.get("problem", "")).strip()
    solution = str(parsed.get("solution", "")).strip()
    if not problem or not solution:
        raise RuntimeError(f"Missing problem or solution in parsed JSON:\n{raw}")
    return {"problem": problem, "solution": solution}


def extract_row(
    row: dict[str, str],
    index: int,
    *,
    base_url: str,
    model: str,
) -> dict[str, object]:
    award_id = row["award_id"]
    user_prompt = USER_TEMPLATE.format(
        award_id=award_id,
        company=row["company"],
        abstract=row["abstract"],
    )
    raw_response = ollama_generate(
        model, SYSTEM_PROMPT, user_prompt, base_url=base_url
    )
    try:
        split = parse_extraction_response(raw_response)
    except RuntimeError:
        LOGGER.warning("Retrying extraction for %s after parse failure", award_id)
        raw_response = ollama_generate(
            model, SYSTEM_PROMPT, user_prompt, base_url=base_url
        )
        split = parse_extraction_response(raw_response)
    seq = f"{index:03d}"
    return {
        "pair_id": seq,
        "problem_id": f"P{seq}",
        "solution_id": f"S{seq}",
        "award_id": award_id,
        "agency": row["agency"],
        "year": row["year"],
        "company": row["company"],
        "abstract": row["abstract"],
        "problem_statement": split["problem"],
        "solution_statement": split["solution"],
        "raw_response": raw_response,
        "user_prompt": user_prompt,
    }


def extracted_path_for_corpus(corpus_id: str) -> Path:
    if corpus_id == "sbir_v1":
        return PROJECT_ROOT / "data" / "corpus_raw" / "sbir_extracted.json"
    return PROJECT_ROOT / "data" / "corpus_raw" / f"{corpus_id}_extracted.json"


def run_extraction(
    input_path: Path,
    output_path: Path,
    *,
    corpus_id: str,
    base_url: str,
    model: str,
) -> dict[str, object]:
    rows = load_sample(input_path)
    items: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        LOGGER.info("Extracting %d/%d: %s", index, len(rows), row["award_id"])
        items.append(
            extract_row(row, index, base_url=base_url, model=model)
        )

    payload = {
        "corpus_id": corpus_id,
        "source_file": str(input_path.relative_to(PROJECT_ROOT)),
        "prompt_version": PROMPT_VERSION,
        "prompt_file": str(PROMPT_PATH.relative_to(PROJECT_ROOT)),
        "model": model,
        "inference_mode": "local_ollama",
        "ollama_base_url": base_url,
        "temperature": TEMPERATURE,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "n_items": len(items),
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def print_random_pairs(payload: dict[str, object], n: int, seed: int) -> None:
    import random

    items = list(payload["items"])
    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))
    print(f"\n{'=' * 72}")
    print(f"RANDOM SAMPLE: {len(sample)} problem/solution pairs (seed={seed})")
    print(f"{'=' * 72}\n")
    for item in sample:
        print(f"[{item['pair_id']}] {item['award_id']} | {item['company']}")
        print(f"  PROBLEM:  {item['problem_statement']}")
        print(f"  SOLUTION: {item['solution_statement']}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract SBIR problem/solution pairs.")
    parser.add_argument("--corpus-id", default="sbir_v1")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--model", default=EXTRACTION_MODEL)
    parser.add_argument("--show-random", type=int, default=15)
    parser.add_argument("--show-seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    input_path = args.input or (
        DEFAULT_INPUT if args.corpus_id == "sbir_v1"
        else PROJECT_ROOT / "data" / "corpus_raw" / f"{args.corpus_id}_sample.csv"
    )
    output_path = args.output or extracted_path_for_corpus(args.corpus_id)
    if not input_path.is_file():
        raise SystemExit(f"ERROR: input not found: {input_path}")
    if not PROMPT_PATH.is_file():
        raise SystemExit(f"ERROR: prompt file not found: {PROMPT_PATH}")

    payload = run_extraction(
        input_path,
        output_path,
        corpus_id=args.corpus_id,
        base_url=args.ollama_url,
        model=args.model,
    )
    print(f"Wrote {payload['n_items']} extractions to {output_path}", file=sys.stderr)
    print_random_pairs(payload, args.show_random, args.show_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
