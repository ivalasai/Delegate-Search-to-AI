#!/usr/bin/env python3
"""Run the Stage-1 L2/L3 model comparison grid with checkpointing and resume.

Default A/B grid: 2 models x 2 architectures x 3 seeds = 12 runs.
C grid (--models cd): nemotron-3-nano:4b, 6 runs.
Model E (--models e): Gemma2 9B, all seeds 11/22/33, ranked + model selection (6 jobs).
The combined D/E seed-11 mode (--models de) remains available for two jobs per model.
Model D (--models d): Qwen2.5 7B, all seeds 11/22/33, ranked + model selection (6 jobs).
Each run checkpoints after every tick via run_layer2_ai.run_condition(resume=True).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from run_layer2_ai import run_condition

LOGGER = logging.getLogger("run_layer2_grid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_STATE_PATH = PROJECT_ROOT / "data" / "simulation_results" / "grid_state.json"

DEFAULT_JOBS = [
  {"model_config": "configs/models/mistral_7b_instruct.json", "condition": "ranked", "seed": 11},
  {"model_config": "configs/models/mistral_7b_instruct.json", "condition": "ranked", "seed": 22},
  {"model_config": "configs/models/mistral_7b_instruct.json", "condition": "ranked", "seed": 33},
  {"model_config": "configs/models/mistral_7b_instruct.json", "condition": "autonomous", "seed": 11},
  {"model_config": "configs/models/mistral_7b_instruct.json", "condition": "autonomous", "seed": 22},
  {"model_config": "configs/models/mistral_7b_instruct.json", "condition": "autonomous", "seed": 33},
  {"model_config": "configs/models/llama3_1_8b.json", "condition": "ranked", "seed": 11},
  {"model_config": "configs/models/llama3_1_8b.json", "condition": "ranked", "seed": 22},
  {"model_config": "configs/models/llama3_1_8b.json", "condition": "ranked", "seed": 33},
  {"model_config": "configs/models/llama3_1_8b.json", "condition": "autonomous", "seed": 11},
  {"model_config": "configs/models/llama3_1_8b.json", "condition": "autonomous", "seed": 22},
  {"model_config": "configs/models/llama3_1_8b.json", "condition": "autonomous", "seed": 33},
]

# Model C only. The stored autonomous condition label means L3 model selection.
CD_JOBS = [
  {"model_config": "configs/models/nemotron_3_nano_4b.json", "condition": "ranked", "seed": 11},
  {"model_config": "configs/models/nemotron_3_nano_4b.json", "condition": "ranked", "seed": 22},
  {"model_config": "configs/models/nemotron_3_nano_4b.json", "condition": "ranked", "seed": 33},
  {"model_config": "configs/models/nemotron_3_nano_4b.json", "condition": "autonomous", "seed": 11},
  {"model_config": "configs/models/nemotron_3_nano_4b.json", "condition": "autonomous", "seed": 22},
  {"model_config": "configs/models/nemotron_3_nano_4b.json", "condition": "autonomous", "seed": 33},
]

# Model D: all seeds (L2 ranked + L3 model selection).
D_JOBS = [
  {"model_config": "configs/models/qwen25_7b.json", "condition": "ranked", "seed": 11},
  {"model_config": "configs/models/qwen25_7b.json", "condition": "ranked", "seed": 22},
  {"model_config": "configs/models/qwen25_7b.json", "condition": "ranked", "seed": 33},
  {"model_config": "configs/models/qwen25_7b.json", "condition": "autonomous", "seed": 11},
  {"model_config": "configs/models/qwen25_7b.json", "condition": "autonomous", "seed": 22},
  {"model_config": "configs/models/qwen25_7b.json", "condition": "autonomous", "seed": 33},
]
D_SEED11_JOBS = [
  {"model_config": "configs/models/qwen25_7b.json", "condition": "ranked", "seed": 11},
  {"model_config": "configs/models/qwen25_7b.json", "condition": "autonomous", "seed": 11},
]
E_SEED11_JOBS = [
  {"model_config": "configs/models/gemma2_9b.json", "condition": "ranked", "seed": 11},
  {"model_config": "configs/models/gemma2_9b.json", "condition": "autonomous", "seed": 11},
]
E_JOBS = [
  {"model_config": "configs/models/gemma2_9b.json", "condition": "ranked", "seed": 11},
  {"model_config": "configs/models/gemma2_9b.json", "condition": "ranked", "seed": 22},
  {"model_config": "configs/models/gemma2_9b.json", "condition": "ranked", "seed": 33},
  {"model_config": "configs/models/gemma2_9b.json", "condition": "autonomous", "seed": 11},
  {"model_config": "configs/models/gemma2_9b.json", "condition": "autonomous", "seed": 22},
  {"model_config": "configs/models/gemma2_9b.json", "condition": "autonomous", "seed": 33},
]


def job_key(job: dict) -> str:
    return f"{Path(job['model_config']).stem}_{job['condition']}_seed{job['seed']}"


def load_grid_state() -> dict:
    if GRID_STATE_PATH.is_file():
        return json.loads(GRID_STATE_PATH.read_text(encoding="utf-8"))
    return {"jobs": {}, "updated_at": None}


def save_grid_state(state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    GRID_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRID_STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def run_grid(
    *,
    corpus_id: str,
    embedding_model: str,
    n_ticks: int,
    repeats: int,
    prompt_version: str,
    jobs: list[dict],
    resume: bool,
    max_jobs: int | None = None,
) -> None:
    state = load_grid_state()
    started = 0
    for index, job in enumerate(jobs, start=1):
        key = job_key(job)
        if resume and (state.get("jobs") or {}).get(key, {}).get("status") == "complete":
            LOGGER.info("Skipping complete job %d/%d: %s", index, len(jobs), key)
            continue
        if max_jobs is not None and started >= max_jobs:
            LOGGER.info("Stopping after %s job(s); next would be %s", max_jobs, key)
            break
        LOGGER.info("Grid job %d/%d: %s", index, len(jobs), key)
        started += 1
        state["jobs"].setdefault(key, {})
        state["jobs"][key]["status"] = "running"
        save_grid_state(state)

        result_path, audit_path = run_condition(
            condition=job["condition"],
            seed=job["seed"],
            corpus_id=corpus_id,
            embedding_model_key=embedding_model,
            n_ticks=n_ticks,
            n_repeats=repeats,
            model_config_path=job["model_config"],
            prompt_version=prompt_version,
            resume=resume,
        )
        state["jobs"][key] = {
            "status": "complete",
            "result_path": str(result_path.relative_to(PROJECT_ROOT)),
            "audit_log": str(audit_path.relative_to(PROJECT_ROOT)),
        }
        save_grid_state(state)
        LOGGER.info("Completed %s -> %s", key, result_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected L2/L3 model grid.")
    parser.add_argument(
        "--corpus-id",
        required=True,
        help="Required. Stage-1 frozen corpus is sbir_v2; no default (avoids silent sbir_v1 fallback).",
    )
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument(
        "--n-ticks",
        type=int,
        required=True,
        help="Required. Stage-1 frozen horizon is 17; no default (avoids silent 100-tick fallback).",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-version", default="coupling_prompts_v1")
    parser.add_argument(
        "--models",
        choices=["ab", "cd", "d", "e", "de", "all"],
        default="ab",
        help=(
            "ab = A/B (mistral, llama3.1); cd = C (nano 4b, all seeds); "
            "d = Qwen2.5 7B all seeds; e = Gemma2 9B all seeds; "
            "de = D/E seed-11 jobs only. `all` is A/B+C and does not include D/E. "
            "The stored autonomous label denotes L3 model selection."
        ),
    )
    parser.add_argument(
        "--only-tier",
        choices=[
            "tier_1_lightly_aligned",
            "tier_2_heavily_aligned",
            "tier_3_nemotron_nano_4b",
            "tier_4_qwen25_7b",
            "tier_5_gemma2_9b",
        ],
        help="Optional: run one legacy model-config tier's six jobs only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each run from its per-tick checkpoint when present.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Run at most this many incomplete jobs, then exit (skip already complete when --resume).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.models == "cd":
        jobs = list(CD_JOBS)
    elif args.models == "d":
        jobs = list(D_JOBS)
    elif args.models == "e":
        jobs = list(E_JOBS)
    elif args.models == "de":
        jobs = list(D_SEED11_JOBS) + list(E_SEED11_JOBS)
    elif args.models == "all":
        jobs = list(DEFAULT_JOBS) + list(CD_JOBS)
    else:
        jobs = list(DEFAULT_JOBS)
    if args.only_tier == "tier_1_lightly_aligned":
        jobs = [j for j in jobs if "mistral" in j["model_config"]]
    elif args.only_tier == "tier_2_heavily_aligned":
        jobs = [j for j in jobs if "llama3" in j["model_config"]]
    elif args.only_tier == "tier_3_nemotron_nano_4b":
        jobs = [j for j in jobs if "nemotron_3_nano" in j["model_config"]]
    elif args.only_tier == "tier_4_qwen25_7b":
        jobs = [j for j in jobs if "qwen25_7b" in j["model_config"]]
    elif args.only_tier == "tier_5_gemma2_9b":
        jobs = [j for j in jobs if "gemma2_9b" in j["model_config"]]

    run_grid(
        corpus_id=args.corpus_id,
        embedding_model=args.embedding_model,
        n_ticks=args.n_ticks,
        repeats=args.repeats,
        prompt_version=args.prompt_version,
        jobs=jobs,
        resume=args.resume,
        max_jobs=args.max_jobs,
    )


if __name__ == "__main__":
    main()
