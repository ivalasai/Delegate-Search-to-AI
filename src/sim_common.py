#!/usr/bin/env python3
"""Shared infrastructure for RuleProxy and model-mediated simulation runs.

Holds everything both layers must share so their outputs are directly
comparable: corpus/embedding loading, seeded arrival schedules, the RuleProxy
satisficing evaluator (L1 rules), and run-manifest/result writing.

Behavioral constants (RuleProxy rules, applied verbatim in L1 and as the
acceptance stage of L2 ranked):
  - local search breadth n=2 candidate evaluations per problem per tick
  - energy decay of 0.15 per evaluation (per-problem budget starting at 1.0)
  - optional perceptual noise when energy < 0.40 (Gaussian, sd 0.15); disabled
    by default in Stage 1 (USE_PERCEPTUAL_NOISE = False)
  - satisficing threshold: accept the first candidate whose perceived cosine
    similarity >= 0.60 (with noise off, perceived equals true similarity)
"""

from __future__ import annotations

import json
import logging
import math
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("sim_common")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "embeddings"
RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = PROJECT_ROOT / "data" / "simulation_results"

LOCAL_SEARCH_N = 2
ENERGY_DECAY_PER_EVAL = 0.15
FIDELITY_DEGRADATION_BELOW = 0.40
FIDELITY_NOISE_SD = 0.15
SATISFICING_THRESHOLD = 0.60
INITIAL_ENERGY = 1.0
# Stage-1 default: acceptance uses true cosine similarity. Set True to add
# Gaussian noise to perceived similarity when energy is low (sensitivity analysis).
USE_PERCEPTUAL_NOISE = False

ARRIVAL_WINDOW_FRACTION = 0.4  # items arrive within the first 40% of ticks


def load_embeddings(corpus_id: str, embedding_model_key: str) -> dict[str, Any]:
    """Load a cached embedding file. Fails loudly if missing — no synthetic fallback."""
    matches = sorted(EMBEDDINGS_ROOT.glob(f"{corpus_id}_{embedding_model_key}_*.json"))
    if not matches:
        matches = sorted(EMBEDDINGS_ROOT.glob(f"{corpus_id}_{embedding_model_key}.json"))
    if not matches:
        sys.exit(
            f"ERROR: no cached embeddings for corpus '{corpus_id}' with model "
            f"'{embedding_model_key}'. Run:\n"
            f"  python src/embed_corpus.py --corpus-id {corpus_id} "
            f"--embedding-model {embedding_model_key}\n"
            "and make sure data/corpus_raw/ is populated. "
            "There is no synthetic-vector fallback."
        )
    if len(matches) > 1:
        LOGGER.warning("Multiple embedding caches match; using %s", matches[-1])
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def trace_variance(vectors: list[list[float]]) -> float:
    """Mean per-dimension variance; 0.0 when fewer than 2 vectors are present."""
    if len(vectors) < 2:
        return 0.0
    dims = len(vectors[0])
    means = [0.0] * dims
    for vec in vectors:
        if len(vec) != dims:
            sys.exit("ERROR: inconsistent vector lengths while computing variance.")
        for idx, value in enumerate(vec):
            means[idx] += value
    means = [value / len(vectors) for value in means]

    variances = [0.0] * dims
    for vec in vectors:
        for idx, value in enumerate(vec):
            diff = value - means[idx]
            variances[idx] += diff * diff
    variances = [value / len(vectors) for value in variances]
    return sum(variances) / dims


def similarity_distribution_diagnostic(items: dict[str, Any]) -> dict[str, Any]:
    """Log corpus similarity stats for audit; does not set the acceptance threshold."""
    problem_ids = sorted(i for i, item in items.items() if item["role"] == "problem")
    solution_ids = sorted(i for i, item in items.items() if item["role"] == "solution")
    similarities = [
        cosine_similarity(items[problem_id]["vector"], items[solution_id]["vector"])
        for problem_id in problem_ids
        for solution_id in solution_ids
    ]
    ordered = sorted(similarities)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = int((n - 1) * (p / 100.0))
        return round(ordered[idx], 6)

    same_index = [
        cosine_similarity(items[p]["vector"], items[s]["vector"])
        for p in problem_ids
        for s in solution_ids
        if p[1:] == s[1:]
    ]
    return {
        "n_problem_solution_pairs": n,
        "min_similarity": round(min(similarities), 6),
        "max_similarity": round(max(similarities), 6),
        "mean_similarity": round(sum(similarities) / n, 6),
        "p85_similarity": pct(85.0),
        "p95_similarity": pct(95.0),
        "fraction_at_or_above_0_60": round(
            sum(1 for value in similarities if value >= SATISFICING_THRESHOLD) / n, 6
        ),
        "same_index_pair_mean_similarity": round(
            sum(same_index) / len(same_index), 6
        ) if same_index else None,
    }


def decouple_solution_pool(
    items: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """Shuffle solution texts/embeddings across display IDs to break origin-index shortcuts.

    Problems keep their original IDs. Each display solution ID (S001, S002, ...)
    is assigned content from a randomly permuted origin solution, so P007 no
    longer structurally aligns with the solution extracted from the same abstract.
    """
    problem_ids = sorted(i for i, item in items.items() if item["role"] == "problem")
    solution_ids = sorted(i for i, item in items.items() if item["role"] == "solution")
    if not problem_ids or not solution_ids:
        sys.exit("ERROR: cannot decouple an empty problem or solution pool.")

    rng = random.Random(seed)
    origin_ids = solution_ids.copy()
    rng.shuffle(origin_ids)
    mapping = {
        display_id: origin_id
        for display_id, origin_id in zip(solution_ids, origin_ids)
    }

    decoupled: dict[str, Any] = {}
    for problem_id in problem_ids:
        decoupled[problem_id] = dict(items[problem_id])
    for display_id, origin_id in mapping.items():
        origin = items[origin_id]
        decoupled[display_id] = {
            **origin,
            "role": "solution",
            "display_id": display_id,
            "origin_solution_id": origin_id,
            "origin_pair_id": origin.get("pair_id"),
        }
    return decoupled, mapping


def origin_solution_id_for_problem(problem_id: str) -> str:
    return f"S{problem_id[1:]}"


def build_origin_pair_exclusions(
    solution_decoupling: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Map each problem to the display solution ID carrying its co-extracted origin text."""
    origin_to_display = {origin: display for display, origin in solution_decoupling.items()}
    exclusions: list[dict[str, str]] = []
    blocked_map: dict[str, str] = {}
    for origin_solution_id in sorted(origin_to_display):
        problem_id = f"P{origin_solution_id[1:]}"
        blocked_display_id = origin_to_display[origin_solution_id]
        exclusions.append(
            {
                "problem_id": problem_id,
                "origin_solution_id": origin_solution_id,
                "blocked_display_solution_id": blocked_display_id,
            }
        )
        blocked_map[problem_id] = blocked_display_id
    return exclusions, blocked_map


def is_same_origin_coupling(
    problem_id: str, solution_id: str, solution_decoupling: dict[str, str]
) -> bool:
    """True when the coupled solution text came from the same abstract as the problem."""
    return solution_decoupling.get(solution_id) == origin_solution_id_for_problem(problem_id)


def is_blocked_origin_match(problem_id: str, solution_id: str, blocked_map: dict[str, str]) -> bool:
    return blocked_map.get(problem_id) == solution_id


def eligible_solutions_for_problem(
    problem_id: str,
    solution_ids: list[str],
    blocked_map: dict[str, str],
    coupled_solutions: set[str] | None = None,
) -> list[str]:
    coupled_solutions = coupled_solutions or set()
    blocked = blocked_map.get(problem_id)
    return [
        sid
        for sid in solution_ids
        if sid not in coupled_solutions and sid != blocked
    ]


def solutions_visible_for_tick(
    active_problems: list[str],
    active_solutions: list[str],
    blocked_map: dict[str, str],
) -> list[str]:
    """Remove origin-matched display solutions for active problems from the tick pool."""
    blocked_display_ids = {blocked_map[p] for p in active_problems if p in blocked_map}
    return [sid for sid in active_solutions if sid not in blocked_display_ids]


def annotate_coupling(
    coupling: dict[str, Any], solution_decoupling: dict[str, str]
) -> dict[str, Any]:
    enriched = dict(coupling)
    enriched["same_origin_reunion"] = is_same_origin_coupling(
        coupling["problem_id"], coupling["solution_id"], solution_decoupling
    )
    return enriched


def summarize_acceptance_counts(couplings: list[dict[str, Any]]) -> dict[str, int]:
    """Split accepted couplings by whether true similarity cleared the threshold."""
    threshold_cleared = sum(
        1 for c in couplings if c.get("acceptance_type") == "threshold_cleared"
    )
    fidelity_noise = sum(
        1 for c in couplings if c.get("acceptance_type") == "fidelity_noise"
    )
    return {
        "n_couplings_total": len(couplings),
        "n_threshold_cleared": threshold_cleared,
        "n_fidelity_noise": fidelity_noise,
    }


def compute_same_origin_reunion_rate(
    couplings: list[dict[str, Any]], solution_decoupling: dict[str, str]
) -> float:
    if not couplings:
        return 0.0
    reunions = sum(
        1
        for coupling in couplings
        if is_same_origin_coupling(
            coupling["problem_id"], coupling["solution_id"], solution_decoupling
        )
    )
    return round(reunions / len(couplings), 6)


def build_arrival_schedule(
    item_ids: list[str], n_ticks: int, rng: random.Random
) -> dict[str, int]:
    """Assign each item a seeded uniform arrival tick early in the horizon."""
    window = max(1, int(n_ticks * ARRIVAL_WINDOW_FRACTION))
    return {item_id: rng.randint(0, window) for item_id in sorted(item_ids)}


class RuleProxy:
    """RuleProxy evaluator implementing the bounded-search rules.

    Maintains a per-problem energy budget. Each candidate evaluation costs
    energy and can_evaluate() gates how many candidates a problem sees.
    When use_perceptual_noise is True and energy is below FIDELITY_DEGRADATION_BELOW,
    perceived similarity becomes noisy; otherwise perceived equals true similarity.
    Accepts the first candidate whose perceived similarity clears the threshold.

    The default satisficing threshold is fixed at SATISFICING_THRESHOLD (0.60),
    identical across L1 and L2 ranked runs. Override explicitly via
    --satisficing-threshold; the value used is recorded in each run's output.
    """

    def __init__(
        self,
        rng: random.Random,
        threshold: float = SATISFICING_THRESHOLD,
        use_perceptual_noise: bool = USE_PERCEPTUAL_NOISE,
    ) -> None:
        self.rng = rng
        self.threshold = threshold
        self.use_perceptual_noise = use_perceptual_noise
        self.energy: dict[str, float] = {}

    def can_evaluate(self, problem_id: str) -> bool:
        return self.energy.get(problem_id, INITIAL_ENERGY) >= ENERGY_DECAY_PER_EVAL

    def evaluate(self, problem_id: str, true_similarity: float) -> dict[str, Any]:
        energy_before = self.energy.get(problem_id, INITIAL_ENERGY)
        self.energy[problem_id] = energy_before - ENERGY_DECAY_PER_EVAL

        degraded = (
            self.use_perceptual_noise
            and energy_before < FIDELITY_DEGRADATION_BELOW
        )
        perceived = true_similarity
        if degraded:
            perceived += self.rng.gauss(0.0, FIDELITY_NOISE_SD)

        accepted = perceived >= self.threshold
        if accepted:
            acceptance_type = (
                "threshold_cleared"
                if true_similarity >= self.threshold
                else "fidelity_noise"
            )
        else:
            acceptance_type = None

        return {
            "true_similarity": round(true_similarity, 6),
            "perceived_similarity": round(perceived, 6),
            "energy_before": round(energy_before, 6),
            "fidelity_degraded": degraded,
            "accepted": accepted,
            "acceptance_type": acceptance_type,
        }


# Backward-compatible import name for legacy result-generation code.
HumanProxy = RuleProxy


def make_run_manifest(
    *,
    seed: int,
    corpus_id: str,
    embeddings: dict[str, Any],
    condition: str,
    n_ticks: int,
    model_config_ref: str | None = None,
    model_config_snapshot: dict[str, Any] | None = None,
    prompt_version: str | None = None,
    n_repeats_per_tick: int | None = None,
) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = None

    return {
        "run_id": f"run_{uuid.uuid4().hex[:12]}",
        "seed": seed,
        "corpus_id": corpus_id,
        "embedding_model": {
            "model_id": embeddings["embedding_model"]["model_id"],
            "version_pinned": embeddings["embedding_model"]["version_pinned"],
        },
        "model_config_ref": model_config_ref,
        "model_config_snapshot": model_config_snapshot,
        "prompt_version": prompt_version,
        "n_ticks": n_ticks,
        "n_repeats_per_tick": n_repeats_per_tick,
        "condition": condition,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_git_commit": commit,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    schema_path = PROJECT_ROOT / "src" / "schema" / "run_manifest.schema.json"
    try:
        import jsonschema

        jsonschema.validate(manifest, json.loads(schema_path.read_text(encoding="utf-8")))
    except ImportError:
        LOGGER.warning("jsonschema not installed; skipping manifest validation")


def write_run_result(
    manifest: dict[str, Any],
    result: dict[str, Any],
    tier_suffix: str | None = None,
) -> Path:
    validate_manifest(manifest)
    run_dir = RESULTS_ROOT / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    name = manifest["condition"] + (f"_{tier_suffix}" if tier_suffix else "")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    result_path = run_dir / f"{name}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote result: %s", result_path)
    return result_path
