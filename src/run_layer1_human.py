#!/usr/bin/env python3
"""Layer 1 RuleProxy simulation on real SBIR embedding pairs.

Uses the fixed satisficing threshold (0.60) shared with Layer 2 ranked runs.
Solution display IDs are decoupled from origin abstracts via a seeded shuffle
before each run so same-index reunification shortcuts are removed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from sim_common import (
    LOCAL_SEARCH_N,
    SATISFICING_THRESHOLD,
    USE_PERCEPTUAL_NOISE,
    RuleProxy,
    annotate_coupling,
    build_arrival_schedule,
    build_origin_pair_exclusions,
    compute_same_origin_reunion_rate,
    cosine_similarity,
    decouple_solution_pool,
    eligible_solutions_for_problem,
    load_embeddings,
    make_run_manifest,
    similarity_distribution_diagnostic,
    summarize_acceptance_counts,
    trace_variance,
    SIMULATION_RESULTS_ROOT,
)

# The module/logger and A_human artifact label are retained for compatibility
# with existing result files; scientifically this is the L1 RuleProxy run.
LOGGER = logging.getLogger("run_layer1_human")
DEFAULT_SEEDS = [11, 22, 33]


def coupled_variance_at_tick(
    couplings: list[dict],
    items: dict[str, dict],
    tick_cutoff: int,
) -> float:
    vectors: list[list[float]] = []
    for coupling in couplings:
        if coupling["tick"] > tick_cutoff:
            continue
        vectors.append(items[coupling["problem_id"]]["vector"])
        vectors.append(items[coupling["solution_id"]]["vector"])
    return round(trace_variance(vectors), 6)


def write_layer1_result(seed: int, result: dict) -> Path:
    SIMULATION_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = SIMULATION_RESULTS_ROOT / f"layer1_seed{seed}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return out_path


def run(
    seed: int,
    corpus_id: str,
    embedding_model_key: str,
    n_ticks: int,
    satisficing_threshold: float = SATISFICING_THRESHOLD,
    use_perceptual_noise: bool = USE_PERCEPTUAL_NOISE,
) -> Path:
    embeddings = load_embeddings(corpus_id, embedding_model_key)
    items, solution_decoupling = decouple_solution_pool(embeddings["items"], seed)
    origin_pair_exclusions, blocked_map = build_origin_pair_exclusions(solution_decoupling)
    similarity_diagnostic = similarity_distribution_diagnostic(items)
    problem_ids = sorted(i for i, item in items.items() if item["role"] == "problem")
    solution_ids = sorted(i for i, item in items.items() if item["role"] == "solution")

    rng = random.Random(seed)
    arrivals = build_arrival_schedule(problem_ids + solution_ids, n_ticks, rng)
    proxy = RuleProxy(
        rng,
        threshold=satisficing_threshold,
        use_perceptual_noise=use_perceptual_noise,
    )

    coupled_problems: set[str] = set()
    coupled_solutions: set[str] = set()
    couplings: list[dict] = []
    evaluations: list[dict] = []

    for tick in range(n_ticks + 1):
        active_problems = [
            p for p in problem_ids
            if arrivals[p] <= tick and p not in coupled_problems
        ]
        active_solutions = [
            s for s in solution_ids
            if arrivals[s] <= tick and s not in coupled_solutions
        ]
        rng.shuffle(active_problems)

        for problem_id in active_problems:
            available = eligible_solutions_for_problem(
                problem_id, active_solutions, blocked_map, coupled_solutions
            )
            if not available:
                continue
            # Local search: bounded breadth, no global scan of the pool.
            candidates = rng.sample(available, min(LOCAL_SEARCH_N, len(available)))
            for solution_id in candidates:
                if not proxy.can_evaluate(problem_id):
                    break
                sim = cosine_similarity(
                    items[problem_id]["vector"], items[solution_id]["vector"]
                )
                record = proxy.evaluate(problem_id, sim)
                evaluations.append(
                    {"tick": tick, "problem_id": problem_id,
                     "solution_id": solution_id, **record}
                )
                if record["accepted"]:
                    coupled_problems.add(problem_id)
                    coupled_solutions.add(solution_id)
                    couplings.append(
                        annotate_coupling(
                            {
                                "tick": tick,
                                "problem_id": problem_id,
                                "solution_id": solution_id,
                                "cosine_similarity": record["true_similarity"],
                                "acceptance_type": record["acceptance_type"],
                                "modal_agreement": None,
                                "repeat_choices": None,
                            },
                            solution_decoupling,
                        )
                    )
                    break

    variance_tick_0 = coupled_variance_at_tick(couplings, items, tick_cutoff=0)
    variance_tick_100 = coupled_variance_at_tick(couplings, items, tick_cutoff=n_ticks)
    manifest = make_run_manifest(
        seed=seed,
        corpus_id=corpus_id,
        embeddings=embeddings,
        condition="A_human",
        n_ticks=n_ticks,
    )
    result = {
        "run_manifest": manifest,
        "condition": "A_human",
        "alignment_tier": None,
        "similarity_diagnostic": similarity_diagnostic,
        "solution_decoupling": solution_decoupling,
        "origin_pair_exclusions": origin_pair_exclusions,
        "same_origin_reunion_rate": compute_same_origin_reunion_rate(
            couplings, solution_decoupling
        ),
        "satisficing_threshold": satisficing_threshold,
        "use_perceptual_noise": use_perceptual_noise,
        "arrival_schedule": arrivals,
        "couplings": couplings,
        "n_evaluations": len(evaluations),
        "evaluations": evaluations,
        "acceptance_summary": summarize_acceptance_counts(couplings),
        "variance": {
            "tick_0": variance_tick_0,
            f"tick_{n_ticks}": variance_tick_100,
        },
        "uncoupled_problem_ids_final": sorted(set(problem_ids) - coupled_problems),
        "uncoupled_solution_ids_final": sorted(set(solution_ids) - coupled_solutions),
        "audit_log": None,
    }
    out_path = write_layer1_result(seed, result)
    LOGGER.info(
        "Condition A complete: seed=%d threshold=%.6f couplings=%d evaluations=%d uncoupled_problems=%d",
        seed,
        satisficing_threshold,
        len(couplings), len(evaluations),
        len(problem_ids) - len(coupled_problems),
    )
    LOGGER.info(
        "Variance summary: tick_0=%.6f tick_%d=%.6f | wrote %s",
        variance_tick_0,
        n_ticks,
        variance_tick_100,
        out_path,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Layer 1 (RuleProxy bounded-search control)."
    )
    parser.add_argument("--seed", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--corpus-id", default="sbir_v1")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--n-ticks", type=int, default=100)
    parser.add_argument(
        "--satisficing-threshold",
        type=float,
        default=SATISFICING_THRESHOLD,
        help="Perceived-similarity acceptance threshold (default 0.60).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for seed in args.seed:
        run(
            seed,
            args.corpus_id,
            args.embedding_model,
            args.n_ticks,
            satisficing_threshold=args.satisficing_threshold,
        )


if __name__ == "__main__":
    main()
