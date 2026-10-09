#!/usr/bin/env python3
"""Run a dependency-free L1 RuleProxy execution proof on the toy corpus."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


THRESHOLD = 0.60
LOCAL_SEARCH_N = 2
INITIAL_ENERGY = 1.0
ENERGY_DECAY = 0.15
ARRIVAL_WINDOW_FRACTION = 0.4


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def run(seed: int, n_ticks: int, corpus_dir: Path) -> dict:
    problems = json.loads((corpus_dir / "problems.json").read_text())
    source_solutions = json.loads((corpus_dir / "solutions.json").read_text())
    solution_ids = sorted(item["id"] for item in source_solutions)

    decoupling_rng = random.Random(seed)
    origin_ids = solution_ids.copy()
    decoupling_rng.shuffle(origin_ids)
    by_origin = {item["id"]: item for item in source_solutions}
    solutions = {
        display_id: {
            **by_origin[origin_id],
            "id": display_id,
            "origin_solution_id": origin_id,
        }
        for display_id, origin_id in zip(solution_ids, origin_ids)
    }
    blocked = {
        problem["id"]: next(
            display_id
            for display_id, solution in solutions.items()
            if solution["origin_solution_id"] == problem["id"].replace("P", "S")
        )
        for problem in problems
    }

    ids = [item["id"] for item in problems] + solution_ids
    rng = random.Random(seed)
    window = max(1, int(n_ticks * ARRIVAL_WINDOW_FRACTION))
    arrivals = {item_id: rng.randint(0, window) for item_id in sorted(ids)}
    problem_by_id = {item["id"]: item for item in problems}
    energy: dict[str, float] = {}
    coupled_problems: set[str] = set()
    coupled_solutions: set[str] = set()
    evaluations: list[dict] = []
    couplings: list[dict] = []

    for tick in range(n_ticks + 1):
        active_problems = [
            item_id
            for item_id in sorted(problem_by_id)
            if arrivals[item_id] <= tick and item_id not in coupled_problems
        ]
        active_solutions = [
            item_id
            for item_id in solution_ids
            if arrivals[item_id] <= tick and item_id not in coupled_solutions
        ]
        rng.shuffle(active_problems)
        for problem_id in active_problems:
            available = [
                solution_id
                for solution_id in active_solutions
                if solution_id != blocked[problem_id]
                and solution_id not in coupled_solutions
            ]
            candidates = rng.sample(available, min(LOCAL_SEARCH_N, len(available)))
            for solution_id in candidates:
                energy_before = energy.get(problem_id, INITIAL_ENERGY)
                if energy_before < ENERGY_DECAY:
                    break
                energy[problem_id] = energy_before - ENERGY_DECAY
                similarity = cosine(
                    problem_by_id[problem_id]["vector"],
                    solutions[solution_id]["vector"],
                )
                evaluations.append(
                    {
                        "tick": tick,
                        "problem_id": problem_id,
                        "solution_id": solution_id,
                        "cosine_similarity": round(similarity, 6),
                        "energy_before": round(energy_before, 6),
                        "accepted": similarity >= THRESHOLD,
                    }
                )
                if similarity >= THRESHOLD:
                    coupled_problems.add(problem_id)
                    coupled_solutions.add(solution_id)
                    couplings.append(
                        {
                            "tick": tick,
                            "problem_id": problem_id,
                            "solution_id": solution_id,
                            "cosine_similarity": round(similarity, 6),
                        }
                    )
                    break

    return {
        "scientific_replication": False,
        "seed": seed,
        "n_ticks": n_ticks,
        "threshold": THRESHOLD,
        "local_search_n": LOCAL_SEARCH_N,
        "initial_energy": INITIAL_ENERGY,
        "energy_decay_per_evaluation": ENERGY_DECAY,
        "n_problems": len(problems),
        "n_solutions": len(source_solutions),
        "n_evaluations": len(evaluations),
        "n_couplings": len(couplings),
        "couplings": couplings,
        "evaluations": evaluations,
        "uncoupled_problem_ids": sorted(set(problem_by_id) - coupled_problems),
        "unused_solution_ids": sorted(set(solution_ids) - coupled_solutions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-ticks", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/toy_reproduction.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(args.seed, args.n_ticks, root / "data" / "toy_corpus")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"toy execution complete: couplings={result['n_couplings']} "
        f"evaluations={result['n_evaluations']} output={args.output}"
    )


if __name__ == "__main__":
    main()
