#!/usr/bin/env python3
"""Generate current identity/replication exports from simulation artifacts.

This is descriptive analysis only. It never promotes an incomplete result into
the numerical evidence set.
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

import networkx as nx

from identity_keys import identity_mapping, origin_id, pair_set

from sim_common import (
    SATISFICING_THRESHOLD,
    build_origin_pair_exclusions,
    cosine_similarity,
    decouple_solution_pool,
    load_embeddings,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "simulation_results"
EXPORTS = ROOT / "analysis" / "exports"
SEEDS = (11, 22, 33)
MODELS = {
    "A": "tier_1_lightly_aligned",
    "B": "tier_2_heavily_aligned",
    "C": "tier_3_nemotron_nano_4b",
    "D": "tier_4_qwen25_7b",
    "E": "tier_5_gemma2_9b",
}
N_MC = 500
N_MATCHING_POOL = 50


def result_path(model: str, condition: str, seed: int) -> Path:
    if condition == "l1":
        return RESULTS / f"layer1_seed{seed}.json"
    return RESULTS / f"layer2_{MODELS[model]}_{condition}_seed{seed}.json"


def load_complete(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == "in_progress":
        return None
    if path.name.startswith("layer2_") and payload.get("status") != "complete":
        return None
    return payload


def jaccard(left: set, right: set) -> float:
    union = left | right
    return round(len(left & right) / len(union), 6) if union else None


def eligible_graph(seed: int) -> tuple[nx.Graph, dict]:
    embeddings = load_embeddings("sbir_v2", "nomic-embed-text")
    items, decoupling = decouple_solution_pool(embeddings["items"], seed)
    _, blocked_map = build_origin_pair_exclusions(decoupling)
    problems = sorted(
        item_id for item_id, item in items.items() if item["role"] == "problem"
    )
    solutions = sorted(
        item_id for item_id, item in items.items() if item["role"] == "solution"
    )
    graph = nx.Graph()
    graph.add_nodes_from((problem_id, {"bipartite": 0}) for problem_id in problems)
    graph.add_nodes_from((decoupling[solution_id], {"bipartite": 1}) for solution_id in solutions)
    for problem_id in problems:
        for solution_id in solutions:
            if solution_id == blocked_map[problem_id]:
                continue
            similarity = cosine_similarity(
                items[problem_id]["vector"], items[solution_id]["vector"]
            )
            if similarity >= SATISFICING_THRESHOLD:
                graph.add_edge(problem_id, decoupling[solution_id])
    degree = [graph.degree(problem_id) for problem_id in problems]
    return graph, {
        "n_edges": graph.number_of_edges(),
        "deg_min": min(degree),
        "deg_mean": round(mean(degree), 6),
        "deg_max": max(degree),
        "n_problems": len(problems),
        "n_zero_degree": sum(value == 0 for value in degree),
    }


def random_matching(
    graph: nx.Graph, rng: random.Random, size: int | None = None
) -> set[tuple[str, str]] | None:
    edges = [
        (left, right) if left.startswith("P") else (right, left)
        for left, right in graph.edges
    ]
    best: set[tuple[str, str]] = set()
    for _ in range(100):
        rng.shuffle(edges)
        used_problems: set[str] = set()
        used_solutions: set[str] = set()
        pairs: set[tuple[str, str]] = set()
        for problem_id, solution_id in edges:
            if problem_id in used_problems or solution_id in used_solutions:
                continue
            pairs.add((problem_id, solution_id))
            used_problems.add(problem_id)
            used_solutions.add(solution_id)
        if len(pairs) > len(best):
            best = pairs
        if size is not None and len(pairs) >= size:
            return set(rng.sample(sorted(pairs), size))
    return best if size is None else None


def null_comparison(
    matching_pool: list[set[tuple[str, str]]],
    left: set[tuple[str, str]],
    right: set[tuple[str, str]],
    rng: random.Random,
    *,
    right_pool: list[set[tuple[str, str]]] | None = None,
) -> dict:
    right_pool = matching_pool if right_pool is None else right_pool
    for pool, target in ((matching_pool, len(left)), (right_pool, len(right))):
        if not pool or any(len(matching) < target for matching in pool):
            raise ValueError("Reference pool cannot support observed matching size")
    draws: list[float] = []
    for _ in range(N_MC):
        random_left = set(rng.sample(
            sorted(rng.choice(matching_pool)), len(left)
        ))
        random_right = set(rng.sample(
            sorted(rng.choice(right_pool)), len(right)
        ))
        draws.append(jaccard(random_left, random_right))
    draws.sort()
    observed = jaccard(left, right)
    if draws:
        p5 = draws[int((len(draws) - 1) * 0.05)]
        p95 = draws[int((len(draws) - 1) * 0.95)]
        verdict = (
            "WITHIN_REFERENCE_INTERVAL (not an equivalence test)"
            if p5 <= observed <= p95
            else "OUTSIDE_REFERENCE_INTERVAL"
        )
    else:
        p5 = p95 = None
        verdict = "NO_SUCCESSFUL_DRAWS"
    return {
        "n_a": len(left),
        "n_b": len(right),
        "observed_jaccard": observed,
        "null_mean": round(mean(draws), 6) if draws else None,
        "null_p5": round(p5, 6) if p5 is not None else None,
        "null_p95": round(p95, 6) if p95 is not None else None,
        "n_successful_draws": len(draws),
        "n_failed_draws": 0,
        "verdict": verdict,
    }


def l1_l2_export() -> dict:
    eligible_graphs = {}
    graphs = {}
    for seed in SEEDS:
        graphs[seed], eligible_graphs[str(seed)] = eligible_graph(seed)
    matching_pools = {
        seed: [
            matching
            for draw in range(N_MATCHING_POOL)
            if (matching := random_matching(
                graphs[seed], random.Random(2000 + seed * 100 + draw)
            )) is not None
        ]
        for seed in SEEDS
    }

    observed: dict[str, dict[str, float | None]] = {}
    null_model: dict[str, dict[str, dict]] = {}
    groups = {"L1": [("l1", None)], **{
        f"L2{model}": [("ranked", model)] for model in MODELS
    }}
    for label, descriptors in groups.items():
        condition, model = descriptors[0]
        payloads = {
            seed: load_complete(
                result_path(model, condition, seed)
                if model is not None
                else result_path("A", condition, seed)
            )
            for seed in SEEDS
        }
        available = {seed: pair_set(payload) for seed, payload in payloads.items() if payload}
        observed[label] = {}
        null_model[label] = {}
        for seed_a, seed_b in combinations(sorted(available), 2):
            key = f"{seed_a}_vs_{seed_b}"
            observed[label][key] = jaccard(available[seed_a], available[seed_b])
            null_model[label][key] = null_comparison(
                matching_pools[seed_a],
                available[seed_a],
                available[seed_b],
                random.Random(1000 + seed_a * 10 + seed_b),
                right_pool=matching_pools[seed_b],
            )
    return {
        "eligible_graphs": eligible_graphs,
        "observed_jaccard": observed,
        "null_model": null_model,
        "n_mc": N_MC,
        "method": (
            "For each seed, construct eligible bipartite edges after solution-pool "
            "decoupling, origin exclusion, and cosine >= 0.60. For each observed "
            "pair of injective matching sizes, precompute 50 random greedy "
            "maximal matchings and sample the target size from that pool; compare "
            "500 random Jaccards. All identities are canonical source solution IDs. "
            "Each side uses its own seed graph and pool. The existing pool algorithm "
            "retains the largest of 100 shuffled-edge greedy matchings for each pool entry; "
            "it is not uniform over feasible matchings. This static, size-conditioned "
            "reference does not preserve arrival schedules, energy, or model preferences "
            "and does not establish path dependence. Incomplete cells are excluded."
        ),
    }


def summary_rows() -> list[dict]:
    summary = json.loads(
        (EXPORTS / "coupling_summary.json").read_text(encoding="utf-8")
    )
    rows = []
    for row in summary.get("layer3_autonomous", []):
        search = row.get("search_behavior") or {}
        model = next((m for m, tier in MODELS.items() if tier == row.get("alignment_tier")), None)
        if model is None:
            raise ValueError("Unknown comparison model in L3 summary")
        payload = load_complete(result_path(model, "autonomous", row["seed"]))
        if payload is None:
            continue
        mapping = identity_mapping(payload)
        if row.get("n_couplings") != len(payload.get("couplings", [])):
            raise ValueError("Stale coupling summary")
        rows.append({
            "model": next(
                (model for model, tier in MODELS.items()
                 if tier == row.get("alignment_tier")),
                None,
            ),
            "seed": row["seed"],
            "status": "complete",
            "n_couplings": row.get("n_couplings"),
            "collisions": row.get("collisions"),
            "modal_select_attempts": search.get("n_modal_attempts"),
            "collision_rate": (
                round(row["collisions"] / search["n_modal_attempts"], 6)
                if row.get("collisions") is not None
                and search.get("n_modal_attempts")
                else None
            ),
            "hhi": search.get("hhi"),
            "entropy": search.get("entropy"),
            "neff": search.get("neff"),
            "gini": search.get("gini"),
            "below_0_60_pct": (
                round(100 * row["final_matching"]["frac_sim_below_0_60"], 2)
                if row.get("final_matching", {}).get("frac_sim_below_0_60") is not None
                else None
            ),
            "sim_mean": row.get("similarity_mean"),
            "top_solutions": search.get("top_solutions", []),
            "top_solutions_origin": [
                (origin_id(mapping, sid), count)
                for sid, count in search.get("top_solutions", [])
            ],
        })
    return rows


def variability(rows: list[dict]) -> list[dict]:
    metrics = (
        "hhi", "entropy", "collision_rate", "collisions", "gini", "neff",
        "n_couplings", "below_0_60_pct", "sim_mean",
    )
    output = []
    for metric in metrics:
        values = {
            model: [row[metric] for row in rows if row["model"] == model and row[metric] is not None]
            for model in MODELS
        }
        model_values = {model: vals for model, vals in values.items() if vals}
        if not model_values:
            continue
        entry = {"metric": metric}
        for model, vals in model_values.items():
            entry[f"{model}_mean"] = round(mean(vals), 6)
            entry[f"{model}_sd"] = round(pstdev(vals), 6)
            entry[f"{model}_range"] = round(max(vals) - min(vals), 6)
            entry[f"{model}_vals"] = vals
        output.append(entry)
    return output


def top_jaccards(rows: list[dict], model: str, top_n: int = 10) -> dict:
    selected = {
        row["seed"]: {solution for solution, _ in row["top_solutions_origin"][:top_n]}
        for row in rows if row["model"] == model
    }
    return {
        f"{left}_vs_{right}": jaccard(selected[left], selected[right])
        for left, right in combinations(sorted(selected), 2)
    }


def l3_exports() -> tuple[dict, dict, dict]:
    rows = summary_rows()
    def model_export(model: str) -> dict:
        selected = [row for row in rows if row["model"] == model]
        return {
            "status": "complete" if len(selected) == 3 else "partial",
            "seeds": sorted(row["seed"] for row in selected),
            "variability": variability(selected),
            "top10_jaccard": top_jaccards(rows, model),
            "method": (
                "Rows are read from the current coupling summary; L3 identity "
                "diagnostics use modal SELECT frequencies including collision skips, mapped "
                "to source solution IDs before cross-seed overlap. Top-10 ties retain "
                "the existing frequency export ordering; overlap is descriptive, "
                "not a null-adjusted test or evidence of path dependence."
            ),
        }

    a_export = model_export("A")
    b_export = model_export("B")
    cross = {
        "status_by_model": {
            model: {
                "n_complete": sum(row["model"] == model for row in rows),
                "seeds": sorted(row["seed"] for row in rows if row["model"] == model),
            }
            for model in MODELS
        },
        "variability": variability(rows),
        "top10_jaccard_by_model": {
            model: top_jaccards(rows, model) for model in MODELS
        },
        "rows": [
            {key: value for key, value in row.items() if key not in ("top_solutions", "top_solutions_origin")}
            for row in rows
        ],
        "method": a_export["method"],
    }
    return a_export, b_export, cross


def l3_collision_export(rows: list[dict]) -> dict:
    return {
        "status": "complete",
        "metric_definition": (
            "Collision rate is collision-driven skipped modal SELECT events "
            "divided by all modal SELECT attempts in each completed L3 cell."
        ),
        "rows": [
            {
                "model": row["model"],
                "seed": row["seed"],
                "collisions": row["collisions"],
                "modal_select_attempts": row["modal_select_attempts"],
                "collision_rate": row["collision_rate"],
            }
            for row in rows
        ],
    }


def write_json(name: str, payload: dict) -> None:
    (EXPORTS / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    write_json("l1_l2_identity_null_jaccard.json", l1_l2_export())
    rows = summary_rows()
    model_a, model_b, cross = l3_exports()
    write_json("l3_model_a_replication_seeds_11_22_33.json", model_a)
    write_json("l3_model_b_replication_seeds_11_22_33.json", model_b)
    write_json("l3_cross_model_comparison_seeds_11_22_33.json", cross)
    write_json("l3_collision_rates_from_summary.json", l3_collision_export(rows))
    print("wrote identity and replication exports")


if __name__ == "__main__":
    main()
