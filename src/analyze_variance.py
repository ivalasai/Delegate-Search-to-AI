#!/usr/bin/env python3
"""Descriptive analysis over real embeddings, across the replication grid.

Metrics
-------
Exploratory variance diagnostic (legacy dN), per run:
    V_pool_t0   = mean per-dimension variance of ALL pool items' embeddings
                  at the start of the run (the available search space).
    V_coupled   = same statistic over the items that ended up coupled by the
                  final tick.
    dN = (V_coupled - V_pool_t0) / V_pool_t0
Negative dN = the coupled set is less dispersed than the space it was
drawn from (variance collapse). This is exploratory and is not a primary
Stage-1 outcome.

Model labels are descriptive robustness identifiers. Legacy ``alignment_tier``
strings remain in artifact paths and manifests for compatibility, but they are
not treated as an experimental alignment-intensity variable.

Aggregation: results are grouped over the full grid (seeds x corpora x
models) and reported as distributions (mean, 95% CI, n), never as single-seed
point estimates.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from identity_keys import pair_set
import numpy as np
from scipy import stats

from sim_common import (
    ENERGY_DECAY_PER_EVAL,
    INITIAL_ENERGY,
    SATISFICING_THRESHOLD,
    build_origin_pair_exclusions,
    cosine_similarity,
    decouple_solution_pool,
    load_embeddings,
)

LOGGER = logging.getLogger("analyze_variance")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = PROJECT_ROOT / "data" / "simulation_results"
# Commit-safe exports (data/simulation_results/ is gitignored).
ANALYSIS_EXPORTS_ROOT = PROJECT_ROOT / "analysis" / "exports"

# Default ceiling for sbir_v2 at seed 11; recomputed dynamically via
# compute_matching_ceiling() when --coupling-summary runs without --ceiling.
SBIR_V2_CEILING = 54
LAYER1_SEEDS = (11, 22, 33)
LAYER1_MATCHED_TICKS = 17
LAYER1_EXTENDED_TICKS = 100
# Directed (problem, solution) pairs after per-problem origin exclusion.
SBIR_V2_FEASIBLE_PAIR_UNIVERSE = 90 * 89
LAYER2_DEFAULT_RESULT = "layer2_tier_1_lightly_aligned_ranked_seed11.json"
LAYER2_MODEL_B_RESULT = "layer2_tier_2_heavily_aligned_ranked_seed11.json"
LAYER3_MODEL_A_RESULT = "layer2_tier_1_lightly_aligned_autonomous_seed11.json"
LAYER3_MODEL_B_RESULT = "layer2_tier_2_heavily_aligned_autonomous_seed11.json"
LAYER3_MODEL_A_SEEDS = LAYER1_SEEDS  # Model A L3 replication seeds
LAYER3_VALIDATION_SEED = 11  # locked forensic targets apply to seed 11 only
THRESHOLD_SENSITIVITY_LEVELS = (0.55, 0.60, 0.65)

COMPARISON_MODELS = (
    ("A", "tier_1_lightly_aligned", "mistral_7b_instruct"),
    ("B", "tier_2_heavily_aligned", "llama3_1_8b"),
    ("C", "tier_3_nemotron_nano_4b", "nemotron_3_nano_4b"),
    ("D", "tier_4_qwen25_7b", "qwen25_7b"),
    ("E", "tier_5_gemma2_9b", "gemma2_9b"),
)


def classify_result_status(
    result: dict | None, grid_state_status: str | None = None
) -> str:
    """Classify a result for exports without treating partial JSON as complete."""
    if result is not None:
        status = result.get("status")
        if status == "complete":
            return "complete"
        if status == "in_progress":
            return "in_progress"
        # Layer 1's validated legacy files predate an explicit status field.
        if "run_manifest" in result and "couplings" in result:
            return "complete"
    if grid_state_status == "running":
        return "in_progress"
    if grid_state_status == "cancelled":
        return "cancelled"
    return "not_run"


def _load_result_if_present(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_complete_result_path(path: Path) -> bool:
    return classify_result_status(_load_result_if_present(path)) == "complete"


def build_experiment_status(
    simulation_results_dir: Path = SIMULATION_RESULTS_ROOT,
    seeds: tuple[int, ...] = LAYER1_SEEDS,
) -> list[dict]:
    """Return complete/in-progress/not-run rows for the expected Stage-1 grid."""
    state_path = simulation_results_dir / "grid_state.json"
    state = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8")).get("jobs", {})
        except (OSError, json.JSONDecodeError):
            state = {}

    rows: list[dict] = []

    for seed in seeds:
        path = simulation_results_dir / f"layer1_seed{seed}.json"
        result = _load_result_if_present(path)
        rows.append({
            "run": f"L1 seed {seed} ({LAYER1_MATCHED_TICKS} ticks)",
            "layer": "L1",
            "architecture": "L1_HumanProxy",
            "model": None,
            "condition": "HumanProxy",
            "seed": seed,
            "status": classify_result_status(result),
            "source_file": path.name if path.is_file() else None,
        })

    for model, tier, config_stem in COMPARISON_MODELS:
        for condition, layer in (("ranked", "L2"), ("autonomous", "L3")):
            for seed in seeds:
                path = simulation_results_dir / (
                    f"layer2_{tier}_{condition}_seed{seed}.json"
                )
                key = f"{config_stem}_{condition}_seed{seed}"
                result = _load_result_if_present(path)
                state_status = (state.get(key) or {}).get("status")
                rows.append({
                    "run": (
                        f"{layer} Model {model} "
                        f"{'model selection' if condition == 'autonomous' else 'ranked'} "
                        f"seed {seed}"
                    ),
                    "layer": layer,
                    "architecture": (
                        "L3_model_selection"
                        if condition == "autonomous"
                        else "L2_ranked"
                    ),
                    "model": model,
                    "condition": (
                        "model selection"
                        if condition == "autonomous"
                        else condition
                    ),
                    "seed": seed,
                    "status": classify_result_status(result, state_status),
                    "source_file": path.name if path.is_file() else None,
                })
    return rows


def trace_variance(vectors: np.ndarray) -> float:
    """Mean per-dimension variance; 0.0 for fewer than 2 vectors."""
    if vectors.shape[0] < 2:
        return 0.0
    return float(np.mean(np.var(vectors, axis=0)))


def load_decoupled_lookup(
    corpus_id: str,
    model_id: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Embedding vectors after seeded solution-pool decoupling (simulation space)."""
    embeddings = load_embeddings(corpus_id, model_id)
    items, _ = decouple_solution_pool(embeddings["items"], seed)
    return {
        item_id: np.array(item["vector"])
        for item_id, item in items.items()
    }


def compute_matching_ceiling(
    corpus_id: str,
    model_id: str,
    seed: int,
    *,
    threshold: float = SATISFICING_THRESHOLD,
) -> int:
    """Maximum bipartite matching on the eligible decoupled graph."""
    embeddings = load_embeddings(corpus_id, model_id)
    items, solution_decoupling = decouple_solution_pool(embeddings["items"], seed)
    _, blocked_map = build_origin_pair_exclusions(solution_decoupling)
    problem_ids = sorted(i for i, item in items.items() if item["role"] == "problem")
    solution_ids = sorted(i for i, item in items.items() if item["role"] == "solution")

    graph = nx.Graph()
    graph.add_nodes_from(problem_ids, bipartite=0)
    graph.add_nodes_from(solution_ids, bipartite=1)
    for problem_id in problem_ids:
        problem_vector = items[problem_id]["vector"]
        blocked = blocked_map.get(problem_id)
        for solution_id in solution_ids:
            if solution_id == blocked:
                continue
            if cosine_similarity(problem_vector, items[solution_id]["vector"]) >= threshold:
                graph.add_edge(problem_id, solution_id)

    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True)
    return len(matching)


def analyze_run(result: dict) -> dict:
    manifest = result["run_manifest"]
    lookup = load_decoupled_lookup(
        manifest["corpus_id"],
        manifest["embedding_model"]["model_id"],
        manifest["seed"],
    )

    pool_vectors = np.stack(list(lookup.values()))
    coupled_ids = [
        item_id
        for coupling in result["couplings"]
        for item_id in (coupling["problem_id"], coupling["solution_id"])
    ]
    coupled_vectors = (
        np.stack([lookup[i] for i in coupled_ids]) if coupled_ids else np.empty((0, 1))
    )

    v_pool = trace_variance(pool_vectors)
    v_coupled = trace_variance(coupled_vectors)
    # dN is undefined (not "total collapse") when fewer than 2 items coupled.
    if coupled_vectors.shape[0] < 2 or v_pool <= 0:
        dn = float("nan")
    else:
        dn = (v_coupled - v_pool) / v_pool

    agreements = [
        c["modal_agreement"] for c in result["couplings"]
        if c.get("modal_agreement") is not None
    ]
    return {
        "run_id": manifest["run_id"],
        "seed": manifest["seed"],
        "corpus_id": manifest["corpus_id"],
        "embedding_model_id": manifest["embedding_model"]["model_id"],
        "condition": result["condition"],
        "alignment_tier": result.get("alignment_tier"),
        "n_couplings": len(result["couplings"]),
        "n_coupled_items": len(set(coupled_ids)),
        "v_pool_t0": round(v_pool, 6),
        "v_coupled_final": round(v_coupled, 6),
        "novelty_decay_rate": round(dn, 6) if not np.isnan(dn) else None,
        "mean_modal_agreement": round(float(np.mean(agreements)), 4) if agreements else None,
        "mean_coupling_similarity": round(
            float(np.mean([c["cosine_similarity"] for c in result["couplings"]])), 4
        ) if result["couplings"] else None,
    }


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return (None, None)
    arr = np.array(values)
    sem = stats.sem(arr)
    half = sem * stats.t.ppf(0.975, len(arr) - 1)
    return (float(arr.mean() - half), float(arr.mean() + half))


def alignment_gradient(per_run: list[dict]) -> dict:
    """Retain the API while disabling the retired alignment-intensity test."""
    return {
        "status": "not_applicable",
        "detail": (
            "Legacy alignment-tier regression is disabled. Model A–E labels "
            "identify descriptive robustness implementations, not an ordinal "
            "alignment-intensity treatment."
        ),
        "n_runs": len(per_run),
    }


def coupling_pairs(result: dict) -> set[tuple[str, str]]:
    """Cross-run pairs use stable source identities, not permuted display IDs."""
    return pair_set(result)


def pairwise_jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float | None:
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def gini_coefficient(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    arr = np.sort(values.astype(float))
    total = float(arr.sum())
    if total <= 0:
        return None
    n = arr.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * arr) / (n * total)) - (n + 1.0) / n)


def entropy_bits(counts: np.ndarray) -> float | None:
    """Shannon entropy in bits over a frequency vector (nonzero mass only)."""
    arr = np.asarray(counts, dtype=float)
    total = float(arr.sum())
    if total <= 0 or arr.size == 0:
        return None
    probs = arr / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def herfindahl_hirschman(counts: np.ndarray) -> float | None:
    """Herfindahl–Hirschman Index: sum of squared shares."""
    arr = np.asarray(counts, dtype=float)
    total = float(arr.sum())
    if total <= 0 or arr.size == 0:
        return None
    shares = arr / total
    return float(np.sum(shares * shares))


def frequency_concentration(solution_ids: list[str], *, top_n: int = 10) -> dict:
    """Unique count, Gini, entropy, HHI, and Neff over selection frequencies."""
    counter = Counter(solution_ids)
    counts = np.array(list(counter.values()), dtype=float) if counter else np.empty(0)
    gini = gini_coefficient(counts)
    # Single-category distributions are perfectly equal (Gini 0) when n>=1.
    if gini is None and counts.size == 1:
        gini = 0.0
    hhi = herfindahl_hirschman(counts)
    neff = (1.0 / hhi) if hhi and hhi > 0 else None
    return {
        "unique_solutions": len(counter),
        "selection_gini": round(gini, 6) if gini is not None else None,
        "selection_entropy": (
            round(entropy_bits(counts), 6) if counts.size else None
        ),
        "selection_hhi": round(hhi, 8) if hhi is not None else None,
        "selection_neff": round(neff, 6) if neff is not None else None,
        "n_selection_events": int(counts.sum()) if counts.size else 0,
        "top_solutions": counter.most_common(top_n),
    }


def pairwise_cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise distances: 1 - cosine similarity."""
    n = vectors.shape[0]
    if n < 2:
        return np.empty(0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = vectors / norms
    sims = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return 1.0 - sims[iu]


def replay_threshold_couplings(
    evaluations: list[dict],
    threshold: float,
    *,
    ceiling: int = SBIR_V2_CEILING,
) -> dict:
    """Hypothetical couplings by replaying logged evaluations in order.

    Threshold sensitivity is computed by replaying the recorded evaluation
    trace under alternative acceptance thresholds. At 0.65, the same observed
    evaluations are rescored against a stricter bar, so totals match the
    original 0.60 run. At 0.55, a satisficing agent would have stopped
    searching earlier under the easier bar, so the replay truncates each
    problem's trace at its first 0.55-qualifying candidate — producing
    genuinely fewer total evaluations, not an inconsistency.

    Mechanically: once a problem is accepted under the replay threshold,
    later logged evaluations for that problem are skipped (``problem_id in
    coupled_problems``). Raising the threshold may still undercount *couplings*
    vs a fresh run because the original log stops candidate search after the
    first accept at the recorded 0.60 threshold.
    """
    coupled_problems: set[str] = set()
    coupled_solutions: set[str] = set()
    energy: dict[str, float] = {}
    accepted_sims: list[float] = []
    n_evaluated = 0
    n_rejected = 0
    for ev in evaluations:
        problem_id = ev["problem_id"]
        solution_id = ev["solution_id"]
        if problem_id in coupled_problems or solution_id in coupled_solutions:
            continue
        energy_before = energy.get(problem_id, INITIAL_ENERGY)
        if energy_before < ENERGY_DECAY_PER_EVAL:
            continue
        energy[problem_id] = energy_before - ENERGY_DECAY_PER_EVAL
        n_evaluated += 1
        sim = float(ev["true_similarity"])
        if sim >= threshold:
            coupled_problems.add(problem_id)
            coupled_solutions.add(solution_id)
            accepted_sims.append(sim)
        else:
            n_rejected += 1
    n_couplings = len(accepted_sims)
    return {
        "threshold": threshold,
        "n_couplings": n_couplings,
        "pct_ceiling": (
            round(100.0 * n_couplings / ceiling, 2) if ceiling else None
        ),
        "n_evaluated": n_evaluated,
        "n_rejected": n_rejected,
        "acceptance_rate": (
            round(n_couplings / n_evaluated, 6) if n_evaluated else None
        ),
        "mean_sim": (
            round(float(np.mean(accepted_sims)), 6) if accepted_sims else None
        ),
        "min_sim": round(min(accepted_sims), 6) if accepted_sims else None,
        "max_sim": round(max(accepted_sims), 6) if accepted_sims else None,
    }


def threshold_sensitivity(
    evaluations: list[dict],
    levels: tuple[float, ...] = THRESHOLD_SENSITIVITY_LEVELS,
    *,
    ceiling: int = SBIR_V2_CEILING,
) -> dict[str, dict]:
    return {
        f"{level:.2f}": replay_threshold_couplings(
            evaluations, level, ceiling=ceiling
        )
        for level in levels
    }


def _modal_choice(values: list) -> object | None:
    """Match run_layer2_ai.modal: majority with first-occurrence tie-break."""
    if not values:
        return None
    counts = Counter(json.dumps(v, sort_keys=True) for v in values)
    top_serialized, _ = counts.most_common(1)[0]
    return json.loads(top_serialized)


def l3_modal_select_events(result: dict, audit_log: list[dict]) -> dict:
    """Replay L3 model-selection modal SELECTs; count collisions and selection freqs.

    Primary concentration support for L3: every modal SELECT, including those
    that collision-skip because the solution was already claimed.
    Distinct from final-matching diversity (accepted couplings only).
    """
    arrivals = result["arrival_schedule"]
    by_tick: dict[int, dict[str, dict[int, object]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for entry in audit_log:
        parsed = entry.get("parsed") or {}
        problem_id = entry["problem_id"]
        by_tick[entry["tick"]][problem_id][entry["repeat"]] = parsed.get(problem_id)

    coupled_problems: set[str] = set()
    coupled_solutions: set[str] = set()
    modal_selects: list[str] = []
    collisions = 0
    collisions_by_tick: list[dict] = []
    n_ticks = int(result["run_manifest"]["n_ticks"])
    problem_ids = sorted(arrivals)

    for tick in range(n_ticks + 1):
        active_problems = [
            pid for pid in problem_ids
            if arrivals[pid] <= tick and pid not in coupled_problems
        ]
        tick_collisions = 0
        tick_couplings = 0
        tick_selects: list[str] = []
        for problem_id in active_problems:
            repeats = by_tick[tick].get(problem_id)
            if not repeats:
                continue
            per_repeat = [repeats.get(r) for r in sorted(repeats)]
            modal_choice = _modal_choice(per_repeat)
            if modal_choice is None:
                continue
            modal_selects.append(str(modal_choice))
            tick_selects.append(str(modal_choice))
            if modal_choice in coupled_solutions:
                collisions += 1
                tick_collisions += 1
                continue
            coupled_problems.add(problem_id)
            coupled_solutions.add(str(modal_choice))
            tick_couplings += 1
        collisions_by_tick.append({
            "tick": tick,
            "n_modal_selects": len(tick_selects),
            "n_couplings": tick_couplings,
            "n_collisions": tick_collisions,
            "top_solutions": Counter(tick_selects).most_common(5),
        })

    freq = frequency_concentration(modal_selects)
    return {
        "collisions": collisions,
        "modal_selects": modal_selects,
        "collisions_by_tick": collisions_by_tick,
        **freq,
        "selection_support": "modal_SELECT_incl_collisions",
    }


def load_audit_log(result: dict, result_path: Path) -> list[dict] | None:
    """Load companion audit log for L3 model-selection concentration metrics."""
    audit_ref = result.get("audit_log")
    candidates: list[Path] = []
    if isinstance(audit_ref, str) and audit_ref:
        ref = Path(audit_ref)
        candidates.append(ref if ref.is_absolute() else PROJECT_ROOT / ref)
    candidates.append(result_path.with_name(result_path.stem + "_log.json"))
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
    return None


def analyze_coupling_metrics(
    result: dict,
    ceiling: int = SBIR_V2_CEILING,
    *,
    audit_log: list[dict] | None = None,
    apply_ceiling: bool = True,
) -> dict:
    """Post-hoc coupling summary for simulation_results JSON runs."""
    manifest = result["run_manifest"]
    lookup = load_decoupled_lookup(
        manifest["corpus_id"],
        manifest["embedding_model"]["model_id"],
        manifest["seed"],
    )
    couplings = result["couplings"]
    similarities = [float(c["cosine_similarity"]) for c in couplings]
    solution_ids = sorted({c["solution_id"] for c in couplings})
    problem_ids = sorted({c["problem_id"] for c in couplings})
    solution_vectors = (
        np.stack([lookup[sid] for sid in solution_ids])
        if solution_ids else np.empty((0, 1))
    )
    dispersion = round(trace_variance(solution_vectors), 6)
    distances = pairwise_cosine_distances(solution_vectors)
    # Spatial concentration: Gini on pairwise cosine distances among accepted
    # solutions (decoupled vectors). Distinct from selection-frequency Gini.
    concentration = (
        round(gini_coefficient(distances), 6)
        if distances.size else None
    )
    n_couplings = len(couplings)
    condition = manifest.get("condition", result.get("condition", "unknown"))
    is_model_selection = condition in ("autonomous", "C_autonomous")

    # FINAL MATCHING diversity: frequencies over accepted coupling solution IDs.
    # For injective one-to-one matchings this is maximally diverse (HHI = 1/n).
    accepted_freq = frequency_concentration([c["solution_id"] for c in couplings])
    final_matching = {
        "n_couplings": n_couplings,
        "unique_problems": len(problem_ids),
        "unique_solutions": len(solution_ids),
        "hhi": accepted_freq["selection_hhi"],
        "entropy": accepted_freq["selection_entropy"],
        "neff": accepted_freq["selection_neff"],
        "gini": accepted_freq["selection_gini"],
        "support": "accepted_couplings",
    }
    if similarities:
        n_below = sum(1 for s in similarities if s < SATISFICING_THRESHOLD)
        final_matching["n_sim_below_0_60"] = n_below
        final_matching["n_sim_at_or_above_0_60"] = n_couplings - n_below
        final_matching["frac_sim_below_0_60"] = round(n_below / n_couplings, 6)

    # Selection-frequency / SEARCH BEHAVIOR:
    #   L1/L2 ranked: solutions in logged evaluation attempts
    #   L3 model selection: modal SELECT distribution including collision skips
    # Keep this SEPARATE from final_matching above.
    collisions = None
    selection_support = "evaluation_attempts"
    search_behavior = None
    if is_model_selection and audit_log is not None:
        l3 = l3_modal_select_events(result, audit_log)
        # Top-level unique_solutions stays accepted-matching count (kernel rule).
        # Modal unique count lives under search_behavior only.
        selection = {
            "unique_solutions": len(solution_ids),
            "selection_gini": l3["selection_gini"],
            "selection_entropy": l3["selection_entropy"],
            "selection_hhi": l3["selection_hhi"],
            "selection_neff": l3["selection_neff"],
            "n_selection_events": l3["n_selection_events"],
            "top_solutions": l3["top_solutions"],
        }
        collisions = l3["collisions"]
        selection_support = l3["selection_support"]
        search_behavior = {
            "n_modal_attempts": l3["n_selection_events"],
            "unique_solutions": l3["unique_solutions"],
            "hhi": l3["selection_hhi"],
            "entropy": l3["selection_entropy"],
            "neff": l3["selection_neff"],
            "gini": l3["selection_gini"],
            "collisions": l3["collisions"],
            "top_solutions": l3["top_solutions"],
            "collisions_by_tick": l3["collisions_by_tick"],
            "support": l3["selection_support"],
        }
    else:
        evaluations = result.get("evaluations") or []
        attempt_ids = [e["solution_id"] for e in evaluations]
        if attempt_ids:
            selection = frequency_concentration(attempt_ids)
            selection_support = "evaluation_attempts"
        else:
            selection = frequency_concentration(
                [c["solution_id"] for c in couplings]
            )
            selection_support = "accepted_couplings"

    thr_sens = None
    evaluations = result.get("evaluations") or []
    if evaluations and not is_model_selection:
        thr_sens = threshold_sensitivity(evaluations, ceiling=ceiling)

    pct_ceiling = None
    if apply_ceiling and ceiling and not is_model_selection:
        pct_ceiling = round(100.0 * n_couplings / ceiling, 2)

    return {
        "label": condition,
        "seed": manifest["seed"],
        "corpus_id": manifest["corpus_id"],
        "n_ticks": manifest.get("n_ticks"),
        "alignment_tier": result.get("alignment_tier"),
        "n_couplings": n_couplings,
        "pct_ceiling": pct_ceiling,
        "similarity_min": round(min(similarities), 6) if similarities else None,
        "similarity_mean": round(float(np.mean(similarities)), 6) if similarities else None,
        "similarity_max": round(max(similarities), 6) if similarities else None,
        "dispersion": dispersion,
        "concentration_gini": concentration,
        "unique_solutions": len(solution_ids),
        "selection_gini": selection["selection_gini"],
        "selection_entropy": selection["selection_entropy"],
        "selection_hhi": selection.get("selection_hhi"),
        "selection_neff": selection.get("selection_neff"),
        "selection_support": selection_support,
        "n_selection_events": selection["n_selection_events"],
        "top_solutions": selection.get("top_solutions", []),
        "collisions": collisions,
        "threshold_sensitivity": thr_sens,
        "final_matching": final_matching,
        "search_behavior": search_behavior,
        "coupling_pairs": sorted(coupling_pairs(result)),
        "accepted_similarities": similarities,
    }


def expected_jaccard_baseline(
    k1: int,
    k2: int,
    universe: int = SBIR_V2_FEASIBLE_PAIR_UNIVERSE,
) -> dict:
    """Naive model: two independent uniform k-subsets from a fixed pair universe."""
    if universe <= 0 or k1 < 0 or k2 < 0:
        return {
            "expected_intersection": None,
            "expected_jaccard": None,
            "p_disjoint": None,
        }
    expected_intersection = k1 * k2 / universe
    expected_union = k1 + k2 - expected_intersection
    expected_jaccard = (
        expected_intersection / expected_union if expected_union > 0 else None
    )
    # Hypergeometric: draw k2 pairs without overlap with a fixed k1-set.
    from math import comb

    p_disjoint = (
        comb(universe - k1, k2) / comb(universe, k2)
        if k1 + k2 <= universe
        else 0.0
    )
    return {
        "expected_intersection": round(expected_intersection, 6),
        "expected_jaccard": round(expected_jaccard, 6) if expected_jaccard is not None else None,
        "p_disjoint": round(p_disjoint, 6),
    }


def layer1_seed_stability(
    rows: list[dict],
    universe: int = SBIR_V2_FEASIBLE_PAIR_UNIVERSE,
) -> dict:
    pair_sets = {
        row["seed"]: set(tuple(p) for p in row.get("coupling_pairs", []))
        for row in rows
    }
    seeds = sorted(pair_sets)
    overlaps: dict[str, dict] = {}
    for i, sa in enumerate(seeds):
        for sb in seeds[i + 1:]:
            key = f"{sa}_vs_{sb}"
            a_set, b_set = pair_sets[sa], pair_sets[sb]
            intersection = sorted(a_set & b_set)
            union_size = len(a_set | b_set)
            overlaps[key] = {
                "jaccard": round(pairwise_jaccard(a_set, b_set), 4),
                "intersection_size": len(intersection),
                "union_size": union_size,
                "shared_pairs": intersection,
                "baseline": expected_jaccard_baseline(len(a_set), len(b_set), universe),
            }
    return {
        "pairwise_jaccard": {k: v["jaccard"] for k, v in overlaps.items()},
        "pairwise_detail": overlaps,
        "seeds": seeds,
        "feasible_pair_universe": universe,
        "identity_coordinate": "source_solution_id",
        "baseline_warning": "Legacy uniform-pair heuristic does not preserve the cosine gate or injectivity; use l1_l2_identity_null_jaccard.json for the static matching reference. Neither establishes path dependence.",
    }


def _row_from_result_path(
    path: Path,
    *,
    ceiling: int,
    comparison_group: str,
    display_label: str | None = None,
    apply_ceiling: bool = True,
) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    audit_log = None
    condition = result.get("condition") or result.get("run_manifest", {}).get("condition")
    if condition in ("autonomous", "C_autonomous"):
        audit_log = load_audit_log(result, path)
        if audit_log is None:
            LOGGER.warning("No audit log for model-selection run %s; collisions/entropy skipped", path.name)
    row = analyze_coupling_metrics(
        result,
        ceiling=ceiling,
        audit_log=audit_log,
        apply_ceiling=apply_ceiling,
    )
    row["source_file"] = path.name
    row["comparison_group"] = comparison_group
    if display_label:
        row["display_label"] = display_label
    return row


def coupling_summary_report(
    matched_paths: list[Path],
    *,
    extended_paths: list[Path] | None = None,
    model_selection_paths: list[Path] | None = None,
    ceiling: int = SBIR_V2_CEILING,
    simulation_results_dir: Path = SIMULATION_RESULTS_ROOT,
) -> dict:
    per_run = [
        _row_from_result_path(
            path,
            ceiling=ceiling,
            comparison_group="matched_17_tick",
        )
        for path in matched_paths
    ]
    extended_horizon: list[dict] = []
    if extended_paths:
        for path in extended_paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            seed = result["run_manifest"]["seed"]
            n_ticks = result["run_manifest"].get("n_ticks", LAYER1_EXTENDED_TICKS)
            extended_horizon.append(
                _row_from_result_path(
                    path,
                    ceiling=ceiling,
                    comparison_group="extended_horizon",
                    display_label=f"Layer 1 seed {seed} ({n_ticks} ticks — extended horizon)",
                )
            )

    model_selection_rows: list[dict] = []
    if model_selection_paths:
        for path in model_selection_paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            tier = result.get("alignment_tier", "")
            model = _model_letter_from_tier(tier) or "B"
            model_selection_rows.append(
                _row_from_result_path(
                    path,
                    ceiling=ceiling,
                    comparison_group="layer3_autonomous",
                    display_label=(
                        f"L3 Model {model} model selection seed "
                        f"{result['run_manifest']['seed']}"
                    ),
                    apply_ceiling=False,
                )
            )

    layer1_rows = [
        r for r in per_run if r["label"] in ("A_human", "human")
    ]
    stability = (
        layer1_seed_stability([dict(r) for r in layer1_rows])
        if layer1_rows else {}
    )
    all_rows = list(per_run) + list(model_selection_rows)
    standardized = build_standardized_tables(all_rows, ceiling=ceiling)
    validation = validate_l3_reference(model_selection_rows)
    experiment_status = build_experiment_status(simulation_results_dir)
    standardized["D_experiment_status"] = experiment_status
    return {
        "ceiling": ceiling,
        "matched_ticks": LAYER1_MATCHED_TICKS,
        "per_run": per_run,
        "extended_horizon": extended_horizon,
        "layer3_autonomous": model_selection_rows,
        "experiment_status": experiment_status,
        "layer1_seed_stability": stability,
        "standardized_tables": standardized,
        "validation": validation,
        "selection_metric_notes": {
            "L1_L2": "selection_gini/entropy/HHI over solution IDs in evaluation attempts",
            "L3_search_behavior": (
                "modal SELECT frequency including collision skips "
                "(HHI/entropy/Gini/Neff/collisions)"
            ),
            "L3_final_matching": (
                "accepted coupling solution IDs only; injective matchings "
                "have HHI=1/n"
            ),
            "concentration_gini": (
                "spatial Gini on pairwise cosine distances of accepted "
                "solutions (decoupled)"
            ),
            "threshold_sensitivity": (
                "Threshold sensitivity is computed by replaying the recorded "
                "evaluation trace under alternative acceptance thresholds. "
                "At 0.65, the same observed evaluations are rescored against a "
                "stricter bar, so totals match the original 0.60 run. At 0.55, "
                "a satisficing agent would have stopped searching earlier under "
                "the easier bar, so the replay truncates each problem's trace "
                "at its first 0.55-qualifying candidate — producing genuinely "
                "fewer total evaluations, not an inconsistency. "
                f"Levels={list(THRESHOLD_SENSITIVITY_LEVELS)}; one-to-one + energy; "
                "includes acceptance_rate, mean/min/max sim, n_rejected."
            ),
        },
    }


# Locked forensic-audit targets for L3 seed-11 modal SELECT metrics.
L3_VALIDATION_TARGETS = {
    "A": {
        "n_couplings": 89,
        "collisions": 151,
        "hhi": 0.02239583,
        "entropy": 5.959394,
    },
    "B": {
        "n_couplings": 90,
        "collisions": 248,
        "hhi": 0.02515668,
        "entropy": 5.894460,
    },
}


def _model_letter_from_tier(tier: str | None) -> str | None:
    text = tier or ""
    if "tier_3_nemotron_nano_4b" in text:
        return "C"
    if "tier_4_qwen25_7b" in text:
        return "D"
    if "tier_5_gemma2_9b" in text:
        return "E"
    if "tier_1" in text or "lightly" in text:
        return "A"
    if "tier_2" in text or "heavily" in text:
        return "B"
    return None


def _model_letter(row: dict) -> str | None:
    return _model_letter_from_tier(row.get("alignment_tier") or "")


def _architecture_for_label(label: str) -> str:
    if label in ("A_human", "human"):
        return "L1_HumanProxy"
    if label in ("autonomous", "C_autonomous"):
        return "L3_model_selection"
    return "L2_ranked"


def _public_condition_for_label(label: str) -> str:
    if label in ("A_human", "human"):
        return "HumanProxy"
    if label in ("autonomous", "C_autonomous"):
        return "model selection"
    return "ranked"


def validate_l3_reference(model_selection_rows: list[dict]) -> dict:
    """Compare L3 seed-11 rows against locked forensic-audit targets."""
    checks: list[dict] = []
    all_ok = True
    for row in model_selection_rows:
        if int(row.get("seed", -1)) != LAYER3_VALIDATION_SEED:
            continue
        model = _model_letter(row)
        if model not in L3_VALIDATION_TARGETS:
            continue
        target = L3_VALIDATION_TARGETS[model]
        sb = row.get("search_behavior") or {}
        observed = {
            "n_couplings": row["n_couplings"],
            "collisions": sb.get("collisions", row.get("collisions")),
            "hhi": sb.get("hhi", row.get("selection_hhi")),
            "entropy": sb.get("entropy", row.get("selection_entropy")),
        }
        field_ok = {}
        for key, expected in target.items():
            got = observed.get(key)
            if key in ("hhi", "entropy") and got is not None:
                field_ok[key] = abs(float(got) - float(expected)) < 1e-8
            else:
                field_ok[key] = got == expected
            if not field_ok[key]:
                all_ok = False
        checks.append({
            "model": model,
            "label": _run_display_label(row),
            "expected": target,
            "observed": observed,
            "field_ok": field_ok,
            "pass": all(field_ok.values()),
        })
    return {"pass": all_ok, "checks": checks}


def build_standardized_tables(
    rows: list[dict],
    *,
    ceiling: int = SBIR_V2_CEILING,
) -> dict:
    """A Throughput / B Similarity Diagnostics / C Search Behavior / Final Matching."""
    throughput = []
    l3_raw_throughput = []
    similarity_diagnostics = []
    search_behavior = []
    final_matching = []
    threshold_rows = []

    for row in rows:
        label = _run_display_label(row)
        is_l3 = row["label"] in ("autonomous", "C_autonomous")
        throughput_row = {
            "run": label,
            "layer": "L3" if is_l3 else ("L1" if row["label"] in ("A_human", "human") else "L2"),
            "n_couplings": row["n_couplings"],
            "pct_ceiling": row.get("pct_ceiling"),
        }
        if is_l3:
            l3_raw_throughput.append(throughput_row)
        else:
            throughput.append(throughput_row)
        similarity_diagnostics.append({
            "run": label,
            "sim_min": row.get("similarity_min"),
            "sim_mean": row.get("similarity_mean"),
            "sim_max": row.get("similarity_max"),
        })
        fm = row.get("final_matching") or {}
        final_matching.append({
            "run": label,
            "n_couplings": fm.get("n_couplings", row["n_couplings"]),
            "unique_solutions": fm.get("unique_solutions", row.get("unique_solutions")),
            "unique_problems": fm.get("unique_problems"),
            "hhi": fm.get("hhi"),
            "entropy": fm.get("entropy"),
            "neff": fm.get("neff"),
            "gini": fm.get("gini"),
            "support": fm.get("support", "accepted_couplings"),
            "frac_sim_below_0_60": fm.get("frac_sim_below_0_60"),
        })
        if is_l3 and row.get("search_behavior"):
            sb = row["search_behavior"]
            search_behavior.append({
                "run": label,
                "n_modal_attempts": sb.get("n_modal_attempts"),
                "unique_solutions_modal": sb.get("unique_solutions"),
                "hhi": sb.get("hhi"),
                "entropy": sb.get("entropy"),
                "neff": sb.get("neff"),
                "gini": sb.get("gini"),
                "collisions": sb.get("collisions"),
                "top_solutions": sb.get("top_solutions", [])[:10],
                "support": sb.get("support"),
            })
        sens = row.get("threshold_sensitivity")
        if sens:
            for level in THRESHOLD_SENSITIVITY_LEVELS:
                key = f"{level:.2f}"
                cell = sens.get(key, {})
                if isinstance(cell, int):
                    # Backward-compat if older count-only payload appears.
                    cell = {"n_couplings": cell}
                threshold_rows.append({
                    "run": label,
                    "threshold": level,
                    "n_couplings": cell.get("n_couplings"),
                    "pct_ceiling": cell.get("pct_ceiling"),
                    "acceptance_rate": cell.get("acceptance_rate"),
                    "mean_sim": cell.get("mean_sim"),
                    "min_sim": cell.get("min_sim"),
                    "max_sim": cell.get("max_sim"),
                    "n_rejected": cell.get("n_rejected"),
                    "n_evaluated": cell.get("n_evaluated"),
                    "actual_n_couplings": row["n_couplings"],
                })

    return {
        "A_throughput": throughput,
        "L3_raw_throughput": l3_raw_throughput,
        "B_similarity_diagnostics": similarity_diagnostics,
        "C_search_behavior_L3": search_behavior,
        "final_matching": final_matching,
        "threshold_sensitivity": threshold_rows,
        "ceiling": ceiling,
    }


def _strip_heavy_fields(row: dict) -> dict:
    """Drop bulky per-row arrays before writing compact JSON summaries."""
    out = dict(row)
    out.pop("accepted_similarities", None)
    out.pop("coupling_pairs", None)
    sb = out.get("search_behavior")
    if isinstance(sb, dict):
        sb = dict(sb)
        # Keep collisions_by_tick for figure export; drop raw modal list if present.
        sb.pop("modal_selects", None)
        out["search_behavior"] = sb
    return out


def export_figure_data(report: dict, export_dir: Path) -> dict[str, Path]:
    """Write paper-ready CSV/JSON figure tables (not plots)."""
    export_dir.mkdir(parents=True, exist_ok=True)
    rows = list(report.get("per_run", [])) + list(report.get("layer3_autonomous", []))
    written: dict[str, Path] = {}

    # Figure 1: Throughput by layer/model. Incomplete expected cells remain
    # visible as status rows with blank metrics.
    fig1 = []
    for row in rows:
        fig1.append({
            "run": _run_display_label(row),
            "layer": (
                "L3" if row["label"] in ("autonomous", "C_autonomous")
                else ("L1" if row["label"] in ("A_human", "human") else "L2")
            ),
            "architecture": _architecture_for_label(row["label"]),
            "model": _model_letter(row),
            "seed": row["seed"],
            "condition": _public_condition_for_label(row["label"]),
            "metric_support": (
                "raw_l3_couplings"
                if row["label"] in ("autonomous", "C_autonomous")
                else "gated_throughput"
            ),
            "status": "complete",
            "n_couplings": row["n_couplings"],
            "pct_ceiling": row.get("pct_ceiling"),
        })
    for status_row in report.get("experiment_status", []):
        if status_row["status"] == "complete":
            continue
        fig1.append({
            "run": status_row["run"],
            "layer": status_row["layer"],
            "architecture": status_row["architecture"],
            "model": status_row["model"],
            "seed": status_row["seed"],
            "condition": status_row["condition"],
            "status": status_row["status"],
            "n_couplings": None,
            "pct_ceiling": None,
        })
    written["fig1_throughput"] = _write_table_pair(export_dir, "fig1_throughput", fig1)

    # Figure 2: Similarity distributions (long form)
    fig2 = []
    for row in rows:
        for sim in row.get("accepted_similarities") or []:
            fig2.append({
                "run": _run_display_label(row),
                "layer": (
                    "L3" if row["label"] in ("autonomous", "C_autonomous")
                    else ("L1" if row["label"] in ("A_human", "human") else "L2")
                ),
                "architecture": _architecture_for_label(row["label"]),
                "model": _model_letter(row),
                "seed": row["seed"],
                "cosine_similarity": sim,
            })
    written["fig2_similarity_distributions"] = _write_table_pair(
        export_dir, "fig2_similarity_distributions", fig2
    )

    # Figure 3: Collision count by tick (L3)
    fig3 = []
    for row in report.get("layer3_autonomous", []):
        sb = row.get("search_behavior") or {}
        for tick_row in sb.get("collisions_by_tick") or []:
            fig3.append({
                "run": _run_display_label(row),
                "model": _model_letter(row),
                "seed": row["seed"],
                "tick": tick_row["tick"],
                "n_modal_selects": tick_row["n_modal_selects"],
                "n_couplings": tick_row["n_couplings"],
                "n_collisions": tick_row["n_collisions"],
            })
    written["fig3_collision_by_tick"] = _write_table_pair(
        export_dir, "fig3_collision_by_tick", fig3
    )

    # Figure 4: Top attractor frequencies (L3 modal SELECT)
    fig4 = []
    for row in report.get("layer3_autonomous", []):
        sb = row.get("search_behavior") or {}
        for rank, (sid, count) in enumerate(sb.get("top_solutions") or [], start=1):
            fig4.append({
                "run": _run_display_label(row),
                "model": _model_letter(row),
                "seed": row["seed"],
                "rank": rank,
                "solution_id": sid,
                "frequency": count,
                "support": sb.get("support"),
            })
    written["fig4_top_attractors"] = _write_table_pair(
        export_dir, "fig4_top_attractors", fig4
    )

    # Figure 5: Modal concentration (HHI / entropy / Gini)
    fig5 = []
    for row in report.get("layer3_autonomous", []):
        sb = row.get("search_behavior") or {}
        fm = row.get("final_matching") or {}
        fig5.append({
            "run": _run_display_label(row),
            "model": _model_letter(row),
            "seed": row["seed"],
            "metric_support": "modal_SELECT",
            "hhi": sb.get("hhi"),
            "entropy": sb.get("entropy"),
            "gini": sb.get("gini"),
            "neff": sb.get("neff"),
            "collisions": sb.get("collisions"),
            "n_events": sb.get("n_modal_attempts"),
        })
        fig5.append({
            "run": _run_display_label(row),
            "model": _model_letter(row),
            "seed": row["seed"],
            "metric_support": "accepted_matching",
            "hhi": fm.get("hhi"),
            "entropy": fm.get("entropy"),
            "gini": fm.get("gini"),
            "neff": fm.get("neff"),
            "collisions": None,
            "n_events": fm.get("n_couplings"),
        })
    written["fig5_modal_concentration"] = _write_table_pair(
        export_dir, "fig5_modal_concentration", fig5
    )

    # Also dump standardized tables as JSON for machine use.
    std_path = export_dir / "standardized_tables.json"
    std_path.write_text(
        json.dumps(report.get("standardized_tables", {}), indent=2) + "\n",
        encoding="utf-8",
    )
    written["standardized_tables"] = std_path
    status_path = export_dir / "experiment_status.json"
    status_path.write_text(
        json.dumps(report.get("experiment_status", []), indent=2) + "\n",
        encoding="utf-8",
    )
    written["experiment_status"] = status_path
    return written


def _write_table_pair(export_dir: Path, stem: str, rows: list[dict]) -> Path:
    json_path = export_dir / f"{stem}.json"
    csv_path = export_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if rows:
        # Flatten nested values for CSV.
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        lines = [",".join(keys)]
        for row in rows:
            cells = []
            for k in keys:
                val = row.get(k)
                if isinstance(val, (list, dict)):
                    cell = json.dumps(val, separators=(",", ":"))
                elif val is None:
                    cell = ""
                else:
                    cell = str(val)
                if any(c in cell for c in ",\"\n"):
                    cell = '"' + cell.replace('"', '""') + '"'
                cells.append(cell)
            lines.append(",".join(cells))
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path


def write_markdown_report(report: dict, path: Path) -> None:
    """Human-readable coupling / threshold / L3 search report."""
    ceiling = report.get("ceiling", SBIR_V2_CEILING)
    std = report.get("standardized_tables", {})
    lines: list[str] = []
    lines.append("# Coupling summary report")
    lines.append("")
    lines.append(
        f"Matched horizon: **{report.get('matched_ticks', LAYER1_MATCHED_TICKS)}** ticks; "
        f"L1/L2 ceiling = **{ceiling}**."
    )
    lines.append("")
    lines.append(
        "Accepted-matching diversity is reported separately from L3 modal-SELECT "
        "search behavior."
    )
    lines.append("")

    def _md_table(headers: list[str], body_rows: list[list[object]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        for body in body_rows:
            cells = []
            for v in body:
                if v is None:
                    cells.append("n/a")
                elif isinstance(v, float):
                    cells.append(f"{v:.6g}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## A — Throughput")
    _md_table(
        ["Run", "Couplings", "% ceiling"],
        [
            [
                r["run"],
                r["n_couplings"],
                f"{r['pct_ceiling']:.2f}%" if r.get("pct_ceiling") is not None else "n/a",
            ]
            for r in std.get("A_throughput", [])
        ],
    )
    lines.append("## L3 — Raw coupling volume (separate stopping rule)")
    _md_table(
        ["Run", "Raw couplings"],
        [
            [r["run"], r["n_couplings"]]
            for r in std.get("L3_raw_throughput", [])
        ],
    )

    lines.append("## B — Similarity diagnostics")
    _md_table(
        ["Run", "sim min", "sim mean", "sim max"],
        [
            [r["run"], r["sim_min"], r["sim_mean"], r["sim_max"]]
            for r in std.get("B_similarity_diagnostics", [])
        ],
    )

    lines.append("## Experiment status")
    lines.append(
        "Only rows marked **complete** enter the numerical summaries. "
        "Incomplete cells are reported as `in_progress` or `not_run`."
    )
    lines.append("")
    _md_table(
        ["Run", "Layer", "Model", "Seed", "Status", "Source"],
        [
            [
                r["run"],
                r["layer"],
                r.get("model") or "—",
                r["seed"],
                r["status"],
                r.get("source_file") or "—",
            ]
            for r in report.get("experiment_status", [])
        ],
    )

    lines.append("## C — Search behavior (L3 modal SELECT)")
    _md_table(
        ["Run", "N modal", "HHI", "Entropy", "Neff", "Gini", "Collisions"],
        [
            [
                r["run"],
                r["n_modal_attempts"],
                r["hhi"],
                r["entropy"],
                r["neff"],
                r["gini"],
                r["collisions"],
            ]
            for r in std.get("C_search_behavior_L3", [])
        ],
    )

    lines.append("## Final matching (accepted couplings only)")
    _md_table(
        ["Run", "n", "uniq S", "HHI", "Entropy", "Neff", "frac <0.60"],
        [
            [
                r["run"],
                r["n_couplings"],
                r["unique_solutions"],
                r["hhi"],
                r["entropy"],
                r["neff"],
                r.get("frac_sim_below_0_60"),
            ]
            for r in std.get("final_matching", [])
        ],
    )

    lines.append("## Threshold sensitivity (L1 + L2 replay)")
    lines.append(
        "One-to-one + energy replay of logged evaluations at true cosine thresholds."
    )
    lines.append("")
    _md_table(
        [
            "Run", "thr", "couplings", "%ceil", "accept rate",
            "mean sim", "min", "max", "n rejected",
        ],
        [
            [
                r["run"],
                r["threshold"],
                r["n_couplings"],
                r["pct_ceiling"],
                r["acceptance_rate"],
                r["mean_sim"],
                r["min_sim"],
                r["max_sim"],
                r["n_rejected"],
            ]
            for r in std.get("threshold_sensitivity", [])
        ],
    )

    val = report.get("validation") or {}
    lines.append("## L3 validation (locked forensic targets)")
    if not val.get("checks"):
        lines.append("No L3 model-selection rows present.")
        lines.append("")
    else:
        status = "PASS" if val.get("pass") else "FAIL"
        lines.append(f"Overall: **{status}**")
        lines.append("")
        for check in val["checks"]:
            lines.append(
                f"- Model {check['model']}: "
                f"{'PASS' if check['pass'] else 'FAIL'} "
                f"(observed={check['observed']}, expected={check['expected']})"
            )
        lines.append("")

    notes = report.get("selection_metric_notes") or {}
    if notes:
        lines.append("## Metric notes")
        for key, val_s in notes.items():
            lines.append(f"- **{key}**: {val_s}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_display_label(row: dict) -> str:
    if row.get("display_label"):
        return row["display_label"]
    ticks = row.get("n_ticks")
    tick_s = f" ({ticks} ticks)" if ticks is not None else ""
    label = row["label"]
    tier = row.get("alignment_tier") or ""
    if label in ("A_human", "human"):
        return f"L1 seed {row['seed']}{tick_s}"
    if label == "ranked":
        model = _model_letter_from_tier(tier) or "B"
        return f"L2 Model {model} ranked seed {row['seed']}{tick_s}"
    if label in ("autonomous", "C_autonomous"):
        model = _model_letter_from_tier(tier) or "B"
        return f"L3 Model {model} model selection seed {row['seed']}{tick_s}"
    return f"L2 seed {row['seed']}{tick_s}"


def _print_coupling_rows(rows: list[dict], header: str, *, ceiling: int) -> None:
    print(f"\n=== {header} (ceiling={ceiling}) ===")
    col_header = (
        f"{'run':<42s} {'couplings':>9s} {'%ceil':>7s} "
        f"{'sim_min':>8s} {'sim_mean':>8s} {'sim_max':>8s} "
        f"{'dispersion':>10s} {'conc_gini':>10s} "
        f"{'uniq':>5s} {'sel_gini':>8s} {'H_bits':>7s}"
    )
    print(col_header)
    print("-" * len(col_header))
    for row in rows:
        label = _run_display_label(row)
        conc = row["concentration_gini"]
        conc_s = f"{conc:.4f}" if conc is not None else "n/a"
        pct = row.get("pct_ceiling")
        pct_s = f"{pct:6.1f}%" if pct is not None else "   n/a"
        sel_g = row.get("selection_gini")
        sel_h = row.get("selection_entropy")
        uniq = row.get("unique_solutions")
        print(
            f"{label:<42s} {row['n_couplings']:>9d} {pct_s:>7s} "
            f"{row['similarity_min']:>8.4f} {row['similarity_mean']:>8.4f} "
            f"{row['similarity_max']:>8.4f} {row['dispersion']:>10.6f} {conc_s:>10s} "
            f"{uniq if uniq is not None else 0:>5d} "
            f"{sel_g if sel_g is not None else float('nan'):>8.4f} "
            f"{sel_h if sel_h is not None else float('nan'):>7.4f}"
        )


def _print_threshold_sensitivity(rows: list[dict]) -> None:
    sens_rows = [r for r in rows if r.get("threshold_sensitivity")]
    if not sens_rows:
        return
    print("\n=== Threshold sensitivity (eval replay; one-to-one + energy) ===")
    print(
        f"{'run':<36s} {'thr':>5s} {'n':>4s} {'%ceil':>7s} "
        f"{'accept':>8s} {'mean':>8s} {'min':>8s} {'max':>8s} {'rej':>5s}"
    )
    print("-" * 100)
    for row in sens_rows:
        sens = row["threshold_sensitivity"]
        label = _run_display_label(row)
        for level in THRESHOLD_SENSITIVITY_LEVELS:
            cell = sens.get(f"{level:.2f}", {})
            if isinstance(cell, int):
                cell = {"n_couplings": cell}
            pct = cell.get("pct_ceiling")
            pct_s = f"{pct:6.2f}%" if pct is not None else "   n/a"
            ar = cell.get("acceptance_rate")
            ar_s = f"{ar:.4f}" if ar is not None else "n/a"
            mean_s = f"{cell['mean_sim']:.4f}" if cell.get("mean_sim") is not None else "n/a"
            min_s = f"{cell['min_sim']:.4f}" if cell.get("min_sim") is not None else "n/a"
            max_s = f"{cell['max_sim']:.4f}" if cell.get("max_sim") is not None else "n/a"
            print(
                f"{label:<36s} {level:>5.2f} {cell.get('n_couplings', 0):>4d} {pct_s:>7s} "
                f"{ar_s:>8s} {mean_s:>8s} {min_s:>8s} {max_s:>8s} "
                f"{cell.get('n_rejected', 0):>5d}"
            )
    print(
        "Threshold sensitivity is computed by replaying the recorded evaluation "
        "trace under alternative acceptance thresholds. At 0.65, the same "
        "observed evaluations are rescored against a stricter bar, so totals "
        "match the original 0.60 run. At 0.55, a satisficing agent would have "
        "stopped searching earlier under the easier bar, so the replay "
        "truncates each problem's trace at its first 0.55-qualifying candidate "
        "— producing genuinely fewer total evaluations, not an inconsistency. "
        "Raising the threshold may still undercount couplings vs a fresh run "
        "(original logs stop after the first accept at 0.60)."
    )


def _print_master_tables(report: dict) -> None:
    std = report.get("standardized_tables") or {}
    print("\n=== A Throughput ===")
    print(f"{'run':<42s} {'couplings':>9s} {'%ceil':>7s}")
    print("-" * 60)
    for row in std.get("A_throughput", []):
        pct = row.get("pct_ceiling")
        pct_s = f"{pct:6.1f}%" if pct is not None else "   n/a"
        print(f"{row['run']:<42s} {row['n_couplings']:>9d} {pct_s:>7s}")

    print("\n=== L3 Raw Coupling Volume (separate stopping rule) ===")
    print(f"{'run':<42s} {'raw couplings':>13s}")
    print("-" * 60)
    for row in std.get("L3_raw_throughput", []):
        print(f"{row['run']:<42s} {row['n_couplings']:>13d}")

    print("\n=== B Similarity Diagnostics ===")
    print(f"{'run':<42s} {'sim_min':>8s} {'sim_mean':>8s} {'sim_max':>8s}")
    print("-" * 70)
    for row in std.get("B_similarity_diagnostics", []):
        print(
            f"{row['run']:<42s} {row['sim_min']:>8.4f} "
            f"{row['sim_mean']:>8.4f} {row['sim_max']:>8.4f}"
        )

    print("\n=== C Search Behavior (L3 modal SELECT) ===")
    print(
        f"{'run':<42s} {'Nmod':>5s} {'HHI':>10s} {'H_bits':>8s} "
        f"{'Neff':>7s} {'Gini':>7s} {'coll':>6s}"
    )
    print("-" * 95)
    for row in std.get("C_search_behavior_L3", []):
        print(
            f"{row['run']:<42s} {row['n_modal_attempts']:>5d} "
            f"{row['hhi']:>10.8f} {row['entropy']:>8.6f} "
            f"{row['neff']:>7.2f} {row['gini']:>7.4f} {row['collisions']:>6d}"
        )

    print("\n=== Final Matching (accepted couplings) ===")
    print(
        f"{'run':<42s} {'n':>4s} {'uniq':>5s} {'HHI':>10s} "
        f"{'H_bits':>8s} {'Neff':>7s}"
    )
    print("-" * 85)
    for row in std.get("final_matching", []):
        print(
            f"{row['run']:<42s} {row['n_couplings']:>4d} "
            f"{row['unique_solutions']:>5d} "
            f"{row['hhi'] if row['hhi'] is not None else float('nan'):>10.8f} "
            f"{row['entropy'] if row['entropy'] is not None else float('nan'):>8.6f} "
            f"{row['neff'] if row['neff'] is not None else float('nan'):>7.2f}"
        )

    val = report.get("validation") or {}
    if val.get("checks"):
        print("\n=== L3 validation ===")
        print(f"Overall: {'PASS' if val.get('pass') else 'FAIL'}")
        for check in val["checks"]:
            print(
                f"  Model {check['model']}: "
                f"{'PASS' if check['pass'] else 'FAIL'} "
                f"obs={check['observed']}"
            )


def print_coupling_summary_table(report: dict) -> None:
    ceiling = report.get("ceiling", SBIR_V2_CEILING)
    matched_ticks = report.get("matched_ticks", LAYER1_MATCHED_TICKS)
    _print_coupling_rows(
        report["per_run"],
        f"Matched-tick comparison (n_ticks={matched_ticks})",
        ceiling=ceiling,
    )
    if report.get("extended_horizon"):
        _print_coupling_rows(
            report["extended_horizon"],
            "Layer 1 extended horizon (reference only)",
            ceiling=ceiling,
        )
    if report.get("layer3_autonomous"):
        _print_coupling_rows(
            report["layer3_autonomous"],
            "Layer 3 model selection (no % ceiling)",
            ceiling=ceiling,
        )

    _print_threshold_sensitivity(report.get("per_run", []))
    _print_master_tables(report)

    stab = report.get("layer1_seed_stability", {})
    if stab.get("pairwise_jaccard"):
        print("\n=== Layer 1 seed stability (coupling-pair Jaccard, matched ticks) ===")
        for key, val in stab["pairwise_jaccard"].items():
            detail = stab.get("pairwise_detail", {}).get(key, {})
            shared = detail.get("shared_pairs", [])
            baseline = detail.get("baseline", {})
            shared_s = f" shared={shared}" if shared else ""
            exp_j = baseline.get("expected_jaccard")
            p0 = baseline.get("p_disjoint")
            print(
                f"  {key}: {val:.4f} "
                f"(intersection={detail.get('intersection_size', 0)}, "
                f"union={detail.get('union_size', 0)}{shared_s})"
            )
            if exp_j is not None and p0 is not None:
                print(
                    f"    baseline: E[Jaccard]={exp_j:.6f}, P(disjoint)={p0:.4f}"
                )

    notes = report.get("selection_metric_notes")
    if notes:
        print("\n=== Selection-metric notes ===")
        for key, val in notes.items():
            print(f"  {key}: {val}")

def aggregate(per_run: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in per_run:
        groups[(run["condition"], run["alignment_tier"])].append(run)

    summary = {}
    for (condition, tier), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        dns = [r["novelty_decay_rate"] for r in runs if r["novelty_decay_rate"] is not None]
        lo, hi = ci95(dns)
        key = condition + (f"/{tier}" if tier else "")
        summary[key] = {
            "n_runs": len(runs),
            "seeds": sorted({r["seed"] for r in runs}),
            "corpora": sorted({r["corpus_id"] for r in runs}),
            "embedding_models": sorted({r["embedding_model_id"] for r in runs}),
            "mean_novelty_decay_rate": round(float(np.mean(dns)), 6) if dns else None,
            "novelty_decay_ci95": [round(lo, 6), round(hi, 6)] if lo is not None else None,
            "mean_n_couplings": round(float(np.mean([r["n_couplings"] for r in runs])), 2),
            "mean_modal_agreement": round(
                float(np.mean([r["mean_modal_agreement"] for r in runs
                               if r["mean_modal_agreement"] is not None])), 4
            ) if any(r["mean_modal_agreement"] is not None for r in runs) else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze variance across all runs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path,
                        default=RESULTS_ROOT / "variance_analysis.json")
    parser.add_argument(
        "--coupling-summary",
        action="store_true",
        help="Post-hoc coupling metrics over simulation_results JSON files.",
    )
    parser.add_argument(
        "--simulation-results-dir",
        type=Path,
        default=SIMULATION_RESULTS_ROOT,
        help="Directory with layer1_seed*.json / layer2_*.json results.",
    )
    parser.add_argument(
        "--coupling-summary-output",
        type=Path,
        default=SIMULATION_RESULTS_ROOT / "coupling_summary.json",
    )
    parser.add_argument(
        "--analysis-export-dir",
        type=Path,
        default=ANALYSIS_EXPORTS_ROOT,
        help="Commit-safe dir for markdown report + figure CSV/JSON exports.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
        help="Markdown report path (default: <analysis-export-dir>/coupling_summary_report.md).",
    )
    parser.add_argument(
        "--ceiling",
        type=int,
        default=None,
        help="Matching ceiling for %% throughput (default: compute from corpus/seed).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Limit coupling-summary to these seeds (default: all LAYER1_SEEDS).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.coupling_summary:
        sim_dir = args.simulation_results_dir
        seeds = tuple(args.seeds) if args.seeds else LAYER1_SEEDS
        matched_paths = [
            sim_dir / f"layer1_seed{seed}.json"
            for seed in seeds
        ]
        # Include only complete results in numerical summaries. Expected but
        # incomplete cells remain in experiment_status and figure 1 exports.
        for seed in seeds:
            for _, tier, _ in COMPARISON_MODELS:
                stem = f"layer2_{tier}_ranked_seed{seed}.json"
                path = sim_dir / stem
                if _is_complete_result_path(path):
                    matched_paths.append(path)
        missing = [p for p in matched_paths if not p.exists()]
        if missing:
            sys.exit(
                "ERROR: missing simulation result files:\n"
                + "\n".join(f"  {p}" for p in missing)
            )
        extended_paths = [
            sim_dir / f"layer1_seed{seed}_{LAYER1_EXTENDED_TICKS}tick_backup.json"
            for seed in seeds
        ]
        extended_paths = [p for p in extended_paths if p.exists()]
        model_selection_paths = []
        l3_seeds = seeds if args.seeds else LAYER3_MODEL_A_SEEDS
        # Full L3 model-selection grid across Models A–E.
        for seed in l3_seeds:
            for _, tier, _ in COMPARISON_MODELS:
                stem = f"layer2_{tier}_autonomous_seed{seed}.json"
                path = sim_dir / stem
                if _is_complete_result_path(path):
                    model_selection_paths.append(path)
        ceiling = args.ceiling
        if ceiling is None:
            ref = json.loads(matched_paths[0].read_text(encoding="utf-8"))
            manifest = ref["run_manifest"]
            ceiling = compute_matching_ceiling(
                manifest["corpus_id"],
                manifest["embedding_model"]["model_id"],
                manifest["seed"],
            )
            LOGGER.info("Computed matching ceiling: %d", ceiling)
        report = coupling_summary_report(
            matched_paths,
            extended_paths=extended_paths or None,
            model_selection_paths=model_selection_paths or None,
            ceiling=ceiling,
            simulation_results_dir=sim_dir,
        )

        export_dir = args.analysis_export_dir
        written = export_figure_data(report, export_dir)
        md_path = args.markdown_report or (export_dir / "coupling_summary_report.md")
        write_markdown_report(report, md_path)
        LOGGER.info("Wrote markdown report: %s", md_path)
        for key, path in written.items():
            LOGGER.info("Wrote figure/table export %s: %s", key, path)

        # Compact JSON: drop bulky similarity vectors / pair lists.
        compact = dict(report)
        compact["per_run"] = [_strip_heavy_fields(r) for r in report.get("per_run", [])]
        compact["extended_horizon"] = [
            _strip_heavy_fields(r) for r in report.get("extended_horizon", [])
        ]
        compact["layer3_autonomous"] = [
            _strip_heavy_fields(r) for r in report.get("layer3_autonomous", [])
        ]
        args.coupling_summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.coupling_summary_output.write_text(
            json.dumps(compact, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Wrote coupling summary: %s", args.coupling_summary_output)

        export_json = export_dir / "coupling_summary.json"
        export_json.write_text(
            json.dumps(compact, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Wrote commit-safe coupling summary: %s", export_json)

        val = report.get("validation") or {}
        if val.get("checks") and not val.get("pass"):
            LOGGER.error(
                "L3 validation FAILED vs locked forensic targets — "
                "do not modify paper claims until resolved."
            )
            for check in val["checks"]:
                if not check["pass"]:
                    LOGGER.error(
                        "Model %s mismatch: observed=%s expected=%s fields=%s",
                        check["model"],
                        check["observed"],
                        check["expected"],
                        check["field_ok"],
                    )

        print_coupling_summary_table(report)
        return

    result_files = [
        path for path in sorted(args.results_dir.glob("run_*/[ABC]_*.json"))
    ]
    if not result_files:
        sys.exit(f"ERROR: no run results found under {args.results_dir}. "
                 "Run Layer 1 / Layer 2 first.")

    per_run = []
    for path in result_files:
        result = json.loads(path.read_text(encoding="utf-8"))
        per_run.append(analyze_run(result))
        LOGGER.info("Analyzed %s", path.relative_to(PROJECT_ROOT))

    report = {
        "n_runs_analyzed": len(per_run),
        "per_run": per_run,
        "grid_summary": aggregate(per_run),
        "alignment_gradient": alignment_gradient(per_run),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote report: %s", args.output)

    print("\n=== Legacy exploratory variance diagnostic (dN; negative = collapse) ===")
    for key, row in report["grid_summary"].items():
        ci = row["novelty_decay_ci95"]
        print(f"  {key:40s} n={row['n_runs']:<3d} "
              f"dN={row['mean_novelty_decay_rate']} "
              f"CI95={ci if ci else 'n/a (need >=2 runs)'} "
              f"couplings/run={row['mean_n_couplings']}")

    print("\n=== Legacy alignment-gradient diagnostic (not used) ===")
    ag = report["alignment_gradient"]
    if ag["status"] != "ok":
        print(f"  {ag['status']}: {ag['detail']}")
    else:
        print(f"  slope={ag['slope']}  CI95={ag['slope_ci95']}  "
              f"p={ag['p_value']}  R^2={ag['r_squared']}  (n={ag['n_runs']})")
        print(f"  tier means: {ag['tier_means']}")
        for flag in ag["flags"]:
            print(f"  ⚠ {flag}")
        if not ag["flags"]:
            print("  Monotonic and significant — consistent with the "
                  "variance-collapse hypothesis on this grid.")


if __name__ == "__main__":
    main()
