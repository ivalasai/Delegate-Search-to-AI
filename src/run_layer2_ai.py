#!/usr/bin/env python3
"""L2/L3 model-mediated coupling, driven by real Ollama model calls.

Two delegation architectures are supported:
  - ranked (L2): model ranks solutions; RuleProxy accepts from that ranking.
  - model selection (L3): model directly selects pairings; no RuleProxy gate,
    threshold, or energy constraint is applied. The stored condition label
    remains ``autonomous`` for compatibility with existing artifacts.

Every tick decision is repeated three times per active problem, all prompts/
responses are logged, and the modal parsed output is used for the final action.
Each model call covers exactly one problem and its candidate solution list.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from model_backend import call_model, load_model_config
from sim_common import (
    LOCAL_SEARCH_N,
    PROJECT_ROOT,
    SATISFICING_THRESHOLD,
    USE_PERCEPTUAL_NOISE,
    SIMULATION_RESULTS_ROOT,
    RuleProxy,
    annotate_coupling,
    build_arrival_schedule,
    build_origin_pair_exclusions,
    compute_same_origin_reunion_rate,
    cosine_similarity,
    decouple_solution_pool,
    is_blocked_origin_match,
    load_embeddings,
    make_run_manifest,
    similarity_distribution_diagnostic,
    solutions_visible_for_tick,
    summarize_acceptance_counts,
    trace_variance,
)

LOGGER = logging.getLogger("run_layer2_ai")

SOLUTION_SUMMARY_MAX_WORDS = 25

RANK_LINE_RE = re.compile(r"RANK\s+P(?P<pid>\w+)\s*:\s*(?P<ranking>.+)", re.IGNORECASE)
SELECT_LINE_RE = re.compile(r"SELECT\s+P(?P<pid>\w+)\s*-\s*S(?P<sid>\w+)", re.IGNORECASE)
# Non-compliant but recognizable variant (solution listed first). Rejected per
# the v1 parsing contract, but recorded as a parse issue for auditability.
SELECT_REVERSED_RE = re.compile(r"SELECT\s+S(?P<sid>\w+)\s*-\s*P(?P<pid>\w+)", re.IGNORECASE)
SOLUTION_ID_RE = re.compile(r"S(\w+)")


def load_prompts(prompt_version: str) -> dict[str, str]:
    """Extract the four prompt texts from the versioned prompt markdown file.

    The file's first four fenced code blocks are, in order: ranked system,
    ranked user, model-selection system, model-selection user. The returned
    dictionary retains legacy prompt keys for artifact compatibility.
    """
    path = PROJECT_ROOT / "prompts" / f"{prompt_version}.md"
    if not path.exists():
        sys.exit(f"ERROR: prompt file not found: {path}")
    blocks = re.findall(r"```\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    if len(blocks) < 4:
        sys.exit(f"ERROR: expected >=4 fenced blocks in {path}, found {len(blocks)}")
    return {
        "ranked_system": blocks[0].strip(),
        "ranked_user": blocks[1].strip(),
        "autonomous_system": blocks[2].strip(),
        "autonomous_user": blocks[3].strip(),
    }


def render_items(item_ids: list[str], items: dict[str, Any]) -> str:
    return "\n".join(f"{item_id}: {items[item_id]['text']}" for item_id in item_ids)


def truncate_words(text: str, max_words: int = SOLUTION_SUMMARY_MAX_WORDS) -> str:
    """Mechanically truncate item text to the first *max_words* words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def render_single_problem_candidates(
    problem_id: str,
    items: dict[str, Any],
    allowed_solution_ids: list[str],
) -> str:
    lines = [f"{problem_id}: {items[problem_id]['text']}", "CANDIDATE SOLUTIONS:"]
    for solution_id in allowed_solution_ids:
        summary = truncate_words(items[solution_id]["text"])
        lines.append(f"{solution_id}: {summary}")
    return "\n".join(lines)


def build_user_prompt_for_problem(
    *,
    condition: str,
    problem_id: str,
    items: dict[str, Any],
    allowed_solution_ids: list[str],
) -> str:
    problem_block = render_single_problem_candidates(
        problem_id, items, allowed_solution_ids
    )
    user_prompt = (
        "For this problem, only use the candidate solutions listed below.\n\n"
        "PROBLEM AND CANDIDATE SOLUTIONS:\n"
        f"{problem_block}\n\n"
    )
    if condition == "ranked":
        user_prompt += (
            f"Respond with exactly one line in the format: "
            f"RANK {problem_id}: S<id>, S<id>, ... "
            "Do not include any other text."
        )
    else:
        user_prompt += (
            f"Respond with exactly one line in the format: "
            f"SELECT {problem_id}-S<id> "
            "Do not include any other text."
        )
    return user_prompt


def render_batched_problem_candidates(
    problem_ids: list[str],
    items: dict[str, Any],
    allowed_by_problem: dict[str, list[str]],
) -> str:
    blocks = [
        render_single_problem_candidates(problem_id, items, allowed_by_problem[problem_id])
        for problem_id in problem_ids
    ]
    return "\n\n".join(blocks)


def build_user_prompt_batched(
    *,
    condition: str,
    problem_ids: list[str],
    items: dict[str, Any],
    allowed_by_problem: dict[str, list[str]],
) -> str:
    problem_blocks = render_batched_problem_candidates(
        problem_ids, items, allowed_by_problem
    )
    user_prompt = (
        "For each problem below, only use the candidate solutions listed "
        "directly under that problem.\n\n"
        "PROBLEMS AND CANDIDATE SOLUTIONS:\n"
        f"{problem_blocks}\n\n"
    )
    if condition == "ranked":
        user_prompt += (
            "Respond with exactly one line per problem in the format: "
            "RANK P<id>: S<id>, S<id>, ... "
            "Do not include any other text."
        )
    else:
        user_prompt += (
            "Respond with exactly one line per pairing in the format: "
            "SELECT P<id>-S<id> "
            "Do not include any other text."
        )
    return user_prompt


def chunk_problems(problem_ids: list[str], problems_per_call: int) -> list[list[str]]:
    if problems_per_call <= 0:
        return [problem_ids]
    return [
        problem_ids[i : i + problems_per_call]
        for i in range(0, len(problem_ids), problems_per_call)
    ]


# Lenient variants: same structure, keyword omitted. Content is still purely
# model-stated; leniency only affects format, and every lenient match is
# flagged so format compliance stays measurable per tier.
RANK_LENIENT_RE = re.compile(r"^\s*P(?P<pid>\w+)\s*:\s*(?P<ranking>.+)$", re.IGNORECASE)
SELECT_LENIENT_RE = re.compile(r"^\s*P(?P<pid>\w+)\s*-\s*S(?P<sid>\w+)\s*$", re.IGNORECASE)


def canonicalize_item_id(prefix: str, raw: str, known_ids: set[str]) -> str:
    """Map a model-returned ID body onto a known pool ID.

    Models often drop leading zeros (S11 vs S011). Prefer an exact match, then
    zero-pad numeric bodies to widths observed in the known pool.
    """
    body = raw.lstrip(prefix + prefix.lower())
    candidate = prefix + body
    if candidate in known_ids:
        return candidate
    if body.isdigit():
        widths = {
            len(item_id) - len(prefix)
            for item_id in known_ids
            if item_id.startswith(prefix) and item_id[len(prefix) :].isdigit()
        }
        for width in sorted(widths):
            padded = prefix + body.zfill(width)
            if padded in known_ids:
                return padded
    return candidate


def parse_rankings(
    text: str, problem_ids: set[str], solution_ids: set[str]
) -> tuple[dict[str, list[str]], list[str]]:
    """Extract per-problem solution rankings; return (rankings, issues)."""
    rankings: dict[str, list[str]] = {}
    issues: list[str] = []

    def ingest(pid_raw: str, ranking_text: str, lenient: bool) -> None:
        pid = canonicalize_item_id("P", pid_raw, problem_ids)
        if pid not in problem_ids:
            issues.append(f"invalid_reference:{pid}")
            return
        seen: list[str] = []
        for sid_raw in SOLUTION_ID_RE.findall(ranking_text):
            sid = canonicalize_item_id("S", sid_raw, solution_ids)
            if sid in solution_ids and sid not in seen:
                seen.append(sid)
            elif sid not in solution_ids:
                issues.append(f"invalid_reference:{sid}")
        if seen and pid not in rankings:
            if lenient:
                issues.append(f"format_deviation:missing_RANK_keyword:{pid}")
            rankings[pid] = seen

    for line in text.splitlines():
        strict = RANK_LINE_RE.search(line)
        if strict:
            ingest(strict.group("pid"), strict.group("ranking"), lenient=False)
            continue
        lenient = RANK_LENIENT_RE.match(line)
        if lenient:
            ingest(lenient.group("pid"), lenient.group("ranking"), lenient=True)
    return rankings, issues


def parse_selections(
    text: str, problem_ids: set[str], solution_ids: set[str]
) -> tuple[dict[str, str], list[str]]:
    """Extract problem->solution selections; return (selections, issues)."""
    selections: dict[str, str] = {}
    issues: list[str] = []

    def ingest(pid_raw: str, sid_raw: str, lenient: bool) -> None:
        pid = canonicalize_item_id("P", pid_raw, problem_ids)
        sid = canonicalize_item_id("S", sid_raw, solution_ids)
        if pid not in problem_ids or sid not in solution_ids:
            issues.append(f"invalid_reference:{pid}-{sid}")
            return
        if pid not in selections:
            if lenient:
                issues.append(f"format_deviation:missing_SELECT_keyword:{pid}")
            selections[pid] = sid

    for line in text.splitlines():
        reversed_match = SELECT_REVERSED_RE.search(line)
        if reversed_match:
            issues.append(
                f"reversed_format:S{reversed_match.group('sid')}-P{reversed_match.group('pid')}"
            )
            continue
        strict = SELECT_LINE_RE.search(line)
        if strict:
            ingest(strict.group("pid"), strict.group("sid"), lenient=False)
            continue
        lenient = SELECT_LENIENT_RE.match(line)
        if lenient:
            ingest(lenient.group("pid"), lenient.group("sid"), lenient=True)
    return selections, issues


def modal(values: list[Any]) -> tuple[Any, float]:
    """Return (modal value, agreement fraction). Ties break by first occurrence."""
    counts = Counter(json.dumps(v, sort_keys=True) for v in values)
    top_serialized, top_count = counts.most_common(1)[0]
    return json.loads(top_serialized), top_count / len(values)


class AuditLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.records: list[dict[str, Any]] = []

    def extend(self, records: list[dict[str, Any]]) -> None:
        self.records.extend(records)

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")

    def close(self) -> None:
        self.flush()


def checkpoint_path(base_name: str) -> Path:
    return SIMULATION_RESULTS_ROOT / "checkpoints" / f"{base_name}_checkpoint.json"


def result_path_for(base_name: str) -> Path:
    return SIMULATION_RESULTS_ROOT / f"{base_name}.json"


def save_checkpoint(
    path: Path,
    *,
    base_name: str,
    manifest: dict[str, Any],
    last_completed_tick: int,
    n_ticks: int,
    coupled_problems: set[str],
    coupled_solutions: set[str],
    couplings: list[dict[str, Any]],
    tick_diagnostics: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    proxy_energy: dict[str, float],
    audit_path: Path,
    extras: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_name": base_name,
        "status": "in_progress",
        "last_completed_tick": last_completed_tick,
        "n_ticks": n_ticks,
        "run_manifest": manifest,
        "coupled_problems": sorted(coupled_problems),
        "coupled_solutions": sorted(coupled_solutions),
        "couplings": couplings,
        "tick_diagnostics": tick_diagnostics,
        "evaluations": evaluations,
        "proxy_energy": proxy_energy,
        "audit_log": str(audit_path.relative_to(PROJECT_ROOT)),
        **extras,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_partial_result(base_name: str, payload: dict[str, Any]) -> Path:
    out_path = result_path_for(base_name)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def coupled_variance_at_tick(couplings: list[dict[str, Any]], items: dict[str, Any], tick_cutoff: int) -> float:
    vectors: list[list[float]] = []
    for coupling in couplings:
        if coupling["tick"] > tick_cutoff:
            continue
        vectors.append(items[coupling["problem_id"]]["vector"])
        vectors.append(items[coupling["solution_id"]]["vector"])
    return round(trace_variance(vectors), 6)


def write_smoke_result(name: str, result: dict[str, Any]) -> Path:
    SIMULATION_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = result_path_for(name)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return out_path


def build_result_payload(
    *,
    manifest: dict[str, Any],
    condition: str,
    tier: str,
    similarity_diagnostic: dict[str, Any],
    solution_decoupling: dict[str, str],
    origin_pair_exclusions: list[dict[str, str]],
    satisficing_threshold: float,
    use_perceptual_noise: bool,
    arrivals: dict[str, int],
    couplings: list[dict[str, Any]],
    tick_diagnostics: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    n_ticks: int,
    items: dict[str, Any],
    problem_ids: list[str],
    solution_ids: list[str],
    coupled_problems: set[str],
    coupled_solutions: set[str],
    audit_path: Path,
    status: str,
    last_completed_tick: int,
) -> dict[str, Any]:
    variance_tick_0 = coupled_variance_at_tick(couplings, items, tick_cutoff=0)
    variance_tick_final = coupled_variance_at_tick(couplings, items, tick_cutoff=n_ticks)
    payload = {
        "status": status,
        "last_completed_tick": last_completed_tick,
        "run_manifest": manifest,
        "condition": condition,
        "alignment_tier": tier,
        "similarity_diagnostic": similarity_diagnostic,
        "solution_decoupling": solution_decoupling,
        "origin_pair_exclusions": origin_pair_exclusions,
        "same_origin_reunion_rate": compute_same_origin_reunion_rate(
            couplings, solution_decoupling
        ),
        "satisficing_threshold": satisficing_threshold if condition == "ranked" else None,
        "use_perceptual_noise": use_perceptual_noise if condition == "ranked" else None,
        "arrival_schedule": arrivals,
        "couplings": couplings,
        "tick_diagnostics": tick_diagnostics,
        "n_evaluations": len(evaluations),
        "evaluations": evaluations if condition == "ranked" else None,
        "variance": {
            "tick_0": variance_tick_0,
            f"tick_{n_ticks}": variance_tick_final,
        },
        "uncoupled_problem_ids_final": sorted(set(problem_ids) - coupled_problems),
        "uncoupled_solution_ids_final": sorted(set(solution_ids) - coupled_solutions),
        "audit_log": str(audit_path.relative_to(PROJECT_ROOT)),
    }
    if condition == "ranked":
        payload["acceptance_summary"] = summarize_acceptance_counts(couplings)
    return payload


def run_condition(
    *,
    condition: str,  # "ranked" or legacy L3 label "autonomous"
    seed: int,
    corpus_id: str,
    embedding_model_key: str,
    n_ticks: int,
    n_repeats: int,
    model_config_path: str,
    prompt_version: str,
    satisficing_threshold: float = SATISFICING_THRESHOLD,
    resume: bool = False,
    problems_per_call: int = 1,
    use_perceptual_noise: bool = USE_PERCEPTUAL_NOISE,
) -> tuple[Path, Path]:
    config = load_model_config(model_config_path)
    tier = config["alignment_tier"]
    prompts = load_prompts(prompt_version)
    embeddings = load_embeddings(corpus_id, embedding_model_key)
    items, solution_decoupling = decouple_solution_pool(embeddings["items"], seed)
    origin_pair_exclusions, blocked_map = build_origin_pair_exclusions(solution_decoupling)
    similarity_diagnostic = similarity_distribution_diagnostic(items)
    problem_ids = sorted(i for i, item in items.items() if item["role"] == "problem")
    solution_ids = sorted(i for i, item in items.items() if item["role"] == "solution")

    base_name = f"layer2_{tier}_{condition}_seed{seed}"
    if problems_per_call != 1:
        batch_label = "all" if problems_per_call <= 0 else str(problems_per_call)
        base_name += f"_batched_{batch_label}"
    ckpt_path = checkpoint_path(base_name)
    audit_path = SIMULATION_RESULTS_ROOT / f"{base_name}_log.json"
    audit = AuditLog(audit_path)

    manifest = make_run_manifest(
        seed=seed,
        corpus_id=corpus_id,
        embeddings=embeddings,
        condition=condition,
        n_ticks=n_ticks,
        model_config_ref=str(model_config_path),
        model_config_snapshot=config,
        prompt_version=prompt_version,
        n_repeats_per_tick=n_repeats,
    )

    rng = random.Random(seed)
    arrivals = build_arrival_schedule(problem_ids + solution_ids, n_ticks, rng)
    proxy = RuleProxy(
        rng,
        threshold=satisficing_threshold,
        use_perceptual_noise=use_perceptual_noise,
    )

    coupled_problems: set[str] = set()
    coupled_solutions: set[str] = set()
    couplings: list[dict[str, Any]] = []
    tick_diagnostics: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    start_tick = 0

    if resume:
        saved = load_checkpoint(ckpt_path)
        if saved and saved.get("status") == "in_progress":
            if saved.get("n_ticks") != n_ticks:
                sys.exit(
                    f"ERROR: checkpoint n_ticks={saved.get('n_ticks')} does not match "
                    f"requested n_ticks={n_ticks}; cannot resume."
                )
            start_tick = int(saved["last_completed_tick"]) + 1
            coupled_problems = set(saved.get("coupled_problems", []))
            coupled_solutions = set(saved.get("coupled_solutions", []))
            couplings = list(saved.get("couplings", []))
            tick_diagnostics = list(saved.get("tick_diagnostics", []))
            evaluations = list(saved.get("evaluations", []))
            proxy.energy = dict(saved.get("proxy_energy", {}))
            manifest = saved.get("run_manifest", manifest)
            if audit_path.is_file():
                audit.extend(json.loads(audit_path.read_text(encoding="utf-8")))
            LOGGER.info(
                "Resuming %s from tick %d (last completed tick %d)",
                base_name, start_tick, saved["last_completed_tick"],
            )
        elif result_path_for(base_name).is_file():
            existing = json.loads(result_path_for(base_name).read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                LOGGER.info("Run already complete: %s", base_name)
                return result_path_for(base_name), audit_path

    extras = {
        "condition": condition,
        "alignment_tier": tier,
        "similarity_diagnostic": similarity_diagnostic,
        "solution_decoupling": solution_decoupling,
        "origin_pair_exclusions": origin_pair_exclusions,
        "satisficing_threshold": satisficing_threshold if condition == "ranked" else None,
        "use_perceptual_noise": use_perceptual_noise if condition == "ranked" else None,
        "arrival_schedule": arrivals,
    }

    for tick in range(start_tick, n_ticks + 1):
        active_problems = [
            p for p in problem_ids if arrivals[p] <= tick and p not in coupled_problems
        ]
        active_solutions = [
            s for s in solution_ids if arrivals[s] <= tick and s not in coupled_solutions
        ]
        if active_problems and active_solutions:
            allowed_by_problem = {
                problem_id: [
                    sid for sid in active_solutions
                    if sid != blocked_map.get(problem_id)
                ]
                for problem_id in active_problems
            }
            visible_counts = {
                problem_id: len(allowed_by_problem[problem_id])
                for problem_id in active_problems
            }
            if not any(visible_counts.values()):
                continue
            system_prompt = prompts[f"{condition}_system"]

            repeat_outputs_by_problem: dict[str, list[dict[str, Any]]] = {
                problem_id: [] for problem_id in active_problems
            }
            n_model_calls = 0
            n_calls_with_valid_output = 0

            if problems_per_call == 1:
                for problem_index, problem_id in enumerate(active_problems):
                    allowed = allowed_by_problem[problem_id]
                    if not allowed:
                        continue
                    user_prompt = build_user_prompt_for_problem(
                        condition=condition,
                        problem_id=problem_id,
                        items=items,
                        allowed_solution_ids=allowed,
                    )
                    for repeat in range(n_repeats):
                        n_model_calls += 1
                        call_seed = (
                            seed * 1_000_000
                            + tick * 10_000
                            + problem_index * 100
                            + repeat
                        )
                        response = call_model(config, system_prompt, user_prompt, call_seed)
                        if condition == "ranked":
                            parsed, issues = parse_rankings(
                                response.text, {problem_id}, set(active_solutions)
                            )
                        else:
                            parsed, issues = parse_selections(
                                response.text, {problem_id}, set(active_solutions)
                            )
                        if parsed:
                            n_calls_with_valid_output += 1
                        repeat_outputs_by_problem[problem_id].append(
                            {"parsed": parsed, "issues": issues}
                        )
                        audit.write(
                            {
                                "tick": tick,
                                "problem_id": problem_id,
                                "problem_ids": [problem_id],
                                "repeat": repeat,
                                "condition": condition,
                                "alignment_tier": tier,
                                "model_id": response.model_id,
                                "version_pinned": response.version_pinned,
                                "rendering": response.rendering,
                                "call_seed": call_seed,
                                "system_prompt": system_prompt,
                                "user_prompt": user_prompt,
                                "raw_response": response.text,
                                "parsed": parsed,
                                "parse_issues": issues,
                                "no_valid_output": not parsed,
                            }
                        )
            else:
                for batch_index, batch_problem_ids in enumerate(
                    chunk_problems(active_problems, problems_per_call)
                ):
                    batch_set = set(batch_problem_ids)
                    user_prompt = build_user_prompt_batched(
                        condition=condition,
                        problem_ids=batch_problem_ids,
                        items=items,
                        allowed_by_problem=allowed_by_problem,
                    )
                    for repeat in range(n_repeats):
                        n_model_calls += 1
                        call_seed = (
                            seed * 1_000_000 + tick * 1_000 + batch_index * 10 + repeat
                        )
                        response = call_model(config, system_prompt, user_prompt, call_seed)
                        if condition == "ranked":
                            parsed, issues = parse_rankings(
                                response.text, batch_set, set(active_solutions)
                            )
                        else:
                            parsed, issues = parse_selections(
                                response.text, batch_set, set(active_solutions)
                            )
                        if parsed:
                            n_calls_with_valid_output += 1
                        for problem_id in batch_problem_ids:
                            repeat_outputs_by_problem[problem_id].append(
                                {"parsed": parsed, "issues": issues}
                            )
                        audit.write(
                            {
                                "tick": tick,
                                "problem_id": None,
                                "problem_ids": batch_problem_ids,
                                "repeat": repeat,
                                "condition": condition,
                                "alignment_tier": tier,
                                "model_id": response.model_id,
                                "version_pinned": response.version_pinned,
                                "rendering": response.rendering,
                                "call_seed": call_seed,
                                "system_prompt": system_prompt,
                                "user_prompt": user_prompt,
                                "raw_response": response.text,
                                "parsed": parsed,
                                "parse_issues": issues,
                                "no_valid_output": not parsed,
                            }
                        )

            tick_diag: dict[str, Any] = {
                "tick": tick,
                "n_repeats": n_repeats,
                "n_model_calls": n_model_calls,
                "n_calls_with_valid_output": n_calls_with_valid_output,
                "n_repeats_with_valid_output": n_calls_with_valid_output,
                "per_problem_agreement": {},
                "visible_solutions_per_problem": visible_counts,
            }

            if condition == "ranked":
                for problem_id in active_problems:
                    repeat_outputs = repeat_outputs_by_problem[problem_id]
                    if not repeat_outputs:
                        continue
                    per_repeat = [r["parsed"].get(problem_id, []) for r in repeat_outputs]
                    modal_ranking, agreement = modal(per_repeat)
                    tick_diag["per_problem_agreement"][problem_id] = agreement
                    if not modal_ranking:
                        continue
                    for solution_id in modal_ranking[:LOCAL_SEARCH_N]:
                        if solution_id not in allowed_by_problem[problem_id]:
                            continue
                        if is_blocked_origin_match(problem_id, solution_id, blocked_map):
                            continue
                        if solution_id in coupled_solutions:
                            continue
                        if not proxy.can_evaluate(problem_id):
                            break
                        sim = cosine_similarity(
                            items[problem_id]["vector"], items[solution_id]["vector"]
                        )
                        record = proxy.evaluate(problem_id, sim)
                        evaluations.append(
                            {
                                "tick": tick,
                                "problem_id": problem_id,
                                "solution_id": solution_id,
                                "modal_agreement": agreement,
                                "repeat_choices": per_repeat,
                                **record,
                            }
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
                                        "modal_agreement": agreement,
                                        "repeat_choices": per_repeat,
                                    },
                                    solution_decoupling,
                                )
                            )
                            break
            else:
                for problem_id in active_problems:
                    repeat_outputs = repeat_outputs_by_problem[problem_id]
                    if not repeat_outputs:
                        continue
                    per_repeat = [r["parsed"].get(problem_id) for r in repeat_outputs]
                    modal_choice, agreement = modal(per_repeat)
                    tick_diag["per_problem_agreement"][problem_id] = agreement
                    if modal_choice is None or modal_choice in coupled_solutions:
                        continue
                    if modal_choice not in allowed_by_problem[problem_id]:
                        continue
                    if is_blocked_origin_match(problem_id, modal_choice, blocked_map):
                        continue
                    sim = cosine_similarity(
                        items[problem_id]["vector"], items[modal_choice]["vector"]
                    )
                    coupled_problems.add(problem_id)
                    coupled_solutions.add(modal_choice)
                    couplings.append(
                        annotate_coupling(
                            {
                                "tick": tick,
                                "problem_id": problem_id,
                                "solution_id": modal_choice,
                                "cosine_similarity": round(sim, 6),
                                "modal_agreement": agreement,
                                "repeat_choices": per_repeat,
                            },
                            solution_decoupling,
                        )
                    )

            tick_diagnostics.append(tick_diag)

        save_checkpoint(
            ckpt_path,
            base_name=base_name,
            manifest=manifest,
            last_completed_tick=tick,
            n_ticks=n_ticks,
            coupled_problems=coupled_problems,
            coupled_solutions=coupled_solutions,
            couplings=couplings,
            tick_diagnostics=tick_diagnostics,
            evaluations=evaluations,
            proxy_energy=proxy.energy,
            audit_path=audit_path,
            extras=extras,
        )
        audit.flush()
        write_partial_result(
            base_name,
            build_result_payload(
                manifest=manifest,
                condition=condition,
                tier=tier,
                similarity_diagnostic=similarity_diagnostic,
                solution_decoupling=solution_decoupling,
                origin_pair_exclusions=origin_pair_exclusions,
                satisficing_threshold=satisficing_threshold,
                use_perceptual_noise=use_perceptual_noise,
                arrivals=arrivals,
                couplings=couplings,
                tick_diagnostics=tick_diagnostics,
                evaluations=evaluations,
                n_ticks=n_ticks,
                items=items,
                problem_ids=problem_ids,
                solution_ids=solution_ids,
                coupled_problems=coupled_problems,
                coupled_solutions=coupled_solutions,
                audit_path=audit_path,
                status="in_progress",
                last_completed_tick=tick,
            ),
        )
        LOGGER.info(
            "%s tick %d/%d complete: %d couplings so far",
            base_name, tick, n_ticks, len(couplings),
        )

    audit.close()
    result = build_result_payload(
        manifest=manifest,
        condition=condition,
        tier=tier,
        similarity_diagnostic=similarity_diagnostic,
        solution_decoupling=solution_decoupling,
        origin_pair_exclusions=origin_pair_exclusions,
        satisficing_threshold=satisficing_threshold,
        use_perceptual_noise=use_perceptual_noise,
        arrivals=arrivals,
        couplings=couplings,
        tick_diagnostics=tick_diagnostics,
        evaluations=evaluations,
        n_ticks=n_ticks,
        items=items,
        problem_ids=problem_ids,
        solution_ids=solution_ids,
        coupled_problems=coupled_problems,
        coupled_solutions=coupled_solutions,
        audit_path=audit_path,
        status="complete",
        last_completed_tick=n_ticks,
    )
    result_path = write_smoke_result(base_name, result)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    LOGGER.info(
        "Condition %s [%s] complete: %d couplings, result=%s audit=%s",
        condition, tier, len(couplings), result_path, audit.path,
    )
    return result_path, audit.path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run L2 ranked or L3 model-selection conditions."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--corpus-id", default="sbir_v1")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--n-ticks", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3,
                        help="Model calls per tick for non-determinism handling (default 3)")
    parser.add_argument("--model-config", required=True,
                        help="Path to a model config JSON (configs/models/*.json)")
    parser.add_argument(
        "--condition",
        choices=["ranked", "autonomous"],
        required=True,
        help="Use ranked for L2 or the legacy autonomous label for L3 model selection.",
    )
    parser.add_argument("--prompt-version", default="coupling_prompts_v1")
    parser.add_argument(
        "--satisficing-threshold",
        type=float,
        default=SATISFICING_THRESHOLD,
        help="L2 RuleProxy threshold (default 0.60, same as L1).",
    )
    parser.add_argument(
        "--problems-per-call",
        type=int,
        default=1,
        help=(
            "Active problems per model call (default 1 = one problem per call). "
            "Use 0 to batch all active problems per call."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if present for this run.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_condition(
        condition=args.condition,
        seed=args.seed,
        corpus_id=args.corpus_id,
        embedding_model_key=args.embedding_model,
        n_ticks=args.n_ticks,
        n_repeats=args.repeats,
        model_config_path=args.model_config,
        prompt_version=args.prompt_version,
        satisficing_threshold=args.satisficing_threshold,
        resume=args.resume,
        problems_per_call=args.problems_per_call,
    )


if __name__ == "__main__":
    main()
