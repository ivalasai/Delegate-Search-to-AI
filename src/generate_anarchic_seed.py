#!/usr/bin/env python3
"""Generate a high-variance, uncoupled multi-stream seed dataset.

Produces a single nested JSON file validated against src/schema/ definitions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Seed generation parameters
# ---------------------------------------------------------------------------

HORIZON_TICKS = 100
SEMANTIC_DIM = 8
SEMANTIC_LOW = -2.0
SEMANTIC_HIGH = 2.0
TAIL_THRESHOLD = 1.5

PROBLEM_COUNT = 30
SOLUTION_COUNT = 30
PARTICIPANT_COUNT = 10
CAN_COUNT = 5

ARRIVAL_TICK_MIN = 1
ARRIVAL_TICK_MAX = 40

HUMAN_COUNT = 7
AGENT_COUNT = 3

PROBLEM_TYPES = ("latent", "active", "chronic", "opportunity")
SOLUTION_TYPES = ("technology", "practice", "rhetoric", "hybrid")
SOLUTION_ORIGINS = ("internal", "imported", "recombinant")
PARTICIPATION_STYLES = ("problem_driver", "solution_carrier", "bystander", "mixed")

PROBLEM_SUBJECTS = (
    "cold-brew saturation",
    "perishable logistics choke points",
    "dark kitchen throughput",
    "suburban corridor demand drift",
    "last-mile refrigerant leakage",
    "ghost franchise brand dilution",
    "micro-fulfillment slot contention",
    "subscription churn cliffs",
    "regulatory labeling ambiguity",
    "seasonal staffing deserts",
    "POS reconciliation drift",
    "vendor invoice mismatch loops",
    "menu SKU proliferation debt",
    "allergen traceability gaps",
    "commissary kitchen idle capacity",
    "franchisee compliance fatigue",
    "dynamic pricing backlash risk",
    "loyalty program redemption spikes",
    "third-party delivery margin erosion",
    "equipment maintenance backlogs",
)

PROBLEM_FRAMES = (
    "Saturated {subject} in {locale}",
    "Unresolved {subject} across {locale}",
    "Volatile {subject} near {locale}",
    "Latent {subject} inside {locale}",
    "Escalating {subject} throughout {locale}",
    "Fragmented {subject} within {locale}",
)

PROBLEM_LOCALES = (
    "suburban corridors",
    "urban micro-markets",
    "airport-adjacent zones",
    "campus peripheries",
    "industrial park fringes",
    "tourist-dense districts",
    "exurban growth rings",
    "transit-hub catchments",
)

SOLUTION_MODIFIERS = (
    "Decentralized",
    "Hyper-local",
    "Automated",
    "Fractional",
    "Adaptive",
    "Modular",
    "Predictive",
    "Asynchronous",
    "Redundant",
    "Composable",
)

SOLUTION_OBJECTS = (
    "temperature routing protocols",
    "delivery network meshes",
    "inventory rebalancing agents",
    "demand sensing overlays",
    "kitchen orchestration kernels",
    "pricing experimentation sandboxes",
    "supplier risk scoring layers",
    "customer narrative templates",
    "compliance audit trails",
    "route optimization heuristics",
    "staffing surge marketplaces",
    "waste heat recovery loops",
    "menu engineering copilots",
    "franchise governance dashboards",
    "cold-chain telemetry fabrics",
)

CAN_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("can_001", "Seed Valuation Round", 10, 30),
    ("can_002", "Co-Founder Alignment Review", 25, 50),
    ("can_003", "Pilot Market Selection Gate", 40, 65),
    ("can_004", "Regulatory Friction Workshop", 55, 80),
    ("can_005", "Bridge Financing Committee", 70, 95),
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def semantic_position(rng: random.Random) -> list[float]:
    """Draw an 8-D coordinate from a wide uniform distribution."""
    return [round(rng.uniform(SEMANTIC_LOW, SEMANTIC_HIGH), 6) for _ in range(SEMANTIC_DIM)]


def chaotic_problem_label(rng: random.Random) -> str:
    frame = rng.choice(PROBLEM_FRAMES)
    subject = rng.choice(PROBLEM_SUBJECTS)
    locale = rng.choice(PROBLEM_LOCALES)
    label = frame.format(subject=subject, locale=locale)
    if rng.random() < 0.35:
        label = f"{label} ({rng.randint(2, 97)}% unverified signal)"
    return label


def uncoupled_solution_label(rng: random.Random) -> str:
    modifier = rng.choice(SOLUTION_MODIFIERS)
    obj = rng.choice(SOLUTION_OBJECTS)
    label = f"{modifier} {obj}"
    if rng.random() < 0.3:
        label = f"{label} v{rng.randint(0, 9)}.{rng.randint(0, 9)}"
    return label


def build_envelope(rng: random.Random) -> dict[str, Any]:
    return {
        "envelope_id": f"env_{uuid.uuid4().hex[:12]}",
        "gcm_mode": "organized_anarchy",
        "temporal_frame": {
            "tick_unit": "discrete",
            "horizon_ticks": HORIZON_TICKS,
            "epoch_origin": datetime.now(timezone.utc).isoformat(),
        },
        "anarchy_conditions": {
            "unclear_preferences": True,
            "unclear_technology": True,
            "fluid_participation": True,
        },
        "structures": {
            "access_structure": {"type": "open"},
            "decision_structure": {
                "type": "resolution",
                "resolution_rule": "energy_threshold",
            },
        },
        "stream_arrival_processes": {
            "problems": {
                "process": "uniform",
                "rate": PROBLEM_COUNT / ARRIVAL_TICK_MAX,
                "variance_param": (ARRIVAL_TICK_MAX - ARRIVAL_TICK_MIN) ** 2 / 12,
            },
            "solutions": {
                "process": "uniform",
                "rate": SOLUTION_COUNT / ARRIVAL_TICK_MAX,
                "variance_param": (ARRIVAL_TICK_MAX - ARRIVAL_TICK_MIN) ** 2 / 12,
            },
            "participants": {
                "process": "uniform",
                "rate": PARTICIPANT_COUNT / HORIZON_TICKS,
                "variance_param": HORIZON_TICKS / 12,
            },
        },
    }


def build_problems(rng: random.Random) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for index in range(1, PROBLEM_COUNT + 1):
        problems.append(
            {
                "problem_id": f"problem_{index:03d}",
                "arrival_tick": rng.randint(ARRIVAL_TICK_MIN, ARRIVAL_TICK_MAX),
                "label": chaotic_problem_label(rng),
                "problem_type": rng.choice(PROBLEM_TYPES),
                "energy_demand": round(rng.uniform(0.15, 0.95), 4),
                "arousal_level": round(rng.uniform(0.1, 1.0), 4),
                "preference_ambiguity": round(rng.uniform(0.5, 1.0), 4),
                "lifecycle_state": "floating",
                "fugitive_params": {
                    "dwell_ticks": rng.randint(3, 25),
                    "flee_probability": round(rng.uniform(0.05, 0.45), 4),
                },
                "semantic_position": semantic_position(rng),
                "source_stream": "exogenous",
            }
        )
    return problems


def build_solutions(rng: random.Random) -> list[dict[str, Any]]:
    solutions: list[dict[str, Any]] = []
    for index in range(1, SOLUTION_COUNT + 1):
        solutions.append(
            {
                "solution_id": f"solution_{index:03d}",
                "arrival_tick": rng.randint(ARRIVAL_TICK_MIN, ARRIVAL_TICK_MAX),
                "label": uncoupled_solution_label(rng),
                "solution_type": rng.choice(SOLUTION_TYPES),
                "carrier_participant_id": None,
                "persistence": round(rng.uniform(5.0, 60.0), 4),
                "coupling_readiness": round(rng.uniform(0.05, 0.85), 4),
                "lifecycle_state": "floating",
                "semantic_position": semantic_position(rng),
                "origin": rng.choice(SOLUTION_ORIGINS),
            }
        )
    return solutions


def build_participants(rng: random.Random, can_ids: Sequence[str]) -> list[dict[str, Any]]:
    types = (["human"] * HUMAN_COUNT) + (["algorithmic_agent"] * AGENT_COUNT)
    rng.shuffle(types)

    participants: list[dict[str, Any]] = []
    for index, participant_type in enumerate(types, start=1):
        arrival_tick = rng.randint(1, 20)
        energy_budget = round(rng.uniform(8.0, 120.0), 4)
        participants.append(
            {
                "participant_id": f"participant_{index:03d}",
                "participant_type": participant_type,
                "arrival_tick": arrival_tick,
                "departure_tick": None,
                "energy_budget": energy_budget,
                "energy_remaining": energy_budget,
                "participation_style": rng.choice(PARTICIPATION_STYLES),
                "access_scope": list(can_ids),
                "availability_windows": [
                    {"start_tick": arrival_tick, "end_tick": HORIZON_TICKS}
                ],
                "problems_held": [],
                "solutions_held": [],
                "preference_signals": None,
                "bounded_rationality_profile": None,
            }
        )
    return participants


def build_choice_opportunities() -> list[dict[str, Any]]:
    cans: list[dict[str, Any]] = []
    for can_id, label, start_tick, end_tick in CAN_SPECS:
        cans.append(
            {
                "can_id": can_id,
                "can_type": "strategic_review",
                "label": label,
                "active_window": {"start_tick": start_tick, "end_tick": end_tick},
                "capacity": 12,
                "current_contents": {
                    "problems": [],
                    "solutions": [],
                    "participants": [],
                },
                "access_gate": [],
                "decision_outcome": None,
                "overflow_state": "empty",
            }
        )
    return cans


def build_stream_arrival_events(
    problems: Sequence[dict[str, Any]],
    solutions: Sequence[dict[str, Any]],
    participants: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    counter = 1

    for entity_type, entities, id_field in (
        ("problem", problems, "problem_id"),
        ("solution", solutions, "solution_id"),
        ("participant", participants, "participant_id"),
    ):
        for entity in entities:
            events.append(
                {
                    "event_id": f"event_{counter:04d}",
                    "event_type": "stream_arrival",
                    "tick": entity["arrival_tick"],
                    "payload": {
                        "entity_type": entity_type,
                        "entity_id": entity[id_field],
                    },
                }
            )
            counter += 1

    events.sort(key=lambda event: (event["tick"], event["event_id"]))
    return events


# ---------------------------------------------------------------------------
# Distributional baseline statistics
# ---------------------------------------------------------------------------


def coordinate_variance(positions: Sequence[Sequence[float]]) -> float:
    """Mean per-dimension variance across all semantic coordinates."""
    if not positions:
        return 0.0
    dim = len(positions[0])
    variances = [
        statistics.pvariance(row[dim_index] for row in positions)
        for dim_index in range(dim)
    ]
    return round(statistics.mean(variances), 6)


def semantic_spread(positions: Sequence[Sequence[float]]) -> float:
    """Mean per-dimension standard deviation (spread proxy)."""
    if not positions:
        return 0.0
    dim = len(positions[0])
    spreads = [
        statistics.pstdev(row[dim_index] for row in positions)
        for dim_index in range(dim)
    ]
    return round(statistics.mean(spreads), 6)


def tail_mass(positions: Sequence[Sequence[float]], threshold: float = TAIL_THRESHOLD) -> float:
    """Share of coordinates in distribution tails (|x| > threshold)."""
    flat = [value for row in positions for value in row]
    if not flat:
        return 0.0
    tail_count = sum(1 for value in flat if abs(value) > threshold)
    return round(tail_count / len(flat), 6)


def participation_style_entropy(participants: Sequence[dict[str, Any]]) -> float:
    counts = Counter(participant["participation_style"] for participant in participants)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )
    return round(entropy, 6)


def build_distributional_baseline(
    problems: Sequence[dict[str, Any]],
    solutions: Sequence[dict[str, Any]],
    participants: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    problem_positions = [problem["semantic_position"] for problem in problems]
    solution_positions = [solution["semantic_position"] for solution in solutions]

    problem_variance = coordinate_variance(problem_positions)
    solution_variance = coordinate_variance(solution_positions)

    LOGGER.info("Problem semantic coordinate variance: %s", problem_variance)
    LOGGER.info("Solution semantic coordinate variance: %s", solution_variance)

    return {
        "baseline_id": f"baseline_{uuid.uuid4().hex[:12]}",
        "computed_at": "input_generation",
        "streams": {
            "problems": {
                "count": len(problems),
                "semantic_spread": semantic_spread(problem_positions),
                "semantic_coordinate_variance": problem_variance,
                "tail_mass": tail_mass(problem_positions),
                "energy_demand_variance": round(
                    statistics.pvariance(problem["energy_demand"] for problem in problems), 6
                ),
            },
            "solutions": {
                "count": len(solutions),
                "semantic_spread": semantic_spread(solution_positions),
                "semantic_coordinate_variance": solution_variance,
                "tail_mass": tail_mass(solution_positions),
                "persistence_variance": round(
                    statistics.pvariance(solution["persistence"] for solution in solutions), 6
                ),
            },
            "participants": {
                "count": len(participants),
                "energy_budget_variance": round(
                    statistics.pvariance(
                        participant["energy_budget"] for participant in participants
                    ),
                    6,
                ),
                "participation_style_entropy": participation_style_entropy(participants),
            },
        },
        "coupling_potential": {
            "possible_pairings": len(problems) * len(solutions),
            "expected_coupling_rate_under_anarchy": round(
                min(1.0, 1.0 / max(len(problems), len(solutions))), 6
            ),
        },
        "notes": (
            "Pre-simulation distributional snapshot. "
            f"Problem semantic variance={problem_variance}; "
            f"solution semantic variance={solution_variance}. "
            "No pre-coupling; streams remain independent."
        ),
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_seed(seed: dict[str, Any], schema_dir: Path) -> None:
    try:
        import jsonschema
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
    except ImportError as exc:
        LOGGER.warning(
            "jsonschema not installed; skipping schema validation (%s)", exc
        )
        return

    resources = []
    for schema_path in sorted(schema_dir.glob("*.schema.json")):
        contents = json.loads(schema_path.read_text(encoding="utf-8"))
        resources.append((schema_path.name, Resource.from_contents(contents)))

    registry = Registry().with_resources(resources)
    root_schema = json.loads(
        (schema_dir / "anarchic_seed.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(root_schema, registry=registry)
    errors = sorted(validator.iter_errors(seed), key=lambda err: list(err.path))
    if errors:
        messages = "\n".join(
            f"  - {error.message} at {list(error.path)}" for error in errors
        )
        raise ValueError(f"Seed failed schema validation:\n{messages}")
    LOGGER.info("Schema validation passed (%d schemas)", len(resources))


# ---------------------------------------------------------------------------
# Assembly & I/O
# ---------------------------------------------------------------------------


def generate_seed(rng: random.Random) -> dict[str, Any]:
    choice_opportunities = build_choice_opportunities()
    can_ids = [can["can_id"] for can in choice_opportunities]

    problems = build_problems(rng)
    solutions = build_solutions(rng)
    participants = build_participants(rng, can_ids)

    seed = {
        "simulation_envelope": build_envelope(rng),
        "problems": problems,
        "solutions": solutions,
        "participants": participants,
        "choice_opportunities": choice_opportunities,
        "events": build_stream_arrival_events(problems, solutions, participants),
        "distributional_baseline": build_distributional_baseline(
            problems, solutions, participants
        ),
    }
    return seed


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate uncoupled multi-stream seed data (high variance)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "raw_anarchic_seed.json",
        help="Output JSON path (default: data/raw_anarchic_seed.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "schema",
        help="Directory containing JSON schemas",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip JSON Schema validation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    rng = random.Random(args.seed)
    if args.seed is not None:
        LOGGER.info("Using random seed: %s", args.seed)

    LOGGER.info(
        "Generating anarchic seed: %d problems, %d solutions, %d participants, %d cans",
        PROBLEM_COUNT,
        SOLUTION_COUNT,
        PARTICIPANT_COUNT,
        CAN_COUNT,
    )

    seed = generate_seed(rng)

    if not args.skip_validation:
        validate_seed(seed, args.schema_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(seed, handle, indent=2)
        handle.write("\n")

    baseline = seed["distributional_baseline"]["streams"]
    LOGGER.info("Wrote seed to %s", args.output.resolve())
    LOGGER.info(
        "Baseline | problems: variance=%.6f spread=%.6f | solutions: variance=%.6f spread=%.6f",
        baseline["problems"]["semantic_coordinate_variance"],
        baseline["problems"]["semantic_spread"],
        baseline["solutions"]["semantic_coordinate_variance"],
        baseline["solutions"]["semantic_spread"],
    )


if __name__ == "__main__":
    main()
