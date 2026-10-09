"""Canonical analysis identities; simulation display IDs remain untouched."""

def identity_mapping(payload: dict) -> dict[str, str]:
    """Require explicit bijective display-to-origin provenance; never guess IDs."""
    mapping = payload.get("solution_decoupling")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Missing solution_decoupling provenance")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("Identity mapping must contain string IDs")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Identity mapping is not bijective")
    return mapping


def origin_id(mapping: dict[str, str], display_id: str) -> str:
    if display_id not in mapping:
        raise ValueError(f"Unmapped solution ID: {display_id}")
    return mapping[display_id]


def pair_set(payload: dict) -> set[tuple[str, str]]:
    mapping = identity_mapping(payload)
    return {
        (row["problem_id"], origin_id(mapping, row["solution_id"]))
        for row in payload.get("couplings", [])
    }


