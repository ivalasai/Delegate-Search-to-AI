#!/usr/bin/env python3
"""Stream-download SBIR award CSV and sample rows with usable abstracts."""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

SBIR_URL = "https://data.www.sbir.gov/awarddatapublic/award_data.csv"
MIN_ABSTRACT_LEN = 200
DEFAULT_SAMPLE_SIZE = 40
DEFAULT_SEED = 42

OUTPUT_COLUMNS = ["award_id", "agency", "year", "company", "abstract"]


def _eligible(row: dict[str, str]) -> bool:
    abstract = (row.get("Abstract") or "").strip()
    return len(abstract) >= MIN_ABSTRACT_LEN


def _normalize(row: dict[str, str]) -> dict[str, str]:
    award_id = (row.get("Agency Tracking Number") or row.get("Contract") or "").strip()
    if not award_id:
        raise ValueError("Eligible row missing award identifier fields.")
    return {
        "award_id": award_id,
        "agency": (row.get("Agency") or "").strip(),
        "year": (row.get("Award Year") or "").strip(),
        "company": (row.get("Company") or "").strip(),
        "abstract": (row.get("Abstract") or "").strip(),
    }


def reservoir_sample_eligible(
    reader: csv.DictReader,
    k: int,
    seed: int,
) -> tuple[list[dict[str, str]], int]:
    """Single-pass reservoir sample over rows passing eligibility filter."""
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    eligible_seen = 0

    for row in reader:
        if not _eligible(row):
            continue
        normalized = _normalize(row)
        eligible_seen += 1
        if len(reservoir) < k:
            reservoir.append(normalized)
        else:
            j = rng.randint(0, eligible_seen - 1)
            if j < k:
                reservoir[j] = normalized

    return reservoir, eligible_seen


def _sample_from_reader(
    reader: csv.DictReader,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], int]:
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row.")
    required = {"Abstract", "Agency", "Company", "Award Year"}
    missing = required - set(reader.fieldnames)
    if missing:
        raise ValueError(f"CSV missing expected columns: {sorted(missing)}")
    return reservoir_sample_eligible(reader, sample_size, seed)


def stream_sample_file(
    path: Path,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], int]:
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return _sample_from_reader(csv.DictReader(handle), sample_size, seed)


def stream_sample_url(
    url: str,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], int]:
    request = urllib.request.Request(url, headers={"User-Agent": "garbage-consensus-pipeline/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            text_stream = io.TextIOWrapper(response, encoding="utf-8", errors="replace")
            return _sample_from_reader(csv.DictReader(text_stream), sample_size, seed)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download SBIR data from {url}: {exc}") from exc


def write_sample(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample SBIR abstracts for pilot corpus.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus_raw/sbir_sample.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--url", default=SBIR_URL)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local award_data.csv path (stream-read). Skips URL download.",
    )
    args = parser.parse_args()

    if args.input is not None:
        print(f"Streaming local file {args.input} ...", file=sys.stderr)
        sample, eligible_count = stream_sample_file(args.input, args.sample_size, args.seed)
    else:
        print(f"Streaming {args.url} ...", file=sys.stderr)
        sample, eligible_count = stream_sample_url(args.url, args.sample_size, args.seed)
    if len(sample) < args.sample_size:
        raise RuntimeError(
            f"Only {len(sample)} eligible rows found; need {args.sample_size} "
            f"(eligible total scanned: {eligible_count})."
        )

    write_sample(args.output, sample)
    print(f"Eligible rows scanned: {eligible_count}", file=sys.stderr)
    print(f"Wrote {len(sample)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
