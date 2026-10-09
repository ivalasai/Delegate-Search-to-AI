# Delegation architecture in organizational search

This repository is the computational instrument, not the paper. It studies how problem–solution couplings are generated and selected in a fixed, garbage-can-inspired environment.

HumanProxy is a transparent computational proxy for boundedly rational organizational search. It is not a human subject and is not intended to reproduce human behavior.

- **L1 — HumanProxy:** bounded candidate sampling and satisficing acceptance.
- **L2 — Model ranking + HumanProxy acceptance:** model ordering followed by the same bounded acceptance rule.
- **L3 — Model selection:** direct model selection without the proxy's gate or energy constraint.

These architectures differ in selection, acceptance, and scheduling. They do not isolate the causal effect of AI autonomy. Cosine 0.60 is an operational gate, not organizational fitness. L3 is never reported as a percentage of the gated ceiling of 54.

## Run the toy

Requires Python 3; no models or downloads:

```sh
./scripts/reproduce_minimal.sh
```

Expected: `couplings=4 evaluations=13`; output: `results/toy_reproduction.json`.
The toy path tests software execution only and is not a scientific replication of the Stage-1 experiment.

## Reproduce the evidence analysis

Tested with Python 3.14 and the pinned analysis dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_stage1_evidence.py
PYTHONPATH=src:scripts python -m unittest discover -s tests
```

Expected: 18 frozen exports match; 17 tests pass. The wrapper invokes the existing analysis on frozen numerical records, writes to a temporary directory, and only then compares to the committed exports. CSV comparison is byte-exact; JSON comparison requires exact parsed values, ignoring object-key order and whitespace. No expected file supplies the calculated answers. A mismatch exits nonzero. The script never calls Ollama.

The main evidence contains 33 complete cells (3 L1, 15 L2, 15 L3), including all six Model E cells. Three historical extended-horizon L1 records are included solely to reproduce the existing combined summary. L1 couplings range 9–15, L2 17–30, and L3 88–90. L3 collision counts range 92–249. Corrected identity comparisons place 17 of 18 L1/L2 comparisons within the existing reference intervals, not an equivalence result.

See [evidence scope and limitations](evidence/README.md) and [PROTOCOL](PROTOCOL.md). Reproducing these analyses does not independently validate source extraction, output parsing, or substantive coupling relevance.

## Reuse and fresh simulation

Provide another corpus and embeddings in the documented schemas and use local Ollama model tags. Retain the frozen prompts, proxy rules, gate and origin exclusion when claiming the same instrument. Full SBIR-grid simulation requires the source corpus and local models; abstract text is not distributed here. Fresh stochastic model runs are not guaranteed to match the saved results. No full simulation launches automatically when cloning or running the evidence checks.

## Release

License: see LICENSE for code; evidence provenance and scope are documented separately. No claim is made to ownership of third-party source material or model weights.

Corrected release commit: pending creation of the anonymous release history. No older development SHA is represented as this release.
