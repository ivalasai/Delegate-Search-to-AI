# Stage-1 Technical Protocol

**Instrument:** Garbage Can Consensus
**Status:** corrected evidence package; simulation protocol unchanged
**Evidence commit SHA:** pending anonymous release commit

## Research question and scientific object

The instrument asks:

> How does delegation of organizational search—from rule-based bounded search,
> to model-assisted ranking, to model selection—change the computational process
> through which problem–solution couplings are generated and selected?

The scientific object is delegation architecture in a garbage-can-inspired
coupling environment, not a model of rational agents, human cognition, or
automated negotiation.

## Corpus and representations

The primary corpus is `sbir_v2`, a local, reproducible sample of 90 public
SBIR award abstracts. Each abstract is split into one problem statement and one
solution statement by extraction model `phi3:mini` through Ollama, using
`prompts/extraction_prompt_v1.md` at temperature `0.1`. Extraction is data
preparation only, not a comparison condition. Problems and solutions are
embedded with pinned `nomic-embed-text` (768 dimensions). A seeded permutation
assigns solution content to display IDs, preventing same-index reunions; the
solution carrying each problem’s source-origin content is blocked for that
problem but remains in the shared pool for other problems.

## Arrival and repetition

The configured horizon is n_ticks=17; the implementation processes ticks 0 through 17 inclusive (18 rounds). Problems and solutions arrive by a
seeded uniform schedule within the first 40% of the horizon. Comparison-model
calls use temperature `0.2`, one problem per call, and three repeats per
decision. The modal parsed response is used. Stage-1 seeds are 11, 22, and 33.
Prompts are frozen at `coupling_prompts_v1`.

## HumanProxy mechanics

HumanProxy is a transparent computational proxy for boundedly rational
organizational search. It is not a human subject and is not intended to
reproduce human behavior. For each active problem, it samples at most `n=2`
eligible solutions per tick. Energy starts at `1.0` and decreases by `0.15`
per evaluation. The first candidate with true cosine at least `0.60` is
accepted. Perceptual noise is off. Each problem and solution can appear in
only one final coupling.

The `0.60` value is an operational satisficing gate in the frozen embedding
geometry, not organizational fitness, substantive relevance, or business
quality.

## Delegation architectures

- **L1 — HumanProxy:** HumanProxy performs local search and acceptance.
- **L2 — Model ranking + HumanProxy acceptance:** the LLM ranks eligible
  solutions; HumanProxy evaluates the top candidates with the L1 gate, energy,
  and satisficing rule.
- **L3 — Model selection:** the LLM directly selects pairings. HumanProxy,
  the `0.60` gate, and energy are absent. A selection for an already claimed
  solution is a shared-pool collision skip, not negotiation.

L3 therefore changes both the selector and the stopping rule. It is a
model-selection architecture, not an isolated causal treatment of AI autonomy.

## Matching and metrics

Origin exclusion is applied per problem. Final matching is injective. The
maximum threshold-cleared matching for `sbir_v2` is 54, and that ceiling is
used only for L1/L2 percentage-throughput reporting.

Reported metrics include raw coupling throughput, cosine distributions,
eligibility/gate-clearing, evaluation/search concentration, entropy, HHI,
effective support, collisions by tick, temporal search structure, attractor
diagnostics, and seed/model comparisons. L1/L2 final-matching HHI equals
`1/n` mechanically and is not a behavioral concentration result. Behavioral
concentration uses evaluation attempts for L1/L2 and modal `SELECT` events,
including collision skips, for L3.

L3 is never reported as a percentage of the 54 ceiling. Below-`0.60` L3
selections are descriptive diagnostics, not proven bad matches. L3 raw volume
is not compared to gated L1/L2 as if all architectures share one stopping
rule.

## Model list

| Label | Comparison model |
|---|---|
| A | Mistral 7B instruct |
| B | Llama 3.1 8B |
| C | Nemotron 3 Nano 4B |
| D | Qwen2.5 7B instruct |
| E | Gemma2 9B |

Models are descriptive robustness implementations, not an alignment-intensity
scale. Legacy `alignment_tier` names remain in config and artifact paths for
compatibility. `phi3:mini` is extraction-only (data preparation), not a
comparison model.

## Reproducibility and SHA

Source, schemas, prompts, model configs, toy materials, corrected exports, and the numerical evidence pack are release files. See evidence/README.md for retained inputs and verification limits. Abstract texts, full raw response logs, checkpoints, and private notes are excluded. Long model runs checkpoint by tick and require their own runtime artifacts to resume. This numerical analysis pack is not a resumable model checkpoint. No old development SHA identifies this corrected release.

The active Model E grid is complete at seeds 11, 22, and 33. Later analyses,
second-corpus tests, selector/gate orthogonalization, and human validation are
Stage-2 extensions rather than requirements of this frozen instrument.

## Reuse

Swap the corpus by providing problem and solution texts plus embeddings in the
same schema. Swap models by pointing the public configs at local Ollama tags.
Do not change the `0.60` threshold, energy rule, local `n=2`, origin exclusion,
or v1 prompts when claiming the same instrument. Extraction model `phi3:mini`
is data preparation, not a comparison condition. Do not add a 2×2 design or a
gated-autonomous condition.

## Known limitations and technical references

The embedding threshold is geometry-specific, HumanProxy is an inspectable
bounded-search control rather than a synthetic human, local model inference
retains residual nondeterminism, and L1 problem order differs from the
model-mediated scheduler. Cosine similarity is therefore reported as an
operational gate diagnostic rather than organizational fitness (Steck,
Ekanadham, & Kallus, 2024). The project does not treat the LLMs as synthetic
human subjects (Horton, 2023), and `nomic-embed-text` is documented by Nussbaum
et al. (2024). The environment is garbage-can-inspired rather than a full
four-stream agent-based reconstruction; Fioretti and Lomi (2008) provide a
useful boundary reference.

References: Horton, J. J. (2023), “Large Language Models as Simulated Economic
Agents: What Can We Learn from Homo Silicus?”, NBER Working Paper 31122,
https://doi.org/10.3386/w31122; Nussbaum, Z., Morris, J. X., Mulyar, A., &
Duderstadt, B. (2024), “Nomic Embed: Training a Reproducible Long Context Text
Embedder,” arXiv:2402.01613; Steck, H., Ekanadham,
C., & Kallus, N. (2024), “Is Cosine-Similarity of Embeddings Really About
Similarity?”, https://doi.org/10.1145/3589335.3651526; Fioretti, G., & Lomi,
A. (2008), “An Agent-Based Representation of the Garbage Can Model of
Organizational Choice,” *Journal of Artificial Societies and Social Simulation,
11*(1), 1, https://www.jasss.org/11/1/1.html.

## Corrected identity analysis

Cross-seed identities use source-solution mappings. Seventeen of eighteen L1/L2 comparisons lie within the existing graph- and size-conditioned reference intervals; this is not an equivalence test. The reference is nonuniform and does not reproduce temporal search. L3 identity overlap is descriptive. L1 updates availability while processing shuffled problems; model conditions expose the tick-start pool and commit in sorted problem order. This scheduler distinction is part of the implemented architecture.
