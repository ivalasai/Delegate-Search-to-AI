# Verification review — October 9, 2026

These checks supplement the frozen evidence; they do not change the simulation or historical results. The original evidence freeze remains the reference. Export agreement is computational consistency, not comprehensive scientific validation.

## Independent numerical checks

All 33 primary cells and three historical L1 records passed recomputed accepted-pair similarities (absolute tolerance 0.00000051 for six-decimal stored values), injectivity, origin exclusion, arrival eligibility, horizon bounds and exact commitment-trace replay. Gated evaluation traces also passed per-problem energy, two-evaluations-per-tick breadth, true/perceived similarity agreement, and acceptance checks. L3 modal replay exactly reproduced accepted pairs and collision totals. See invariant_audit.json and scripts/evidence_invariants.py.

One individual response anomaly remains visible: Model C, seed 22, tick 5, problem P083, repeat 2 selected S049 (source S083), forbidden by origin exclusion. Repeats 0 and 1 selected S041, so S041 won. No forbidden pair was committed. The parser admits globally active solution IDs; the per-problem exclusion is enforced later at commitment. This minority response is reported as an anomaly, not an invalid accepted coupling.

## Historical model-call controls

- **Verified from code:** the generation seed is sent at the top level of the Ollama request; official examples specify options.seed. An earlier available development version uses the same top-level field. Environment seeds still control Python arrival/permutation logic.
- **Unknown:** whether the historical Ollama runtime honored the top-level generation seed. No historical Ollama version/effective-options record was found in the inspected manifests and 30 model audit logs. Do not claim verified generation-seed control.
- **Verified from code:** version_pinned is copied from configuration, not checked against an installed digest. The logs repeat that declaration. Independently verified historical installed digests are not established by these files.
- **Verified from manifests/configs:** A/B omit num_ctx; C/D/E specify 8192. Effective A/B context and historical truncation are unknown from the inspected records. Current installed settings would not prove historical settings.

These are historical provenance limitations. Fixing request construction, enforcing digests, or standardizing context would change future execution. Those changes are deliberately not made silently in this frozen implementation. Fresh runs must first use an explicitly versioned execution correction/preflight.

Ollama reference: https://github.com/ollama/ollama/blob/main/docs/api.md#request-reproducible-outputs

## Historical parser discrepancy — needs author resolution

A local audit reparsed 58,458 saved responses across 30 model cells using the published parser and reconstructed tick-start global eligibility. Eleven responses in Model A ranked seed 11 differ from the stored parsed rankings. All other checked responses reproduce the stored parsed output. The original raw logs remain private; local_parser_audit_summary.json is an audit report, not independently replayable raw-response evidence.

For example, at tick 0 for P024, short IDs S11 and S58 occur at the front of a raw ranking. The stored ranking begins S072, S066; the current parser instead begins S011, S058 after canonicalization. This can change the two candidates evaluated and is not merely a cosmetic label difference. The run manifest's referenced historical source commit was not available through the inspected local Git history. Do not infer the exact old parser implementation from this pattern alone.

The current evidence reconstruction intentionally uses recorded decisions. Its passing result does not show that the published parser would regenerate all those decisions from raw outputs. Preserve the historical records. Before claiming end-to-end replication, recover the actual historical parser if possible and determine the affected cell's impact; otherwise disclose this difference and define a separate prospective reproduction version. No cell was rerun and no ranking was replaced.

## Scope of reference checks and exploratory analyses

The existing A/B seed-11 locked values are regression targets, not independent scientific validation of all conditions. General invariant checks now cover all provided cells. A matching checksum proves consistency with a pinned package, not authenticity independent of that package.

Threshold replay conditions on evaluations already logged under the original gate. Changing the gate would alter future searches, candidate availability and commitments. It is not a fresh threshold experiment. Its percentage fields retain the original 54 reference ceiling; they are not percentages of a recomputed alternative-threshold optimum. Do not present those fields as independent robustness evidence.

The identity reference retains the largest of repeated shuffled greedy matchings, so it is nonuniform over feasible matchings and lacks temporal/energy dynamics. Seventeen of eighteen comparisons within its intervals does not establish equivalence, random allocation, or absence of path dependence.

## Worked collision trace

Model A, L3, seed 11, tick 0: P024 selected S011 and committed it. P029 also selected S011, but was processed later and collision-skipped. Both saw the tick-start pool. This illustrates shared-pool contention created by selection plus commitment order; it does not identify a pure model-preference effect or bargaining behavior.

The public pack lacks substantive text and raw model responses. A source-text example with permitted provenance and substantive assessment remains necessary for evaluating organizational meaning. The private author example prepared separately is not evidence that redistribution permission has been settled.

## Priorities

Before submission: disclose/resolve the parser discrepancy; distinguish intended from verified runtime settings; provide a permitted substantive example or reviewer-access route. Verification improvements alone do not resolve these issues.

For future experiments: explicitly version effective seed placement, digest checks, context and truncation records. Any rerun decision should be scoped to affected claims/cells rather than assume all 33 must be replaced.
