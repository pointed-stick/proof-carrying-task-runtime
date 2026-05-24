# mock_trial_v1.2 — research artifact (final freeze)

The project's frozen state at end-of-development. **A strong research artifact;
a weak tool prototype.**

## Headline

```
A0 success rate over 3 versions:
  v1.0 (development arc):     1/3   — could succeed, accepted bad/truncated answers
  v1.1 (truncation guard):    1/10  — stricter; fewer wrong accepts, more escalations
  v1.2 (founder fixes):       5/10  — founder fixes improved recall; not tool-reliable
```

Per the user's pre-registered decision rule:
- `A0 ≥ 7/10` → credible tool-prototype trajectory
- `A0 ≤ 6/10` → freeze as research artifact

**v1.2 lands at 5/10. Frozen.**

## What's in this directory

| file | what |
|---|---|
| `proof_tree_example_success.json` | one successful end-to-end run (run 6, "Marcus Pohlmann") |
| `graph_example_success.json` | accepted triples for that run |
| `a0_x10_results.json` | full A0 ×10 characterization results |
| `founder_diagnostic_results.json` | F0/F0_inverse/F1/F2 diagnostic results (all 4 PASS in v1.2) |
| this README | frozen-state context |

## Founder-slot diagnostic — all 4 modes pass in v1.2

| mode | v1.1 (pre-patch) | v1.2 (post-patch) |
|---|---|---|
| F0 (forward direction) | PASS | PASS |
| F0_inverse (reverse direction) | FAIL — `value_position` only matched subject overlap | **PASS** — `value_position: "either"` accepts both directions |
| F1 (deterministic scan, honorific ambiguity) | FAIL — uniqueness invariant flagged "Marcus Pohlmann" vs "Professor Marcus Pohlmann" as distinct | **PASS** — `value_canonicalization: "person"` normalizes honorifics |
| F2 (constrained LLM acquisition, compound-entity object) | FAIL — role-sensitive object policy rejected partial-form mentions like "the program" | **PASS** — `compound_object_aliases` in `RELATION_SCHEMAS["founded"]` authorize partial forms |

## What landed in v1.2

| component | purpose |
|---|---|
| `normalize_person_value(s)` | strips honorifics (Professor, Dr., etc.) for slot uniqueness |
| `resolve_relation_follow_slot` `value_position: "either"` | accepts triples in either direction; slot value = non-consumed side |
| `resolve_relation_follow_slot` `value_canonicalization: "person"` | uniqueness comparison uses normalized values; original surface preserved |
| `RELATION_SCHEMAS["founded"].compound_object_aliases` | "mock trial program" can be mentioned as "mock trial" / "the program" |
| `_propose_triples` consults schema's compound aliases | object resolution succeeds for compound entities w/o weakening role-sensitive policy for persons |
| 3 new regression tests | one per fix; locked down in `test_synthetic_benchmark.py` |

## Test catalog

```
test_recursion.py                       1 test
test_runtime_invariants.py             10 tests
test_knowledge_qa.py                   65 tests
test_synthetic_benchmark.py            16 tests
                                       ─────
                                       92 tests, all passing
```

## Failure-class catalog (final)

```
Development arc (rounds 1-12, 15 primitives):
  recursion, slot composition, planning, tool selection, acquisition cost,
  predicate semantics, witness recall, mention resolution, source-local
  aliases, relation surface forms, disallowed-cue direction, uniqueness,
  full-name extraction, avoid-self, graph-backed relation_follow

Characterization additions (rounds 13-14, 4 classes):
  16. Window-truncation partial-name false positives  ← fixed
  17. Last-word object aliases bypassing canonical    ← fixed
  18. Role-insensitive mention resolution             ← fixed
  19. Graph-salvage when child fails to package       ← fixed (locked down via Task 6)

Founder-slot diagnostic (round 15, 3 classes):
  20. Relation direction (forward-only)               ← fixed (value_position: either)
  21. Honorific normalization ambiguity               ← fixed (value_canonicalization)
  22. Compound-entity object policy over-strict      ← fixed (compound_object_aliases)

Open (round 16 characterization, unfixed):
  23. Founder-slot LLM acquisition variance           ← 5/10 still escalate at founder slot
                                                       even with all known transitions fixed
```

22 named failure classes locked down by regression tests. 1 remaining variance class
that the architecture exposes but doesn't yet eliminate.

## Reproduction

```bash
# Reproduce A0 ×10 (cached sources)
python -m task_runtime.examples.mock_trial_multihop_demo \
    --runs 10 --modes oracle_chain_executor

# Reproduce founder-slot diagnostic
python -m task_runtime.examples.founder_slot_diagnostic \
    --modes F0,F0_inverse,F1,F2

# Reproduce ablation matrix
python -m task_runtime.examples.mock_trial_ablation_harness \
    --runs 3 --ablations A1,A2,A4,A5,A6,A8

# All 92 unit tests
python -m pytest -q task_runtime/tests/
```

## The artifact's central contribution

The architecture's main success is not that it reliably answers the
benchmark; it is that it makes unreliability **inspectable**,
**localizable**, and **experimentally reducible**.
