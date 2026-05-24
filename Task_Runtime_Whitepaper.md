# Proof-Carrying Recursive Task Runtime

*A whitepaper on a recursive LLM-agent research artifact*

**Review target:** `task_runtime.zip`  
**Prepared:** May 24, 2026

> **Whitepaper thesis**  
> The artifact does not yet demonstrate a reliable multi-hop QA tool. It does demonstrate a proof-carrying recursive task runtime and a repeatable method for making LLM-agent failures inspectable, localizable, and experimentally reducible.

---

## Executive Summary

This whitepaper reviews `task_runtime.zip`, a research artifact that explores whether a lightweight recursive LLM runtime can accomplish complex tasks by decomposing them into smaller sub-tasks, executing those sub-tasks through typed contracts, and returning a proof tree rather than a bare answer. The artifact contains a universal runtime, two domain plugins, a multi-hop knowledge-QA benchmark, diagnostics, ablation harnesses, frozen proof outputs, and 92 passing tests verified during this review.

The work started with a broad hypothesis: recursive sub-agent calling might provide a universal mechanism for arbitrary task accomplishment. The final result is more precise. Recursion is the power source, but reliability comes from contracts, verifiers, repair loops, domain-specific executors, and proof-carrying traces. The runtime provides the universal control and proof interface; the plugin supplies task-specific knowledge, tools, schemas, and verification.

| Finding | Whitepaper conclusion |
|---|---|
| Architecture | The core runtime is compact and general: typed contracts, proof nodes, recursive spawning, slot dataflow, verifier-driven repair, and plugin boundaries. |
| Empirical result | The mock-trial benchmark improved from 1/10 to 5/10 after founder-slot diagnostics and targeted fixes, but did not meet the 7/10 threshold for a credible tool prototype. |
| Methodological result | The strongest contribution is the failure-driven development loop: failure → diagnostic → named transition → narrow patch → regression test → re-measure. |
| Reliability tradeoff | Stricter verification increased precision and auditability, but also caused escalations when acquisition failed to produce acceptable evidence. |
| Future direction | The next phase should study acquisition variance, schema generation, proof-tree reproducibility, and generalization across additional benchmarks. |

---

## 1. Problem

LLM agents can often produce plausible answers to difficult questions, but their intermediate reasoning is typically implicit, brittle, and hard to audit. In multi-hop tasks, a single mistaken entity, relation, or citation can cascade into a wrong answer while still sounding coherent. The problem explored by this artifact is not simply answer generation; it is how to turn a difficult task into a structured, inspectable artifact whose subclaims and subdecisions can be verified.

The motivating benchmark asks:

> **Benchmark question**  
> Who founded the mock trial program at the college attended by the Trump-appointed Supreme Court justice whose father’s first name is Michael?

The expected chain is:

| Hop | Required resolution |
|---:|---|
| 1 | Trump-appointed Supreme Court justice whose father’s first name is Michael → Amy Coney Barrett |
| 2 | Undergraduate college attended by Amy Coney Barrett → Rhodes College |
| 3 | Founder of the Rhodes College mock trial program → Marcus Pohlmann |

This task is intentionally awkward: it requires entity enumeration, constraint checking, relation following, avoidance of misleading but true facts, and grounded source evidence. It is therefore useful for testing whether an LLM-agent framework can preserve semantic dataflow under pressure.

---

## 2. Hypothesis

The initial broad hypothesis was:

> **Initial hypothesis**  
> A lightweight recursive runtime might allow an LLM to accomplish arbitrary tasks by recursively spawning sub-agents.

The artifact refined this into a sharper, more testable hypothesis:

> **Refined hypothesis**  
> A recursive runtime can preserve evidence-checkable semantic dataflow when each layer enforces what it can deterministically check, lets the LLM judge only among constrained alternatives, and treats accepted artifacts as an active proof substrate.

This revision matters. Recursive sub-agent calls alone do not create reliability. The reliability boundary is supplied by task contracts, schemas, verifier verdicts, repair hints, evidence records, and domain plugins that know how to certify their own objects.

---

## 3. System Overview

The repository separates a universal task runtime from domain plugins. The runtime owns orchestration; plugins own tools, stores, verifiers, and domain-specific context rendering. In the reviewed zip, this separation is exercised by both a code-generation plugin using pytest and a knowledge-QA plugin using Wikipedia-backed evidence.

| Component | Role in the system | Observed implementation |
|---|---|---|
| `core.py` | Defines `TaskContract`, `TaskResult`, verdicts, `ProofNode`, budgets, repair policy, and a minimal output-schema validator. | 279 lines; contract elasticity levels 0–4. |
| `runtime.py` | Executes tasks, exposes `finish` and `spawn_subtasks` tools, runs repair attempts, merges slot outputs, and writes proof-tree state. | 650 lines; lazy LLM client, Stage 0 sanity, schema verification, plugin verifier dispatch. |
| `plugins/code_tests.py` | Non-QA plugin used to test universality. Writes files, runs pytest, records verifier metadata. | 291 lines; deterministic pytest verifier with hashes. |
| `plugins/knowledge_qa.py` | Domain plugin for evidence-backed QA. Contains graph store, Wikipedia tools, relation schemas, mention resolution, slot certifiers, and executors. | 2,936 lines; most task-specific complexity lives here. |
| `examples/` | Executable demos and diagnostics. | Slugify, repair loop, atomic QA, SCOTUS diagnostics, founder diagnostics, mock-trial benchmark, ablation harness. |
| `tests/` | Regression suite for runtime and plugin invariants. | 92 tests passed in this review. |

> **Review verification**  
> I unpacked `task_runtime.zip` and ran `python -m pytest -q` from the repository root. The result was `92 passed in 0.39 seconds`.

---

## 4. Architecture

### 4.1 Contract-elastic runtime

A task can be as light as a goal string or as strict as a full contract with an output schema, verifier, success criteria, repair policy, budget, dependencies, and allowed tools. This elasticity is important because not every task admits the same level of formal verification. Creative tasks may use rubrics; code tasks can use tests; factual QA can use source-backed claim checks.

The artifact treats contract strictness as a dial rather than a doctrine:

| Level | Contract shape | Typical use |
|---:|---|---|
| 0 | Goal only | Exploratory or creative tasks where no objective verifier exists. |
| 1 | Goal + budget/context | Lightweight task execution with bounded cost. |
| 2 | Goal + output schema | Tasks where malformed output should trigger repair. |
| 3 | Schema + verifier + success criteria | Code tests, evidence checks, or other externally checkable work. |
| 4 | Verifier + slots/dependencies/proof obligations/repair policy | Multi-hop tasks where intermediate values must be certified before downstream use. |

Each level rests on the same small set of runtime concepts:

| Runtime concept | Purpose |
|---|---|
| `TaskContract` | Declares the goal, schema, verifier, allowed tools, budget, repair policy, inputs, and dependencies. |
| `TaskResult` | Captures status, output, artifacts, cost, and notes from an attempt. |
| `Verdict` | `Accept`, `RejectWithRepairHint`, `Escalate`, or `IncoherentContract`. The verifier’s output drives repair or termination. |
| `ProofNode` | Records attempts, child tasks, final result, final verdict, and slot table. This is the proof-tree node. |
| `spawn_subtasks` | Runtime-provided tool that lets a task create child contracts while preserving parent-child proof links. |
| `slot_table` | Dataflow table that carries produced slot values across children. |

### 4.2 Plugin boundary

The runtime deliberately does not know what a triple, pytest test, Wikipedia page, source alias, or mock-trial program is. Those belong to plugins. This design produced a useful separation: the same `run_task` interface works for code generation and knowledge QA, while reliability mechanisms remain domain-specific.

| Domain | Plugin verifier | What is certified |
|---|---|---|
| Code generation | `PytestVerifier` | The implementation passes a specified pytest command; proof records include command, exit code, stdout/stderr tail, and workspace hashes. |
| Knowledge QA | Claim and slot verifiers | Every accepted triple has source provenance, excerpt containment, value-in-evidence checks, predicate support, and slot-level relevance checks. |

### 4.3 Proof tree as product

A central design move is to treat the trace as the product, not as a debug log. The proof tree records each task, attempt, verdict, repair hint, child, cost, and slot table. For code tasks, verifier records include command and file hashes. For QA tasks, graph triples include canonical subject/object, source-surface mentions, source IDs, excerpts, and judge reasons.

A concrete example is **graph salvage**. In the successful chain, the college acquisition child returned the correct value (`Rhodes College`) and cited an accepted triple `(Amy Coney Barrett, attended, Rhodes College)` for the obligation. The strict slot-certificate verifier rejected the citation because the obligation expected the certifying triple's subject to overlap the slot value `Rhodes College`, while the cited triple's subject was the *consumed* slot `Amy Coney Barrett`. Rather than discarding the work, the runtime invoked a `relation_follow` salvage step that re-interpreted the same accepted graph fact from the consumed subject's perspective and certified the slot mechanically. This turns the graph from passive memory into an active proof substrate: if accepted evidence already satisfies a slot obligation under some valid reading, the runtime/plugin can use it even when a child's own framing was rejected.

---

## 5. Experimental Design

The experiment was not a single benchmark run. It was an iterative research process in which each failure became a diagnostic target. The process deliberately avoided adding broad mechanisms until a specific failure demanded them.

1. Build a minimal recursive runtime with structured task contracts and proof nodes.
2. Port a non-QA plugin to test universality: code generation with pytest.
3. Port a knowledge-QA plugin to test evidence-grounded factual work.
4. Use the mock-trial question as a live multi-hop benchmark.
5. When a run fails, inspect the proof tree, name the failure class, add a narrow primitive, and lock it down with tests.
6. Run characterization: repeated A0 baseline, ablations, founder-slot diagnostics, and synthetic benchmark tests.

### 5.1 Failure classes and primitives

The artifact ended with 22 named failure classes fixed and one remaining open variance class. The table below condenses the arc into the major mechanism families while preserving the load-bearing primitives that the ablation and diagnostic work made visible.

| Failure family | Representative primitive added | Effect |
|---|---|---|
| Recursive composition | `spawn_subtasks`, `ProofNode`, `slot_table` | Children attach to parents and produced slot values can feed later tasks. |
| Free-form child outputs | `TaskContract`, schema validation, Stage 0 sanity | Tasks must finish structurally; malformed/no-finish attempts trigger repair or escalation. |
| Invented or weak evidence | Source provenance, excerpt-in-source, value-in-evidence, predicate judge | Triples cannot be written unless source evidence supports them. |
| True but irrelevant facts | `SlotCertificateVerifier` with proof obligations and disallowed predicates | A fact may be valid in the graph yet invalid for a slot. |
| Bad candidate selection | `EntityConstraintResolutionExecutor`; enumerate → fill → filter → certify | The justice slot must enumerate candidates, fill each candidate’s obligations, and accept exactly one satisfying candidate. |
| Name and mention mismatch | Canonical/surface mention split, source-local aliases, role-sensitive mention policy | Graph identity stays canonical while evidence can cite surface forms like `Barrett` or `Amy Vivian Coney`. |
| Relation surface mismatch | `RELATION_SCHEMAS`, relation-targeted search, compound aliases | The schema configuration surface captures allowed cues, disallowed cues, object aliases, compound-object aliases, and relation direction. |
| LLM packaging failures | Graph-backed `relation_follow` and graph salvage | Accepted graph facts can certify slots directly even if a child fails to package output correctly. |
| Partial-name false positives | Truncation guard with source-continuation check | The system rejects names like `Professor Ma` only when the source proves the excerpt cut through a longer name. |
| Founder-slot transition failures | `value_position="either"`, `value_canonicalization="person"`, `compound_object_aliases` | v1.2 fixed relation direction, honorific ambiguity, and compound-object over-strictness in the founder slot. |


---

## 6. Findings

### 6.1 The runtime abstraction held across domains

The same runtime drove a code-generation task and a knowledge-QA task. This supports the claim that the core runtime is domain-neutral. However, the knowledge-QA plugin contains most of the domain-specific reliability machinery, showing that universality belongs to the interface while reliability belongs to the plugin.

### 6.2 Strict verification improved precision but initially reduced recall

The v1.0 system could occasionally answer the benchmark, but it accepted bad or truncated answers such as “Professor Ma.” v1.1 added stricter verification, which reduced wrong accepts but increased honest escalations. This drove answer-level success down to 1/10 while making failures more trustworthy.

| State | A0 success | Interpretation |
|---|---:|---|
| v1.0 | 1/3 | Could succeed, but accepted bad/truncated answers. |
| v1.1 | 1/10 | Stricter; fewer wrong accepts, more escalations. |
| v1.2 | 5/10 | Founder fixes improved recall by 5×, but not enough for tool reliability. |

### 6.3 Founder-slot diagnostics converted variance into transitions

The dominant v1.1 failure was founder-slot acquisition: the system usually resolved justice and college, then failed to certify the founder. The founder diagnostic decomposed this into three concrete transitions: relation direction, honorific normalization, and compound-entity object policy. v1.2 patched all three, and all four founder diagnostic modes passed live.

| Diagnostic mode | v1.1 | v1.2 result |
|---|---|---|
| F0 forward | Pass | Pass |
| F0 inverse direction | Fail | Pass via `value_position="either"` |
| F1 honorific ambiguity | Fail | Pass via person-value normalization |
| F2 compound-object policy | Fail | Pass via `compound_object_aliases` |

### 6.4 Final A0 characterization: improved but not reliable

The frozen v1.2 A0 run over 10 **cached-source** trials produced 5 correct answers. These numbers should be read as cached-source characterization results, not cache-bypassed live-source robustness results. All correct runs returned Marcus Pohlmann. The remaining failures still resolved justice and college but escalated at founder, indicating that the main unresolved issue is not chain structure but acquisition variance in the founder slot.

A key precision point is that the four named founder transitions now pass in v1.2. The residual 5/10 failure is therefore a distinct, newly named open class: **LLM acquisition variance at the founder slot**. The architecture exposes this variance and narrows it, but does not yet eliminate it.

| Metric | v1.2 A0 ×10 result |
|---|---|
| Correct answer | 5/10 |
| Mean chain coverage | 2.5/3 |
| Mean LLM calls | 21.4 |
| Mean tool calls | Approximately 22.0 |
| Dominant remaining failure | Founder-slot LLM acquisition variance |
| Decision rule outcome | Freeze as research artifact because 5/10 is below the 7/10 tool-prototype threshold. |

### 6.5 Ablations mostly reproduced predicted failures

The ablation results indicate that most named mechanisms are not decorative. Removing graph salvage, source-local aliases, full-name father extraction, uniqueness, or `relation_follow` reproduced the expected failure. The avoid-self ablation was anomalous and should be studied with more runs.

| Ablation | Expected failure | Observed characterization |
|---|---|---|
| A1: no graph-salvage | College/founder slot fails when LLM packaging fails. | 0/3 correct; chain stops downstream. |
| A2: no source-local aliases | Barrett father evidence fails because source uses Amy Vivian Coney. | 0/3 correct; justice slot escalates. |
| A4: no full-name father extraction | Bare-Michael failures or no father slot certification. | 0/3 correct; no father triples land. |
| A5: no uniqueness invariant | Multiple candidates can satisfy and wrong candidate may be selected. | 0/3 correct; one run selected Kavanaugh. |
| A6: no avoid-self | Michael Kavanaugh/self-name carving should reappear. | 2/3 correct; anomalous, likely small-N or over-strictness interaction. |
| A8: no relation_follow | Downstream slots depend on brittle child packaging. | 0/3 correct; chain stops at college. |

### 6.6 The final contribution is methodological

The artifact’s most defensible claim is methodological rather than product-level. It shows how to make LLM-agent failures structural. Each failure became visible in a proof tree, was isolated by a diagnostic, transformed into a named transition, patched narrowly, regression-tested, and re-measured.

> **Central finding**  
> The architecture’s main success is not that it reliably answers the benchmark; it is that it makes unreliability inspectable, localizable, and experimentally reducible.

---

## 7. Limitations

- The system is not yet a reliable tool. A final 5/10 success rate is a research artifact result, not a deployment result.
- The benchmark is one hand-authored multi-hop question. The slot specs and relation schemas are tuned to this task family.
- The knowledge-QA plugin is large relative to the runtime, showing that reliability required substantial domain-specific machinery.
- Some proof-tree data is structurally auditable but not fully reproducible: model versions, seeds, full LLM transcripts, and environment details are not fully frozen in every record.
- The LLM judge remains stochastic, and acquisition variance remains the open failure class.
- Live-source variance is not fully separated from cached-source behavior. The reported A0 ×10 characterization is cached-source; a robust evaluation should add explicit cache-bypassed live-source trials.

---

## 8. Future Research

### 8.1 Founder-slot acquisition variance

The immediate open problem is the remaining 5/10 failure rate, concentrated in founder-slot acquisition. Future work should compare runs that succeed and fail at founder and identify whether failures arise from source-window search, witness generation, predicate direction, object normalization, or child finish behavior.

### 8.2 Automatic schema and slot-spec generation

The current system relies on hand-authored `SlotSpec`s and relation schemas. A next research direction is compiling natural-language questions into slot graphs, proof obligations, relation families, disallowed relations, and acquisition strategies automatically.

### 8.3 Stronger proof-tree reproducibility

To make proof trees independently re-checkable, future versions should record model identifiers, model parameters, timestamps, source-page snapshots, tool inputs/outputs, full message transcripts or transcript hashes, and environment versions. For code tasks, this would extend the already-present pytest command and workspace-hash records.

### 8.4 Benchmarks beyond the mock-trial chain

The next benchmark should preserve the same abstract shape while changing entities and sources: entity satisfying constraints → institution relation → organization/program at that institution → founder/creator. This tests generalization without changing every variable at once. After that, the framework should be tested on different task families, including code workflows, document transformations, and constrained planning.

### 8.5 Calibration of strictness

The characterization shows a precision–recall tradeoff: stricter validation avoids bad accepts but can increase escalations. Future systems should expose strictness as a configurable policy, possibly tuned by task risk, model strength, verifier confidence, and human-review requirements.

---

## 9. Conclusion

The reviewed artifact is best understood as a research system for proof-carrying recursive task execution. It does not establish that recursive LLM agents can reliably solve arbitrary tasks. It does establish a promising architecture and a disciplined methodology for converting vague failures into named, testable, and incrementally reducible failures.

The runtime’s universal contribution is a compact interface for tasks, recursion, repair, slots, and proof trees. The plugin’s domain contribution is a detailed apparatus for evidence-bound QA. The experiments show that these layers can produce a correct proof-carrying answer, that many primitives are load-bearing, and that stricter verification has real costs. The final state is therefore a strong research artifact, a weak tool prototype, and a clear agenda for future work.

> **Final sentence**  
> The architecture’s main success is not that it reliably answers the benchmark; it is that it makes unreliability inspectable, localizable, and experimentally reducible.

---

## Appendix A. Artifact Inventory Reviewed

| Artifact | Notes |
|---|---|
| `task_runtime/RESEARCH_REPORT.md` | Existing research report and frozen final framing. |
| `task_runtime/founder_slot_diagnostic.md` | Diagnostic table decomposing founder-slot failure transitions. |
| `task_runtime/core.py` + `runtime.py` | Universal runtime layer; under 1,000 lines in the reviewed zip. |
| `task_runtime/plugins/code_tests.py` | Pytest-based plugin used as non-QA universality check. |
| `task_runtime/plugins/knowledge_qa.py` | Knowledge-QA plugin with 22 primitives and relation/slot machinery. |
| `task_runtime/examples/sample_outputs/mock_trial_v1.2_research_artifact/` | Final frozen artifact with A0 ×10 results, founder diagnostics, graph example, and proof tree example. |
| `task_runtime/tests/` | 92 tests; all passed locally during this review. |

---

## Appendix B. Representative Success Trace

The frozen v1.2 success proof tree records the following slot table:

| Slot | Value |
|---|---|
| justice | Amy Coney Barrett |
| college | Rhodes College |
| founder | Marcus Pohlmann |

The final proof verdict was `Accept` with the reason: `chain executor: every slot certified; answer='Marcus Pohlmann'`. The frozen graph contained eight accepted triples (`t-0000` through `t-0007`), of which six were cited in the final supporting set, spanning the four relations required for the chain: `appointed`, `father`, `attended`, and `founded`.

The JSON excerpts below are drawn directly from `proof_tree_example_success.json` and `graph_example_success.json` in the frozen artifact; only surrounding fields irrelevant to the point being illustrated have been elided.

A representative accepted triple shows how the proof tree records both canonical meaning and source-surface grounding:

```json
{
  "id": "t-0007",
  "subject": "Marcus Pohlmann",
  "subject_mention": "Pohlmann",
  "predicate": "founded",
  "object": "mock trial program",
  "source": "wiki:Marcus Pohlmann",
  "excerpt": "Pohlmann is an accomplished mock trial coach. He founded the mock trial program at Rhodes in 1986 and led the program until his retirement in 2018.",
  "judge_reason": "witness span: Pohlmann is an accomplished mock trial coach. He founded the mock trial program at Rhodes in 1986 a"
}
```

The same proof tree also demonstrates graph salvage. The college acquisition child returned `Rhodes College` and cited triple `t-0004` `(Amy Coney Barrett, attended, Rhodes College)`, but the strict slot-certificate verifier rejected the citation: the obligation expected the certifying triple's subject to be the slot value `Rhodes College`, not the consumed subject `Amy Coney Barrett`. The runtime then created a salvage node that re-interpreted the same accepted triple via `relation_follow` and certified the slot mechanically:

```json
{
  "task_id": "T45dff940",
  "contract": {
    "goal": "relation_follow salvage: 'college'",
    "inputs": {"kind": "relation_follow_salvage", "slot_name": "college"}
  },
  "final_result": {
    "status": "success",
    "output": {
      "value": "Rhodes College",
      "certificate": {
        "method": "relation_follow",
        "cited_obligation_triples": {"0": "t-0004"},
        "candidate_table": [
          {
            "id": "t-0004",
            "subject": "Amy Coney Barrett",
            "predicate": "attended",
            "object": "Rhodes College",
            "source": "wiki:Amy Coney Barrett",
            "excerpt": "After high school, Barrett attended Rhodes College in Memphis, Tennessee, where she majored in English literature and minored in French.",
            "subject_mention": "Barrett"
          }
        ]
      },
      "supporting_triple_ids": ["t-0004"]
    },
    "notes": "salvaged after child: single graph triple satisfies: ('Amy Coney Barrett', 'attended', 'Rhodes College') -> slot value = 'Rhodes College'"
  },
  "final_verdict": {
    "kind": "Accept",
    "reason": "relation_follow (salvage): 'college' = 'Rhodes College'",
    "record": {"verifier": "relation_follow_graph_salvage"}
  }
}
```

This is the graph-as-active-proof-substrate principle in miniature: once a fact is accepted into the graph, a downstream slot can be certified mechanically even when a child's own slot-certificate framing is rejected by the strict verifier.


---

## Appendix C. Per-round failure → primitive arc (development log)

The §5.1 family table condenses the development arc into 10 mechanism families. The table below preserves the original per-round granularity: each row is one round in the failure-driven loop, with the specific failure observed and the single primitive added in response. Earlier strictness was preserved at every step — when a later round changed a primitive, it raised recall or precision without weakening a prior layer.

| # | Failure observed | Primitive added | Effect |
|---:|---|---|---|
| 1 | LLM-as-CPU fakes deterministic work; runtime cannot recurse | `spawn_subtasks` + `ProofNode` plumbing | Recursive children attach to root proof. |
| 2 | Sub-agents return prose; parent cannot compose | `TaskContract` + `Verdict` types | Typed work orders, structured results. |
| 3 | Free-form `add_triple` produces invented facts | `propose_triples` + source-provenance + value-in-evidence + predicate judge | 4-stage write validation. |
| 4 | Slots accepted bare claims of any shape | `SlotCertificateVerifier` with proof obligations + disallowed predicates | Fact-validity separated from slot-relevance. |
| 5 | Chain executor lost question structure across hops | Typed `multi_hop_chain` + `chain_completeness` verifier | Dataflow check across slots. |
| 6 | Model mixed `add_triple` / `propose_triples`; list pages did not ground | `EntityConstraintResolutionExecutor` (mechanical workflow) | Runtime owns enumerate/fill/filter loop. |
| 7 | Per-add_triple cost too high; rejection ratio ~30:1 | `wiki_windows_around` deterministic excerpts + batched `propose_triples` + research ledger in repair context | Cost dropped ~50% per slot. |
| 8 | Yes/no judge accepted "Kavanaugh middle-name Michael" via adjacent text | Structured-witness judge (V2): support_span + subject_mention + predicate_cue + binding_explanation + mechanical validator | Adjacent-text false positives blocked. |
| 9 | V2 witness recall dropped on true claims (LLM picked imperfect spans) | Deterministic candidate generator + LLM classifier bounded to candidate ids (V3) | Precision held; recall improved structurally. |
| 10 | Bio uses "Amy Vivian Coney"; canonical 'Amy Coney Barrett' not matched | Source-local alias extractor registered per-source during `_wiki_read`, threaded into `_find_mention` and judge | Birth names match without polluting global identity. |
| 11 | "appointed by Trump" failed when source said "nominated by"; tables lack "Trump" | Expanded `RELATION_SCHEMAS["appointed_by"]` with nomination cues + object aliases; executor uses schema cues as `wiki_windows_around` search terms | Relation-targeted acquisition. |
| 12 | Disallowed-cue check direction bug ("attended" matched "law school attended") | One-directional substring check; checks both predicate and object | `attended` no longer self-rejects. |
| 13 | Both Kavanaugh and Barrett satisfied father-Michael (false positive let through) | Uniqueness invariant: `>1 satisfying candidates → escalate, not arbitrary pick` | Wrong upstream value cannot propagate. |
| 14 | Bare "Michael" attaches to wrong person; "Michael Kavanaugh" carved from "Brett Michael Kavanaugh" | Full-name extraction (`_extract_person_name_starting_with`) + avoid-self filter; body-coordinate disallowed-cue check | Structural prevention of bare-first-name attachment. |
| 15 | LLM acquisition agent found the right fact but failed packaging | Graph-backed `resolve_relation_follow_slot` + graph salvage (re-scan graph after child fails) | Graph as active proof substrate. |
| 16 | `wiki_windows_around` excerpts ending mid-name ("Professor Ma") accepted as founder | Boundary-truncation guard `_is_truncated_at_boundary` + source-text threading to extractors | Partial-name false positives rejected only when source proves the cut. |
| 17 | Last-word object aliases ("Marsh") satisfying obligations for canonical full names ("Samuel Marsh") via adjacent text | Role-sensitive `_mention_candidates(role)` with stricter subject-vs-object policies | Wrong-person aliases blocked at candidate generation. |
| 18 | Role-insensitive mention resolution let stale role policies pass | Per-role mention policy threaded through judge and validator | Role-sensitive validation throughout. |
| 19 | LLM acquisition child failed to package even when accepted graph fact existed | Locked-down graph-salvage path in synthetic regression suite | Salvage now a tested invariant. |
| 20 | Founder slot's relation_follow only matched subject overlap; reverse-direction triples invisible | `value_position: "either"` on slot spec; resolver checks both sides | Forward and reverse founder triples both certify. |
| 21 | Uniqueness invariant treated "Marcus Pohlmann" and "Professor Marcus Pohlmann" as distinct | `value_canonicalization: "person"` + `normalize_person_value(s)` strips honorifics for uniqueness comparison | Honorific variants collapse to one identity. |
| 22 | Role-sensitive object policy too strict for compound entities ("mock trial program" vs source "mock trial"/"the program") | `RELATION_SCHEMAS["founded"].compound_object_aliases` + `_propose_triples` consults schema aliases | Compound-entity objects ground without weakening person-object strictness. |

22 named failure classes addressed by 22 narrow primitives, each protected by at least one regression test. One additional failure class remains open: **23. Founder-slot LLM acquisition variance** — 5/10 v1.2 runs still escalate at the founder slot even with all 22 named transitions fixed. This is the class the next phase of work should target.

The discipline of the loop: never add a primitive proactively. Add it only when a specific failure of the previous round demands it. The result is 22 primitives, each tracked back to a specific failure mode, each protected by a regression test, with no "feature theater."

---

## Appendix D. Glossary

| Term | Meaning in this artifact |
|---|---|
| Proof tree | A structured execution trace in which each task node records attempts, children, verdicts, and slot dataflow. |
| Stage 0 sanity | Runtime-level verification that catches protocol failures before domain verification, such as no `finish()` call, missing output, or a model self-reporting `failed`/`missing`. |
| Repair policy | The bounded retry policy that determines whether a rejected attempt receives a repair hint and another attempt, or escalates. |
| Slot | A typed intermediate value such as `justice`, `college`, or `founder`. |
| Slot certificate | Evidence that a slot value satisfies the slot’s proof obligations. |
| Graph salvage | Mechanical certification of a slot from already accepted graph facts after a child fails to package the slot. |
| `EntityConstraintResolutionExecutor` | Mechanical executor for constrained entity slots: enumerate candidates, fill obligations for each candidate, filter by constraints, and certify exactly one satisfying candidate. |
| `relation_follow` | A slot executor that follows an accepted relation from a consumed slot value to a produced slot value. |
| `RELATION_SCHEMAS` | The relation configuration surface for allowed cues, disallowed cues, object aliases, compound-object aliases, and relation direction. |
| `value_position` | A `relation_follow` setting that tells the resolver whether the produced slot value appears in the subject, object, or either side of a matching triple. |
| `compound_object_aliases` | Relation-schema aliases allowing compound objects such as `mock trial program` to be grounded by source mentions such as `mock trial`, `the program`, or `program` when context supports it. |
| Source-local alias | A mention form valid only within a specific source, such as `Amy Vivian Coney` on Amy Coney Barrett’s page. |
| Truncation guard | A verifier/extractor guard that rejects partial names when the source text proves that a candidate name was cut off by a window boundary. |
| Contract elasticity | The ability for tasks to range from bare goals to fully typed and verified contracts, from level 0 goal-only tasks through level 4 slot/dataflow/proof-obligation tasks. |
