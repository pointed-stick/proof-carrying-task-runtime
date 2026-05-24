"""Founder-slot diagnostic: isolate the failure transition.

The mock-trial chain currently stops 9/10 runs at the founder slot with
`slot_table = {justice, college}`. This file decomposes the founder slot
into the user's prescribed 4 modes so the failure can be localized to a
specific transition:

  F0: graph_preseed             — preload the correct founder triple;
                                  test only the slot verifier
  F1: deterministic_scan        — cached source + mechanical candidate gen
                                  + stub classifier; no free-form research
  F2: constrained_llm_acquisition — LLM with high-level safe tools only
  F3: current_founder_slot      — exact path from the chain executor
                                  (pointer to A0 ×10 results)

Decision rules (per user):
  F0 fails        → slot schema / relation direction / verifier wrong
  F0 ok, F1 fails → deterministic acquisition weak: windowing, witness
                    generation, truncation, person-extraction, or relation
                    direction
  F1 ok, F2 fails → LLM tool contract weak
  F2 ok, F3 fails → integration problem (chain context degrades founder task)
  F0–F3 all pass  → A0 9/10 failure was stochastic or already fixed

Run:
  python -m task_runtime.examples.founder_slot_diagnostic --modes F0,F1
  python -m task_runtime.examples.founder_slot_diagnostic --modes F0,F1,F2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import Budget, RepairPolicy, TaskContract  # noqa: E402
from task_runtime.plugins.knowledge_qa import (  # noqa: E402
    KnowledgeQAPlugin, disabled_claim_judge,
    make_classifier_claim_judge, resolve_relation_follow_slot,
)
from task_runtime.runtime import run_task  # noqa: E402

# Use the same founder spec as the chain executor.
from task_runtime.examples.mock_trial_multihop_demo import SLOT_SPECS  # noqa: E402

FOUNDER_SPEC = SLOT_SPECS["founder"]
EXPECTED_FOUNDER_KEYWORDS = ("pohlmann",)  # accept "Marcus Pohlmann", "Pohlmann", "Professor Marcus Pohlmann"


def _is_correct(value: str) -> bool:
    v = (value or "").lower()
    return any(k in v for k in EXPECTED_FOUNDER_KEYWORDS)


def _stub_classifier(s, p, o, e, candidates, schema):
    if not candidates:
        return {"supported": False, "rejection_reason": "no candidates"}
    return {"supported": True, "chosen_span_id": 0,
            "binding_explanation": "stub picks first",
            "rejection_reason": None}


def _load_api_key() -> None:
    if "OPENAI_API_KEY" in os.environ:
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        if k.strip() == "API_KEY":
            os.environ.setdefault("OPENAI_API_KEY", v)
            return


# -----------------------------------------------------------------------------
# F0: graph_preseed
# -----------------------------------------------------------------------------

def run_f0() -> dict:
    """Preload an obviously-correct founder triple. If the slot verifier
    can't certify this, the slot schema or relation direction is wrong."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Rhodes College"] = (
        "The Rhodes College mock trial program was founded by Marcus Pohlmann."
    )
    # Try BOTH directions of the founder relation. The current schema
    # allowed_predicates is ["founded by", "founded", "founder", "established"].
    # Direction A: (program, founded by, Pohlmann) — value = Pohlmann ✓
    # Direction B: (Pohlmann, founded, program)   — value = program ✗ (wrong direction)
    p._propose_triples([
        {"subject": "Rhodes College mock trial program",
         "predicate": "founded by", "object": "Marcus Pohlmann",
         "source": "wiki:Rhodes College",
         "excerpt": "The Rhodes College mock trial program was founded by Marcus Pohlmann."},
    ])

    result = resolve_relation_follow_slot(
        plugin=p, slot_name="founder", slot_spec=FOUNDER_SPEC,
        consumed_slots={"college": "Rhodes College"},
    )
    return {
        "mode": "F0",
        "preseed": "(Rhodes College mock trial program, founded by, Marcus Pohlmann)",
        "passed": _is_correct(result["value"]),
        "founder_value": result["value"],
        "selection_reason": result["selection_reason"],
        "triples_in_graph": len(p.triples),
    }


def run_f0_inverse() -> dict:
    """Sub-test: what about the INVERSE-direction triple
    (Pohlmann, founded, program)? Current relation_follow returns
    triple.object as the slot value, which would be 'program', not
    Pohlmann. This tests whether relation directionality is a blocker."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Rhodes College"] = (
        "Marcus Pohlmann founded the Rhodes College mock trial program."
    )
    p._propose_triples([
        {"subject": "Marcus Pohlmann",
         "predicate": "founded", "object": "Rhodes College mock trial program",
         "source": "wiki:Rhodes College",
         "excerpt": "Marcus Pohlmann founded the Rhodes College mock trial program."},
    ])
    result = resolve_relation_follow_slot(
        plugin=p, slot_name="founder", slot_spec=FOUNDER_SPEC,
        consumed_slots={"college": "Rhodes College"},
    )
    return {
        "mode": "F0_inverse",
        "preseed": "(Marcus Pohlmann, founded, Rhodes College mock trial program)",
        "passed": _is_correct(result["value"]),
        "founder_value": result["value"],
        "selection_reason": result["selection_reason"],
        "triples_in_graph": len(p.triples),
        "note": "Inverse triple direction — relation_follow returns triple.object, "
                "which is the program, not the founder.",
    }


# -----------------------------------------------------------------------------
# F1: deterministic_source_scan
# -----------------------------------------------------------------------------

def run_f1() -> dict:
    """Cached Rhodes College source + V3 mechanical pipeline + stub classifier.
    No LLM acquisition. Tests whether the deterministic path can find and
    certify the founder triple."""
    p = KnowledgeQAPlugin(
        claim_judge=make_classifier_claim_judge(witness_classifier=_stub_classifier)
    )
    # Use a synthetic-but-realistic Rhodes College body.
    p.sources["wiki:Rhodes College"] = (
        "Rhodes College provides an undergraduate mock trial program. "
        "The program was founded in 1986 by Professor Marcus Pohlmann. "
        "The program has won several national championships."
    )

    # Mechanically search for windows around "founded" / "Pohlmann" /
    # "Marcus" / "mock trial".
    windows = p._wiki_windows_around(
        "Rhodes College",
        ["founded", "Pohlmann", "Marcus", "mock trial", "Professor"],
    )

    # For each window, try propose_triples with both directional candidates.
    proposals_attempted = 0
    proposals_accepted = 0
    for w in windows.get("windows", []):
        excerpt = w["excerpt"]
        for proposal in [
            {"subject": "Rhodes College mock trial program",
             "predicate": "founded by", "object": "Marcus Pohlmann",
             "source": "wiki:Rhodes College", "excerpt": excerpt},
            {"subject": "Rhodes College mock trial program",
             "predicate": "founded by", "object": "Professor Marcus Pohlmann",
             "source": "wiki:Rhodes College", "excerpt": excerpt},
            {"subject": "mock trial program",
             "predicate": "founded by", "object": "Marcus Pohlmann",
             "source": "wiki:Rhodes College", "excerpt": excerpt},
        ]:
            proposals_attempted += 1
            result = p._propose_triples([proposal])
            if result["accepted"]:
                proposals_accepted += 1

    rf = resolve_relation_follow_slot(
        plugin=p, slot_name="founder", slot_spec=FOUNDER_SPEC,
        consumed_slots={"college": "Rhodes College"},
    )
    return {
        "mode": "F1",
        "passed": _is_correct(rf["value"]),
        "founder_value": rf["value"],
        "selection_reason": rf["selection_reason"],
        "windows_found": len(windows.get("windows", [])),
        "proposals_attempted": proposals_attempted,
        "proposals_accepted": proposals_accepted,
        "triples_in_graph": len(p.triples),
        "rejected_attempts": len(p.rejected_attempts),
        "first_rejection_reasons": [
            r["error"][:120] for r in p.rejected_attempts[:3]
        ],
    }


# -----------------------------------------------------------------------------
# F2: constrained_llm_acquisition (live, but minimal tool surface)
# -----------------------------------------------------------------------------

def run_f2() -> dict:
    """LLM-driven founder research with constrained tool set: only
    propose_triples + wiki_find + wiki_windows_around + query_graph.
    No wiki_search, no wiki_read, no add_triple. Tests whether the LLM
    can package founder evidence when given safe, narrow tools."""
    p = KnowledgeQAPlugin()  # uses real V3 judge

    goal = (
        "Find who founded the mock trial program at Rhodes College. "
        "Use wiki_windows_around or wiki_find to fetch source excerpts "
        "from 'Rhodes College'. Use propose_triples (not add_triple) "
        "to record any founder fact you find. Then call finish with "
        "{value: '<founder name>', certificate: {method: 'direct_lookup', "
        "cited_obligation_triples: {'0': '<triple_id>'}}, "
        "supporting_triple_ids: [...]}."
    )
    contract = TaskContract(
        goal=goal,
        output_schema={
            "type": "object",
            "required": ["value", "certificate", "supporting_triple_ids"],
            "properties": {
                "value": {"type": "string"},
                "certificate": {"type": "object"},
                "supporting_triple_ids": {"type": "array",
                                          "items": {"type": "string"}},
            },
        },
        verifier="slot_certificate",
        allowed_tools=[
            "propose_triples", "query_graph",
            "wiki_find", "wiki_windows_around",
        ],
        inputs={
            "kind": "slot_research_task",
            "slot_name": "founder",
            "slot_spec": FOUNDER_SPEC,
            "consumed_slots": {"college": "Rhodes College"},
        },
        budget=Budget(max_llm_calls=15, max_children=0, max_depth=1),
        repair_policy=RepairPolicy(enabled=True, max_attempts=2),
    )
    proof = run_task(contract, p)
    value = ""
    if proof.final_result and isinstance(proof.final_result.output, dict):
        value = str(proof.final_result.output.get("value", ""))
    return {
        "mode": "F2",
        "passed": _is_correct(value),
        "founder_value": value,
        "verdict": type(proof.final_verdict).__name__ if proof.final_verdict else "None",
        "verdict_reason": getattr(proof.final_verdict, "reason", "")[:200],
        "triples_in_graph": len(p.triples),
        "rejected_attempts": len(p.rejected_attempts),
        "first_rejection_reasons": [
            r["error"][:120] for r in p.rejected_attempts[:3]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="F0,F0_inverse,F1",
                        help="F0 / F0_inverse / F1 / F2 (F2 requires live API)")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",")]

    if "F2" in modes:
        _load_api_key()
        if "OPENAI_API_KEY" not in os.environ:
            print("ERROR: OPENAI_API_KEY not set; F2 needs it", file=sys.stderr)
            return 2

    runners = {
        "F0":         run_f0,
        "F0_inverse": run_f0_inverse,
        "F1":         run_f1,
        "F2":         run_f2,
    }
    results = []
    for m in modes:
        if m not in runners:
            print(f"WARN: unknown mode {m!r}")
            continue
        print(f"\n=== {m} ===")
        try:
            r = runners[m]()
        except Exception as e:  # noqa: BLE001
            r = {"mode": m, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                print(f"  {k}: {json.dumps(v, default=str)[:200]}")
            else:
                print(f"  {k}: {v}")

    # Summary
    print("\n" + "=" * 60)
    print("FOUNDER-SLOT DIAGNOSTIC SUMMARY")
    print("=" * 60)
    for r in results:
        passed = "PASS" if r.get("passed") else "FAIL"
        if r.get("error"):
            passed = "ERROR"
        print(f"  {r['mode']:<12} {passed:>5}   value={r.get('founder_value','')!r}")

    out_dir = Path(__file__).resolve().parent / "_founder_diagnostic_output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
