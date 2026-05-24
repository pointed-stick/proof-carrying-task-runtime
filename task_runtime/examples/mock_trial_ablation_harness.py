"""Ablation harness for mock_trial_v1.

Disables one mechanism at a time and re-runs the full chain. Each ablation
should reproduce a SPECIFIC historical failure (see RESEARCH_REPORT §6.3
for the prediction table).

Usage:
  python -m task_runtime.examples.mock_trial_ablation_harness \\
      --runs 1 --ablations A0,A1,A2,A4

Available ablations:
  A0  full system (baseline)
  A1  no graph-salvage
  A2  no source-local aliases
  A4  no full-name father extraction
  A5  no uniqueness invariant
  A6  no avoid-self extractor
  A8  no relation_follow executor (LLM packaging only)

(A3, A7 require deeper surgery and are left as TODOs.)

The harness writes per-ablation dumps to _mock_trial_ablation_output/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.plugins import knowledge_qa as kqa  # noqa: E402
from task_runtime.examples import mock_trial_multihop_demo as demo  # noqa: E402


# -----------------------------------------------------------------------------
# Context managers that temporarily disable a single mechanism.
# -----------------------------------------------------------------------------

@contextmanager
def ablation(name: str):
    """Activate one ablation by temporarily patching the relevant module(s)."""
    if name == "A0":
        yield "full system (baseline)"
        return

    if name == "A1":
        # No graph-salvage: relation_follow becomes "graph-first only";
        # after LLM acquisition, skip the second relation_follow scan.
        original_chain = demo.chain_executor
        # We patch by monkey-patching resolve_relation_follow_slot to always
        # return no-match on the second call from the chain executor. The
        # simplest concrete patch: replace the function with one that always
        # returns empty.
        from task_runtime.plugins import knowledge_qa as _kqa
        original_rf = _kqa.resolve_relation_follow_slot

        def disabled_rf(plugin, slot_name, slot_spec, consumed_slots):
            return {
                "slot_name": slot_name, "value": "",
                "method": "relation_follow", "supporting_triple_ids": [],
                "cited_obligation_triples": {}, "candidate_table": [],
                "selection_reason": "ABLATION A1: relation_follow disabled",
                "ambiguous": False, "satisfying_count": 0,
            }
        _kqa.resolve_relation_follow_slot = disabled_rf
        # Also patch in the demo module since it imported the symbol.
        demo.chain_executor.__globals__["resolve_relation_follow_slot"] = disabled_rf
        try:
            yield "no graph-salvage (relation_follow returns nothing)"
        finally:
            _kqa.resolve_relation_follow_slot = original_rf
        return

    if name == "A2":
        # No source-local aliases: extract_source_local_aliases returns empty.
        original = kqa.extract_source_local_aliases
        kqa.extract_source_local_aliases = lambda body, canonical: set()
        try:
            yield "no source-local aliases"
        finally:
            kqa.extract_source_local_aliases = original
        return

    if name == "A4":
        # No full-name father extraction: the extractor returns None for
        # any first_word — forcing the executor to fall back to bare
        # object_target matching (which doesn't exist for object_first_word
        # obligations, so the obligation effectively can't be satisfied).
        original = kqa._extract_person_name_avoiding_self
        kqa._extract_person_name_avoiding_self = (
            lambda text, first_word, exclude_alias_spans=None,
            source_text=None: None
        )
        try:
            yield "no full-name father extraction (returns None)"
        finally:
            kqa._extract_person_name_avoiding_self = original
        return

    if name == "A5":
        # No uniqueness invariant: pick the first satisfying candidate even
        # if multiple satisfy. Patches resolve_entity_constraint_slot.
        original = kqa.resolve_entity_constraint_slot

        def no_uniqueness(plugin, slot_name, slot_spec, candidates):
            result = original(plugin, slot_name, slot_spec, candidates)
            if not result["value"] and result.get("ambiguous"):
                # Pick the first satisfying candidate from the table.
                satisfying = [
                    r for r in result["candidate_table"] if r.get("satisfies")
                ]
                if satisfying:
                    chosen = satisfying[0]
                    result["value"] = chosen["candidate"]
                    result["ambiguous"] = False
                    result["selection_reason"] = (
                        "ABLATION A5: uniqueness invariant disabled; "
                        f"arbitrarily picked first of {[r['candidate'] for r in satisfying]}"
                    )
            return result
        kqa.resolve_entity_constraint_slot = no_uniqueness
        try:
            yield "no uniqueness invariant (first-satisfying pick)"
        finally:
            kqa.resolve_entity_constraint_slot = original
        return

    if name == "A6":
        # No avoid-self filter: extractor uses the basic person-name
        # function which allows self-name carving.
        original = kqa._extract_person_name_avoiding_self

        def naive(text, first_word, exclude_alias_spans=None,
                  source_text=None):
            # Ignore exclude_alias_spans entirely; preserve truncation guard.
            return kqa._extract_person_name_starting_with(
                text, first_word, source_text=source_text,
            )

        kqa._extract_person_name_avoiding_self = naive
        try:
            yield "no avoid-self extractor"
        finally:
            kqa._extract_person_name_avoiding_self = original
        return

    if name == "A8":
        # No relation_follow executor for downstream slots: chain executor
        # falls through to LLM packaging only. Patch by making
        # resolve_relation_follow_slot always say no match, AND the demo's
        # chain executor would then need to skip route 2... but route 2
        # checks for "operation": "relation_follow" first. Simplest patch:
        # temporarily remove "operation": "relation_follow" from SLOT_SPECS.
        original_specs = demo.SLOT_SPECS
        modified = {}
        for k, v in original_specs.items():
            v2 = dict(v)
            v2.pop("operation", None)
            modified[k] = v2
        demo.SLOT_SPECS = modified
        try:
            yield "no relation_follow executor (LLM packaging only)"
        finally:
            demo.SLOT_SPECS = original_specs
        return

    raise ValueError(f"unknown ablation: {name!r}")


# -----------------------------------------------------------------------------
# Per-ablation runner
# -----------------------------------------------------------------------------

def run_one(ablation_name: str, run_id: int, out_root: Path) -> dict:
    from task_runtime.plugins.knowledge_qa import KnowledgeQAPlugin
    plugin = KnowledgeQAPlugin()
    dump_dir = out_root / ablation_name / f"run_{run_id}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ablation(ablation_name) as description:
            proof = demo.chain_executor(plugin)
    except Exception as e:  # noqa: BLE001
        return {
            "ablation": ablation_name, "run_id": run_id,
            "description": description if 'description' in locals() else "?",
            "error": f"{type(e).__name__}: {e}",
        }

    answer = ""
    if proof.final_result and isinstance(proof.final_result.output, dict):
        answer = str(proof.final_result.output.get("answer", ""))
    correct = "pohlmann" in answer.lower()

    # Save proof tree + graph
    (dump_dir / "proof_tree.json").write_text(
        json.dumps(proof.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (dump_dir / "graph.json").write_text(
        json.dumps([t.to_dict() for t in plugin.triples], indent=2),
        encoding="utf-8"
    )

    # Cost / structure metrics
    total_llm = total_tool = total_attempts = 0
    def walk(n):
        nonlocal total_llm, total_tool, total_attempts
        for a in n.attempts:
            total_attempts += 1
            total_llm += a.cost.get("llm_calls", 0)
            total_tool += a.cost.get("tool_calls", 0)
        for c in n.children:
            walk(c)
    walk(proof)

    return {
        "ablation": ablation_name,
        "description": description,
        "run_id": run_id,
        "verdict_kind": (
            type(proof.final_verdict).__name__ if proof.final_verdict else "None"
        ),
        "answer": answer,
        "correct": correct,
        "slot_table": dict(proof.slot_table),
        "llm_calls": total_llm,
        "tool_calls": total_tool,
        "attempts": total_attempts,
        "graph_size": len(plugin.triples),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--ablations", default="A0,A2,A4,A5,A6",
        help=("Comma-separated subset of {A0,A1,A2,A4,A5,A6,A8}. "
              "Default skips the most expensive ones (A1, A8) to keep cost down."),
    )
    args = parser.parse_args()
    _load_api_key()
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    out_root = Path(__file__).resolve().parent / "_mock_trial_ablation_output"
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir()

    results = []
    for ablation_name in args.ablations.split(","):
        ablation_name = ablation_name.strip()
        for run_id in range(1, args.runs + 1):
            print(f"\n=== {ablation_name} run {run_id} ===")
            r = run_one(ablation_name, run_id, out_root)
            results.append(r)
            err = r.get("error")
            if err:
                print(f"  EXCEPTION: {err}")
                continue
            print(f"  {r['description']}")
            print(f"  verdict: {r['verdict_kind']}")
            print(f"  answer: {r['answer']!r}")
            print(f"  correct: {'Y' if r['correct'] else 'n'}")
            print(f"  slot_table: {r['slot_table']}")
            print(f"  cost: {r['llm_calls']} llm, {r['tool_calls']} tool, "
                  f"{r['attempts']} attempts; {r['graph_size']} triples")

    (out_root / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Summary
    print()
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'ablation':<5} {'desc':<40} {'correct':>8} {'llm':>5} {'graph':>5}")
    for r in results:
        if r.get("error"):
            print(f"{r['ablation']:<5} {'(error)':<40}")
            continue
        print(f"{r['ablation']:<5} {r['description'][:40]:<40} "
              f"{'Y' if r['correct'] else 'n':>8} "
              f"{r['llm_calls']:>5} {r['graph_size']:>5}")
    print()
    print(f"Dumps: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
