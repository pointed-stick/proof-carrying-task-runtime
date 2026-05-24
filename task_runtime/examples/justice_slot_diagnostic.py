"""Focused diagnostic for the `justice` slot — the live bottleneck.

The previous mock-trial run got stuck on this slot: 75 LLM calls, 180 tool
calls, 3 attempts, only 1 triple survived validation. The hypothesis was
that the bottleneck was evidence-acquisition cost, not certification logic.

This diagnostic isolates the justice slot and compares two acquisition
modes:

  baseline:   add_triple + wiki_search + wiki_read + wiki_find
              (the model must quote excerpts from memory/context, often
              failing the excerpt-in-source check)

  scaffolded: also has wiki_windows_around + propose_triples
              (excerpts are deterministically extracted from sources; the
              model batches triple submissions instead of one-by-one
              shepherding through the validator)

Success criteria (from the user's spec):
  1. accepted justice slot certificate
  2. selected value contains "Barrett"
  3. candidate table includes Gorsuch, Kavanaugh, Barrett
  4. Kavanaugh middle-name evidence, if discovered, does NOT satisfy the slot
  5. LLM/tool cost is lower than the 75/180 failed run

Run:
  python -m task_runtime.examples.justice_slot_diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import Accept, Budget, RepairPolicy, TaskContract  # noqa: E402
from task_runtime.plugins.knowledge_qa import (  # noqa: E402
    KnowledgeQAPlugin, resolve_entity_constraint_slot,
)
from task_runtime.runtime import run_task  # noqa: E402


JUSTICE_SPEC = {
    "name": "justice",
    "description": ("Trump-appointed Supreme Court justice whose father's "
                    "first name is Michael"),
    "proof_obligations": [
        {"predicate_contains": "appointed", "object_contains": "Trump",
         "description": "value was appointed by Donald Trump"},
        {"predicate_contains": "father", "object_contains": "Michael",
         "description": "value's father is named Michael"},
    ],
    "disallowed_predicates": ["middle name", "first name of"],
    "preferred_method": "enumerate_filter",
}

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["value", "certificate", "supporting_triple_ids"],
    "properties": {
        "value": {"type": "string"},
        "certificate": {
            "type": "object",
            "required": ["method", "cited_obligation_triples"],
            "properties": {
                "method": {"type": "string"},
                "cited_obligation_triples": {"type": "object"},
                "candidate_table": {"type": "array"},
                "search_ledger": {"type": "array"},
            },
        },
        "supporting_triple_ids": {"type": "array", "items": {"type": "string"}},
    },
}


def _contract_for(mode: str) -> TaskContract:
    baseline_tools = ["add_triple", "query_graph",
                      "wiki_search", "wiki_read", "wiki_find"]
    scaffolded_tools = baseline_tools + ["propose_triples", "wiki_windows_around"]
    return TaskContract(
        goal=(
            "Resolve the 'justice' slot for the mock-trial chain. See the "
            "system prompt for the proof obligations, disallowed predicates, "
            "and required certificate shape."
        ),
        output_schema=OUTPUT_SCHEMA,
        verifier="slot_certificate",
        inputs={
            "kind": "slot_research_task",
            "slot_name": "justice",
            "slot_spec": JUSTICE_SPEC,
        },
        allowed_tools=(baseline_tools if mode == "baseline" else scaffolded_tools),
        success_criteria=[
            "value names a specific Trump-appointed Supreme Court justice",
            "certificate cites a triple per proof obligation",
            "candidate_table populated (enumerate_filter)",
        ],
        budget=Budget(max_llm_calls=30, max_children=0, max_depth=1),
        repair_policy=RepairPolicy(enabled=True, max_attempts=3),
    )


def score(plugin: KnowledgeQAPlugin, output: dict) -> dict:
    blob = " ".join(
        f"{t.subject} {t.predicate} {t.object} {t.excerpt}"
        for t in plugin.triples
    ).lower()
    cands = ["Gorsuch", "Kavanaugh", "Barrett"]
    seen = [c for c in cands if c.lower() in blob]
    value = str((output or {}).get("value", "") or "")
    table = (output or {}).get("certificate", {}).get("candidate_table") or []
    # Check Kavanaugh-middle-name purity: if any triple has Kavanaugh subject +
    # middle-name predicate, it's fine in the graph; we just want it to NOT
    # be the certifying triple for the answer.
    middle_name_kav = any(
        ("kavanaugh" in t.subject.lower()
         and "middle name" in t.predicate.lower())
        for t in plugin.triples
    )
    return {
        "value_is_barrett": "barrett" in value.lower(),
        "enumeration_seen_in_graph": seen,
        "enum_complete": len(seen) == 3,
        "candidate_table_size": len(table),
        "middle_name_triple_in_graph": middle_name_kav,
    }


def metrics(node) -> dict:
    out = {"llm_calls": 0, "tool_calls": 0, "attempts": 0, "repairs": 0}
    for a in node.attempts:
        out["attempts"] += 1
        out["llm_calls"] += a.cost.get("llm_calls", 0)
        out["tool_calls"] += a.cost.get("tool_calls", 0)
        if a.repair_hint:
            out["repairs"] += 1
    return out


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


def run_one(mode: str, out_dir: Path) -> dict:
    plugin = KnowledgeQAPlugin()
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    if mode == "executor":
        # No LLM at the executor level — only the predicate judge inside
        # propose_triples consumes LLM calls. The model never decides which
        # tool to use because the executor calls propose_triples directly.
        result = resolve_entity_constraint_slot(
            plugin=plugin,
            slot_name="justice",
            slot_spec=JUSTICE_SPEC,
            candidates=["Neil Gorsuch", "Brett Kavanaugh", "Amy Coney Barrett"],
        )
        output = {
            "value": result["value"],
            "certificate": {
                "method": result["method"],
                "cited_obligation_triples": result["cited_obligation_triples"],
                "candidate_table": result["candidate_table"],
            },
            "supporting_triple_ids": result["supporting_triple_ids"],
        }
        verdict_kind = "Accept" if result["value"] else "Escalate"
        # No proof tree — the executor isn't run through run_task.
        (mode_dir / "executor_result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        (mode_dir / "graph.json").write_text(
            json.dumps([t.to_dict() for t in plugin.triples], indent=2),
            encoding="utf-8")
        (mode_dir / "rejected_attempts.json").write_text(
            json.dumps(plugin.rejected_attempts, indent=2), encoding="utf-8")
        s = score(plugin, output)
        # Executor LLM cost = predicate-judge calls. The plugin doesn't expose
        # those directly; approximate as: one judge per accepted+rejected
        # propose_triples submission attempt = len(triples)+len(rejected_attempts).
        approx_judge_calls = len(plugin.triples) + len(plugin.rejected_attempts)
        m = {
            "llm_calls": approx_judge_calls,
            "tool_calls": approx_judge_calls,  # propose_triples calls
            "attempts": 1,
            "repairs": 0,
        }
        return {
            "mode": mode, "verdict_kind": verdict_kind,
            "value": output.get("value", ""),
            "score": s, "metrics": m,
            "graph_size": len(plugin.triples),
            "rejected_count": len(plugin.rejected_attempts),
        }

    contract = _contract_for(mode)
    proof = run_task(contract, plugin)
    output = (proof.final_result.output
              if proof.final_result and isinstance(proof.final_result.output, dict)
              else {})
    s = score(plugin, output)
    m = metrics(proof)
    (mode_dir / "proof_tree.json").write_text(
        json.dumps(proof.to_dict(), indent=2, default=str), encoding="utf-8")
    (mode_dir / "graph.json").write_text(
        json.dumps([t.to_dict() for t in plugin.triples], indent=2),
        encoding="utf-8")
    (mode_dir / "rejected_attempts.json").write_text(
        json.dumps(plugin.rejected_attempts, indent=2), encoding="utf-8")
    return {
        "mode": mode,
        "verdict_kind": (type(proof.final_verdict).__name__
                         if proof.final_verdict else "None"),
        "value": output.get("value", ""),
        "score": s,
        "metrics": m,
        "graph_size": len(plugin.triples),
        "rejected_count": len(plugin.rejected_attempts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", default="executor",
        help=("Comma-separated subset of {baseline, scaffolded, executor}. "
              "Default is the new mechanical executor."),
    )
    args = parser.parse_args()

    _load_api_key()
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set (and no API_KEY= in .env).", file=sys.stderr)
        return 2

    out_dir = Path(__file__).resolve().parent / "_justice_slot_output"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()

    results = []
    for mode in args.modes.split(","):
        mode = mode.strip()
        print()
        print("-" * 70)
        print(f"MODE: {mode}")
        print("-" * 70)
        r = run_one(mode, out_dir)
        results.append(r)
        s = r["score"]
        m = r["metrics"]
        print(f"  verdict:     {r['verdict_kind']}")
        print(f"  value:       {r['value']!r}")
        print(f"  is_barrett:  {'Y' if s['value_is_barrett'] else 'n'}")
        print(f"  enum_seen:   {s['enumeration_seen_in_graph']}  "
              f"({'complete' if s['enum_complete'] else 'incomplete'})")
        print(f"  table_size:  {s['candidate_table_size']}")
        print(f"  middle_name_triple_in_graph: "
              f"{'Y' if s['middle_name_triple_in_graph'] else 'n'}")
        print(f"  cost:        {m['llm_calls']} LLM calls, "
              f"{m['tool_calls']} tool calls, {m['attempts']} attempts, "
              f"{m['repairs']} repairs")
        print(f"  graph size:  {r['graph_size']} accepted triples")
        print(f"  rejections:  {r['rejected_count']} (full ledger in "
              f"{mode}/rejected_attempts.json)")

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'mode':<12} {'verdict':<14} {'barrett':>7} {'enum':>6} "
          f"{'tbl':>4} {'llm':>5} {'tools':>5} {'reject':>6}")
    for r in results:
        s = r["score"]
        m = r["metrics"]
        print(f"{r['mode']:<12} {r['verdict_kind']:<14} "
              f"{'Y' if s['value_is_barrett'] else 'n':>7} "
              f"{len(s['enumeration_seen_in_graph']):>6} "
              f"{s['candidate_table_size']:>4} "
              f"{m['llm_calls']:>5} {m['tool_calls']:>5} "
              f"{r['rejected_count']:>6}")
    print()
    print(f"Dumps: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
