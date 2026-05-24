"""Diagnostic for the SCOTUS-constraint subtask that broke the mock-trial chain.

The full mock-trial demo's child 0 — "Identify the Trump-appointed Supreme
Court justice whose father's first name is Michael" — failed across all
three modes in the previous experiment. This file isolates that subtask
and tests three different research strategies against it:

  direct:    just the question; the model decides the strategy on its own
             (reproduces the failing behavior in the harness)
  oracle:    goal text spells out the enumerate->fill->filter strategy
             AND names the 3 Trump appointees, so research is just
             biographical lookup
  recursive: root spawns one child per Trump appointee asking the child to
             find that justice's father's first name; root then filters

All three use the entity_constraint_resolution contract shape so the plugin
injects the enumerate->fill->filter guidance into the system prompt.

The output schema asks for both `answer` (the selected justice) AND an
optional `candidates` table — so a successful run is independently
auditable: a reader can see Gorsuch / Kavanaugh / Barrett, the father name
the model found for each, and which one was selected.

Run:
  python -m task_runtime.examples.scotus_constraint_diagnostic
  python -m task_runtime.examples.scotus_constraint_diagnostic --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import Accept, Budget, ProofNode, RepairPolicy, TaskContract  # noqa: E402
from task_runtime.plugins.knowledge_qa import KnowledgeQAPlugin  # noqa: E402
from task_runtime.runtime import run_task  # noqa: E402


QUESTION = (
    "Identify the Trump-appointed Supreme Court justice whose father's "
    "first name is Michael."
)

CANDIDATES = ["Gorsuch", "Kavanaugh", "Barrett"]
EXPECTED_ANSWER = "Barrett"  # Amy Coney Barrett; father is Michael Coney.

CONSTRAINT_INPUTS = {
    "kind": "entity_constraint_resolution",
    "entity_class": "Supreme Court justice appointed by Donald Trump",
    "constraints": [{"relation": "father_first_name", "value": "Michael"}],
}

# The output schema asks for the answer + a candidate-table. The candidates
# field is optional in the schema (not in `required`) so a model that gives
# only the answer still passes validation, but the diagnostic scores it
# higher if the table is populated.
DIAGNOSTIC_SCHEMA = {
    "type": "object",
    "required": ["answer", "supporting_triple_ids"],
    "properties": {
        "answer": {"type": "string"},
        "supporting_triple_ids": {"type": "array", "items": {"type": "string"}},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "father_first_name": {"type": "string"},
                    "satisfies": {"type": "boolean"},
                    "triple_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


# -----------------------------------------------------------------------------
# Goal text per variant
# -----------------------------------------------------------------------------

DIRECT_GOAL = (
    QUESTION
    + "\n\nReturn the justice's name as the answer and cite supporting triples."
)

ORACLE_GOAL = (
    QUESTION
    + "\n\nStrategy: enumerate -> fill -> filter."
    + "\n  1. The Trump-appointed Supreme Court justices are: "
    + ", ".join(CANDIDATES) + " (Neil Gorsuch, Brett Kavanaugh, Amy Coney Barrett)."
    + "\n  2. For EACH justice, wiki_read their Wikipedia article and find their"
    + "\n     father's first name. Record a triple (justice, father_first_name, NAME)"
    + "\n     for each candidate, regardless of whether they match the constraint."
    + "\n  3. Select the justice whose father is named Michael and put that name"
    + "\n     in `answer`. Populate the `candidates` field with the full table."
)

RECURSIVE_GOAL = (
    QUESTION
    + "\n\nUse spawn_subtasks to check each candidate in parallel:"
    + "\n  Spawn one child per Trump appointee (Neil Gorsuch, Brett Kavanaugh,"
    + "\n  Amy Coney Barrett). Each child's task is to find that ONE justice's"
    + "\n  father's first name from Wikipedia and write a triple for it."
    + "\n  After all children return, select the candidate whose father is named"
    + "\n  Michael and report that justice as `answer`. Populate `candidates`"
    + "\n  with the table the children produced."
)


# -----------------------------------------------------------------------------
# Score: how complete was the enumeration + did it find the right answer?
# -----------------------------------------------------------------------------

def score_diagnostic(plugin: KnowledgeQAPlugin, answer: str,
                     candidates_field: list | None) -> dict:
    blob = " ".join(
        f"{t.subject} {t.predicate} {t.object} {t.excerpt}" for t in plugin.triples
    ).lower()
    candidates_seen = [c for c in CANDIDATES if c.lower() in blob]
    michael_grounded = ("michael" in blob and EXPECTED_ANSWER.lower() in blob)
    answer_correct = EXPECTED_ANSWER.lower() in (answer or "").lower()

    table_seen = []
    if candidates_field:
        for c in candidates_field:
            if isinstance(c, dict) and c.get("name"):
                table_seen.append(c["name"])

    return {
        "candidates_seen_in_graph": candidates_seen,   # subset of CANDIDATES
        "enumeration_complete": len(candidates_seen) == len(CANDIDATES),
        "michael_grounded": michael_grounded,
        "answer_correct": answer_correct,
        "candidate_table_size": len(table_seen),
    }


def aggregate_metrics(root: ProofNode) -> dict:
    out = {"llm_calls": 0, "tool_calls": 0, "attempts": 0,
           "repairs": 0, "task_nodes": 0, "direct_children": 0}

    def walk(node, depth=0):
        out["task_nodes"] += 1
        for a in node.attempts:
            out["attempts"] += 1
            out["llm_calls"] += a.cost.get("llm_calls", 0)
            out["tool_calls"] += a.cost.get("tool_calls", 0)
            if a.repair_hint:
                out["repairs"] += 1
        for c in node.children:
            walk(c, depth + 1)

    walk(root)
    out["direct_children"] = len(root.children)
    return out


# -----------------------------------------------------------------------------
# Per-run driver
# -----------------------------------------------------------------------------

@dataclass
class RunResult:
    variant: str
    run_id: int
    verdict_kind: str
    answer: str
    score: dict
    metrics: dict
    proof_dir: Path
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "run_id": self.run_id,
            "verdict_kind": self.verdict_kind,
            "answer": self.answer,
            "score": self.score,
            "metrics": self.metrics,
            "proof_dir": str(self.proof_dir),
            "error": self.error,
        }


def _contract_for(variant: str) -> TaskContract:
    goal = {"direct": DIRECT_GOAL, "oracle": ORACLE_GOAL,
            "recursive": RECURSIVE_GOAL}[variant]
    return TaskContract(
        goal=goal,
        output_schema=DIAGNOSTIC_SCHEMA,
        verifier="claim_evidence_alignment",
        inputs=CONSTRAINT_INPUTS,
        budget=Budget(max_llm_calls=20, max_children=5, max_depth=2),
        repair_policy=RepairPolicy(enabled=True, max_attempts=3),
    )


def run_one(variant: str, run_id: int, out_root: Path) -> RunResult:
    plugin = KnowledgeQAPlugin()
    contract = _contract_for(variant)
    proof_dir = out_root / variant / f"run_{run_id}"
    proof_dir.mkdir(parents=True, exist_ok=True)

    try:
        proof = run_task(contract, plugin)
    except Exception as e:  # noqa: BLE001
        return RunResult(
            variant=variant, run_id=run_id, verdict_kind="Exception",
            answer="", score={}, metrics={}, proof_dir=proof_dir,
            error=f"{type(e).__name__}: {e}",
        )

    answer = ""
    candidates = None
    if proof.final_result and isinstance(proof.final_result.output, dict):
        answer = str(proof.final_result.output.get("answer", ""))
        candidates = proof.final_result.output.get("candidates")

    score = score_diagnostic(plugin, answer, candidates)
    metrics = aggregate_metrics(proof)

    (proof_dir / "proof_tree.json").write_text(
        json.dumps(proof.to_dict(), indent=2, default=str), encoding="utf-8")
    (proof_dir / "graph.json").write_text(
        json.dumps([t.to_dict() for t in plugin.triples], indent=2), encoding="utf-8")

    return RunResult(
        variant=variant, run_id=run_id,
        verdict_kind=type(proof.final_verdict).__name__ if proof.final_verdict else "None",
        answer=answer, score=score, metrics=metrics, proof_dir=proof_dir,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def _row(r: RunResult) -> str:
    m, s = r.metrics, r.score
    return (
        f"  run {r.run_id}: verdict={r.verdict_kind:<10} "
        f"correct={'Y' if s.get('answer_correct') else 'n'} "
        f"enum={len(s.get('candidates_seen_in_graph', []))}/{len(CANDIDATES)} "
        f"michael_grounded={'Y' if s.get('michael_grounded') else 'n'} "
        f"table_size={s.get('candidate_table_size', 0)} "
        f"calls={m.get('llm_calls', '?'):>3} "
        f"children={m.get('direct_children', '?')} "
        f"answer={(r.answer or '(none)')[:50]!r}"
    )


def _summary(results: list[RunResult]) -> None:
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'variant':<10} {'n':>3} {'correct':>8} {'enum_avg':>9} "
          f"{'llm_avg':>8} {'children_avg':>13} {'table_avg':>10}")
    for variant in ["direct", "oracle", "recursive"]:
        rs = [r for r in results if r.variant == variant]
        if not rs:
            continue
        n = len(rs)
        correct = sum(1 for r in rs if r.score.get("answer_correct"))
        enum_avg = sum(len(r.score.get("candidates_seen_in_graph", []))
                       for r in rs) / n
        llm_avg = sum(r.metrics.get("llm_calls", 0) for r in rs) / n
        child_avg = sum(r.metrics.get("direct_children", 0) for r in rs) / n
        tbl_avg = sum(r.score.get("candidate_table_size", 0) for r in rs) / n
        print(f"{variant:<10} {n:>3} {correct:>4}/{n:<3} {enum_avg:>9.2f} "
              f"{llm_avg:>8.1f} {child_avg:>13.2f} {tbl_avg:>10.2f}")


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
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per variant (default 1)")
    parser.add_argument("--variants", default="direct,oracle,recursive")
    args = parser.parse_args()

    _load_api_key()
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set (and no API_KEY= in .env).", file=sys.stderr)
        return 2

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_root = Path(__file__).resolve().parent / "_scotus_diagnostic_output"
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True)

    results: list[RunResult] = []
    for variant in variants:
        print()
        print("-" * 78)
        print(f"VARIANT: {variant}  ({args.runs} run(s))")
        print("-" * 78)
        for i in range(1, args.runs + 1):
            print(f"  starting run {i}...")
            r = run_one(variant, i, out_root)
            results.append(r)
            print(_row(r))
            if r.error:
                print(f"    exception: {r.error}")

    _summary(results)
    (out_root / "results.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    print()
    print(f"All proof trees + summary: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
