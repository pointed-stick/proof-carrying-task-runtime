"""Atomic-question knowledge-QA demo — the other half of the universality test.

If the same runtime that ran slugify can also run an evidence-grounded factual
question with a totally different plugin (knowledge_qa instead of code_tests),
the universality claim has its second supporting data point.

Workflow the runtime drives:
  goal → wiki_search → wiki_read → add_triple(...) → finish(answer, triple_ids)

The verifier (claim_evidence_alignment) re-checks at the end that every cited
triple id exists. Each triple was validated at add_triple time against its
excerpt's subject+object containment.

Run:
  python -m task_runtime.examples.atomic_qa_demo
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import Budget, RepairPolicy, TaskContract  # noqa: E402
from task_runtime.plugins.knowledge_qa import KnowledgeQAPlugin  # noqa: E402
from task_runtime.runtime import run_task  # noqa: E402


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
    _load_api_key()
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set (and no API_KEY= in .env).", file=sys.stderr)
        return 2

    plugin = KnowledgeQAPlugin()
    contract = TaskContract(
        goal="Who founded the mock trial program at Rhodes College?",
        output_schema={
            "type": "object",
            "required": ["answer", "supporting_triple_ids"],
            "properties": {
                "answer": {"type": "string"},
                "supporting_triple_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        verifier="claim_evidence_alignment",
        success_criteria=[
            "answer names a specific person",
            "at least one supporting triple id is cited",
            "each cited triple's excerpt contains both subject and object",
        ],
        budget=Budget(max_llm_calls=15, max_children=0, max_depth=1),
        repair_policy=RepairPolicy(enabled=True, max_attempts=3),
    )

    print(f"Goal: {contract.goal}")
    print(f"Contract level: {contract.level}  (verifier={contract.verifier})")
    print()
    proof = run_task(contract, plugin)

    # Dump proof tree alongside the demo file so it's inspectable.
    out_dir = Path(__file__).resolve().parent / "_atomic_qa_output"
    out_dir.mkdir(exist_ok=True)
    proof_path = out_dir / "proof_tree.json"
    proof_path.write_text(
        json.dumps(proof.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    triples_path = out_dir / "graph_triples.json"
    triples_path.write_text(
        json.dumps([t.to_dict() for t in plugin.triples], indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print(f"final verdict: {type(proof.final_verdict).__name__ if proof.final_verdict else 'None'}")
    print(f"attempts:      {len(proof.attempts)}")
    print(f"graph size:    {len(plugin.triples)} triple(s)")
    print(f"proof tree:    {proof_path}")
    print(f"triples:       {triples_path}")
    print()
    if proof.final_result and proof.final_result.output:
        out = proof.final_result.output
        if isinstance(out, dict):
            print(f"answer:        {out.get('answer')!r}")
            print(f"cited triples: {out.get('supporting_triple_ids')}")
    if not proof.accepted and proof.final_verdict:
        print()
        print(f"reason: {getattr(proof.final_verdict, 'reason', '')}")

    return 0 if proof.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
