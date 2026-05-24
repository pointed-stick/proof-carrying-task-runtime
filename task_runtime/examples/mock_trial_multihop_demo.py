"""Multi-hop knowledge-QA experiment harness.

Compares three modes on the mock-trial multi-hop question:

  Q: "Who founded the mock trial program at the college attended by the
      Trump-appointed Supreme Court justice whose father's first name is
      Michael?"
  Expected chain: Amy Coney Barrett -> Rhodes College -> Marcus Pohlmann

Modes:
  flat    — one task does everything; no required decomposition
  oracle  — goal text spells out the 3-step decomposition; LLM executes it
  llm     — just the question; LLM must author its own decomposition

Per-run metrics:
  correct            answer text contains "Marcus Pohlmann"
  path_coverage      0–3, how many of {justice, college, founder} are in the graph
  llm_calls          summed across root + all child attempts
  tool_calls         summed across root + all child attempts
  attempts           total attempts across root + children
  children           direct children of root
  repairs            attempts where a repair hint was fed in
  verdict_kind       final verdict kind on the root proof node

Saves proof trees + graphs to _mock_trial_output/<mode>/run_<n>/ so a reader
can audit any individual run independently of the summary table.

Run:
  python -m task_runtime.examples.mock_trial_multihop_demo
  python -m task_runtime.examples.mock_trial_multihop_demo --runs 5

This is an experiment, not a unit test. The LLM is stochastic. The point is
COMPARATIVE evidence across modes, not single-run success.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import Accept, Budget, ProofNode, RepairPolicy, TaskContract  # noqa: E402
from task_runtime.plugins.knowledge_qa import KnowledgeQAPlugin  # noqa: E402
from task_runtime.runtime import run_task  # noqa: E402


QUESTION = (
    "Who founded the mock trial program at the college attended by the "
    "Trump-appointed Supreme Court justice whose father's first name is "
    "Michael?"
)

# The expected resolution of the chain. Used by the path verifier; NOT given
# to the LLM in any mode.
EXPECTED = {
    "justice": "Barrett",      # Amy Coney Barrett
    "college": "Rhodes",       # Rhodes College
    "founder": "Pohlmann",     # Marcus Pohlmann
}

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["answer", "supporting_triple_ids"],
    "properties": {
        "answer": {"type": "string"},
        "supporting_triple_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

# Output schema for chain modes — adds slot_values which chain_completeness needs.
CHAIN_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["answer", "supporting_triple_ids", "slot_values"],
    "properties": {
        "answer": {"type": "string"},
        "supporting_triple_ids": {"type": "array", "items": {"type": "string"}},
        "slot_values": {
            "type": "object",
            "properties": {
                "justice": {"type": "string"},
                "college": {"type": "string"},
                "founder": {"type": "string"},
            },
        },
    },
}

# The typed chain. Drives both the rendered context (slot/edge guidance) and
# the chain_completeness verifier (must find a triple per edge).
CHAIN_INPUTS = {
    "kind": "multi_hop_chain",
    "question": QUESTION,
    "slots": {
        "justice": {
            "description": "Trump-appointed Supreme Court justice whose father's first name is Michael",
        },
        "college": {
            "description": "undergraduate college attended by that justice (NOT law school)",
            "must_preserve_relation": "undergraduate college attended",
            "not_relation": "law school attended",
        },
        "founder": {
            "description": "person who founded the mock trial program at that college",
        },
    },
    "edges": [
        {"from": "justice", "to": "college", "relation": "attended"},
        {"from": "college", "to": "founder", "relation": "founded"},
    ],
    "final_answer_slot": "founder",
}


# -----------------------------------------------------------------------------
# Goal text per mode — the central experimental control variable
# -----------------------------------------------------------------------------

FLAT_GOAL = QUESTION

ORACLE_GOAL = (
    QUESTION
    + "\n\n"
    + "Decompose this into three sub-tasks via spawn_subtasks, in this order:\n"
    + "  1. Find the Trump-appointed Supreme Court justice whose father's "
    + "first name is Michael. Return the justice's name in your output.\n"
    + "  2. Given that justice's name as input, find the college they "
    + "attended for their undergraduate degree.\n"
    + "  3. Given that college's name as input, find who founded the mock "
    + "trial program there.\n"
    + "After all three sub-tasks complete, finish() with the founder's name "
    + "as the answer, citing supporting_triple_ids drawn from the shared "
    + "graph.\n\n"
    + "IMPORTANT for each child: set verifier='claim_evidence_alignment' "
    + "(this is the only valid verifier name for knowledge_qa). DO NOT put "
    + "natural-language requirements in the verifier field — use the "
    + "success_criteria field for that."
)

LLM_GOAL = (
    QUESTION
    + "\n\n"
    + "If you cannot answer this in a single research pass, you may use "
    + "spawn_subtasks to break it into smaller research questions. Each "
    + "sub-agent shares the same graph, so triples written by children are "
    + "available for citation in your final answer."
)

# Chain modes use the typed multi_hop_chain contract shape so the plugin
# renders the slot/edge structure into the system prompt and the
# chain_completeness verifier can check connectivity.
ORACLE_CHAIN_GOAL = (
    QUESTION
    + "\n\n"
    + "This is a multi-hop chain task. Spawn THREE sub-tasks via "
    + "spawn_subtasks, in order, with explicit dataflow:\n"
    + "  child 1:\n"
    + "    goal: 'Find the Trump-appointed Supreme Court justice whose "
    + "father's first name is Michael.'\n"
    + "    produces: ['justice']\n"
    + "    verifier: 'claim_evidence_alignment'\n"
    + "  child 2:\n"
    + "    goal: 'Find the undergraduate college attended by that justice. "
    + "(NOT their law school.)'\n"
    + "    consumes: ['justice']\n"
    + "    produces: ['college']\n"
    + "    relation_lock: 'undergraduate college attended'\n"
    + "    verifier: 'claim_evidence_alignment'\n"
    + "  child 3:\n"
    + "    goal: 'Find who founded the mock trial program at that college.'\n"
    + "    consumes: ['college']\n"
    + "    produces: ['founder']\n"
    + "    relation_lock: 'founded by'\n"
    + "    verifier: 'claim_evidence_alignment'\n"
    + "\n"
    + "Each child's finish() output MUST include slot_values={<slot>: <value>} "
    + "for its produced slot. The runtime auto-injects consumed slot values "
    + "into the next child's inputs.\n"
    + "\n"
    + "After all 3 children complete, ROOT calls finish with:\n"
    + "  output={\n"
    + "    answer: <the founder's name from slot_values['founder']>,\n"
    + "    supporting_triple_ids: [<every triple cited by any child>],\n"
    + "    slot_values: {justice: '...', college: '...', founder: '...'}\n"
    + "  }"
)

LLM_CHAIN_GOAL = (
    QUESTION
    + "\n\n"
    + "Use spawn_subtasks with the multi_hop_chain primitive (see the slots "
    + "and edges in the system prompt). Each child should declare consumes "
    + "and produces and pick an appropriate relation_lock. The "
    + "chain_completeness verifier will check that every slot is filled and "
    + "every edge is supported by a triple in the graph."
)

LLM_CHAIN_REPAIR_GOAL = (
    LLM_CHAIN_GOAL
    + "\n\n"
    + "PLAN REPAIR: if a child returns missing/failed or its slot_values "
    + "field is empty, RE-SPAWN that child with a different research "
    + "strategy (e.g. enumerate-fill-filter for entity-with-constraint "
    + "questions). Do NOT proceed to dependent children until the parent "
    + "child has produced its required slot value. Only the root may call "
    + "finish; do so only when every slot in the chain is populated."
)

# The fourth mode is "llm + plan repair": the model still authors its own
# decomposition, but the goal text instructs it to *repair* the plan when
# a child returns an empty/failed required output, instead of continuing
# the chain with no input. This tests whether prompt-level plan-repair
# guidance is enough to recover from the failure mode seen in the previous
# experiment (child 0 failed → child 1 took empty input → root reported
# failed honestly but with no answer).
LLM_REPAIR_GOAL = (
    QUESTION
    + "\n\n"
    + "PLAN-LEVEL REPAIR: if a sub-task you spawn returns an EMPTY or FAILED "
    + "required output, do NOT proceed to dependent sub-tasks. Instead "
    + "RE-SPAWN that sub-task with a DIFFERENT research strategy.\n"
    + "  - If 'find the X satisfying constraint Y' returns nothing, try "
    + "    ENUMERATE -> FILL -> FILTER: list all X candidates, look up "
    + "    attribute Y for each, then filter to those satisfying Y.\n"
    + "  - If a dependent sub-task needs an entity name and the parent "
    + "    sub-task didn't produce one, repair the parent first.\n"
    + "Only proceed to the next sub-task once the previous one has produced "
    + "a non-empty required output.\n"
    + "\n"
    + "For the specific bottleneck in this question: identifying the "
    + "Trump-appointed Supreme Court justice whose father is named "
    + "Michael typically fails on direct search. Use enumerate -> fill -> "
    + "filter from the start: enumerate Trump's SCOTUS appointees (Gorsuch, "
    + "Kavanaugh, Barrett), look up each one's father's first name, then "
    + "filter."
)


# -----------------------------------------------------------------------------
# Path verifier — scores graph coverage of the expected chain
# -----------------------------------------------------------------------------

def path_score(plugin: KnowledgeQAPlugin, answer: str) -> dict:
    """How much of the expected multi-hop chain is present in the graph?

    Returns:
      justice_found   any triple mentions Barrett (subject OR object)
      college_found   any triple mentions Rhodes
      founder_found   any triple mentions Pohlmann
      answer_correct  the final answer contains "Pohlmann"
      coverage        sum of the three found-flags (0..3)
    """
    blob = " ".join(
        f"{t.subject} {t.predicate} {t.object} {t.excerpt}" for t in plugin.triples
    ).lower()
    justice_found = EXPECTED["justice"].lower() in blob
    college_found = EXPECTED["college"].lower() in blob
    founder_found = EXPECTED["founder"].lower() in blob
    answer_l = (answer or "").lower()
    return {
        "justice_found": justice_found,
        "college_found": college_found,
        "founder_found": founder_found,
        "answer_correct": EXPECTED["founder"].lower() in answer_l,
        "coverage": sum([justice_found, college_found, founder_found]),
    }


# -----------------------------------------------------------------------------
# Cost / structure metrics from a proof tree
# -----------------------------------------------------------------------------

def aggregate_metrics(root: ProofNode) -> dict:
    """Walk root + all descendants; sum LLM/tool calls; count repairs."""
    out = {
        "llm_calls": 0,
        "tool_calls": 0,
        "attempts": 0,
        "repairs": 0,           # attempts that ran with a repair_hint
        "rejections": 0,        # verdicts that were not Accept
        "task_nodes": 0,
        "max_depth_seen": 0,
    }

    def walk(node: ProofNode, depth: int) -> None:
        out["task_nodes"] += 1
        out["max_depth_seen"] = max(out["max_depth_seen"], depth)
        for a in node.attempts:
            out["attempts"] += 1
            out["llm_calls"] += a.cost.get("llm_calls", 0)
            out["tool_calls"] += a.cost.get("tool_calls", 0)
            if a.repair_hint:
                out["repairs"] += 1
            if a.verdict and type(a.verdict).__name__ != "Accept":
                out["rejections"] += 1
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    out["direct_children"] = len(root.children)
    return out


# -----------------------------------------------------------------------------
# Per-run driver
# -----------------------------------------------------------------------------

@dataclass
class RunResult:
    mode: str
    run_id: int
    verdict_kind: str
    answer: str
    path: dict
    metrics: dict
    proof_dir: Path
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "run_id": self.run_id,
            "verdict_kind": self.verdict_kind,
            "answer": self.answer,
            "path": self.path,
            "metrics": self.metrics,
            "proof_dir": str(self.proof_dir),
            "error": self.error,
        }


# Slot specs for the slot-certificate / chain-executor modes. Each slot
# carries proof_obligations (patterns that triples must satisfy to certify
# the value) and disallowed_predicates (predicates that CANNOT certify even
# if individually true — the Kavanaugh-middle-name failure class).
SLOT_SPECS = {
    "justice": {
        "name": "justice",
        "description": ("Trump-appointed Supreme Court justice whose "
                        "father's first name is Michael"),
        "proof_obligations": [
            {"predicate_contains": "appointed", "object_contains": "Trump",
             "description": "value was appointed by Donald Trump"},
            # Full-name extraction: instead of accepting bare 'Michael' as
            # the father, require a 2+ word person name starting with
            # 'Michael' (e.g. 'Michael Coney'). The graph then stores the
            # complete entity, and bare-'Michael' false positives
            # (great-grandfather, middle name, etc.) become structurally
            # impossible.
            {"predicate_contains": "father", "object_first_word": "Michael",
             "description": "value's father is a person whose first name is Michael"},
        ],
        "disallowed_predicates": ["middle name", "first name of"],
        "preferred_method": "enumerate_filter",
        # Enumeration target — the runtime resolves which one satisfies all
        # obligations via resolve_entity_constraint_slot.
        "candidates": ["Neil Gorsuch", "Brett Kavanaugh", "Amy Coney Barrett"],
    },
    "college": {
        "name": "college",
        "description": (
            "UNDERGRADUATE college attended by the selected justice. "
            "NOT law school. NOT graduate school. "
            "For Amy Coney Barrett the correct answer is Rhodes College; "
            "Notre Dame is her law school, not her undergrad."
        ),
        "consumes": ["justice"],
        "subject_slot": "justice",
        # New: operation makes the chain executor route this slot through
        # graph-backed relation_follow rather than purely LLM research.
        # LLM acquires triples; the runtime then mechanically certifies.
        "operation": "relation_follow",
        "allowed_predicates": [
            "attended", "graduated from", "alma mater",
            "undergraduate", "received BA",
        ],
        "disallowed_predicates": [
            "law school attended", "law school",
            "graduate school attended", "graduate school",
            "J.D. from", "law degree from",
            "Juris Doctor", "JD from",
        ],
        "disallowed_objects": [
            "Notre Dame Law School",
            "law school",
        ],
        "uniqueness": "exactly_one",
        # Search hints for the LLM acquisition phase.
        "acquisition_search_terms": [
            "undergraduate", "graduated", "alma mater", "attended", "Rhodes",
        ],
        "proof_obligations": [
            {"predicate_contains": "attended", "object_contains": "",
             "description": "justice attended this college as undergrad"},
        ],
        "preferred_method": "direct_lookup",
    },
    "founder": {
        "name": "founder",
        "description": (
            "person who founded the mock trial program at the selected college"
        ),
        "consumes": ["college"],
        "subject_slot": "college",
        "operation": "relation_follow",
        # Founder triples appear in BOTH directions in real prose:
        #   (program, founded by, founder)  — value_position=object
        #   (founder, founded, program)     — value_position=subject
        # Set 'either' so resolve_relation_follow_slot accepts either shape
        # and yields the founder as the slot value.
        "value_position": "either",
        # Honorific normalization: 'Marcus Pohlmann' and 'Professor Marcus
        # Pohlmann' should resolve to one slot value, not be treated as
        # distinct candidates by the uniqueness invariant.
        "value_canonicalization": "person",
        "subject_aliases": [
            "mock trial program", "mock trial team", "mock trial",
            "program",
        ],
        "allowed_predicates": [
            "founded by", "founded", "founder", "established",
        ],
        "disallowed_predicates": [
            "provides", "offers", "hosts",
        ],
        "uniqueness": "exactly_one",
        "acquisition_search_terms": [
            "mock trial", "founded", "founder", "program",
        ],
        "proof_obligations": [
            {"predicate_contains": "founded", "object_contains": "",
             "description": "value founded the mock trial program"},
        ],
        "preferred_method": "direct_lookup",
    },
}

# Chain definition used by the chain executor. Slot order is the topological
# order; each slot's `consumes` references prior slots.
SLOT_ORDER = ["justice", "college", "founder"]
FINAL_ANSWER_SLOT = "founder"

SLOT_RESEARCH_OUTPUT_SCHEMA = {
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


def chain_executor(plugin, client=None):
    """Mechanically execute the mock-trial slot chain.

    The runtime drives the chain — not the LLM. For each slot in topological
    order, build a slot_research_task contract carrying the slot_spec and
    any consumed_slots from prior resolution, call run_task, and only
    advance if the slot_certificate verifier accepts. Each slot still uses
    an LLM agent for the research itself; the LLM is no longer responsible
    for remembering the chain structure.
    """
    from task_runtime.core import (  # noqa: PLC0415
        Accept, Attempt, Budget, Escalate, ProofNode, RepairPolicy, TaskContract, TaskResult,
    )
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_entity_constraint_slot, resolve_relation_follow_slot,
    )
    from task_runtime.runtime import _default_client, run_task  # noqa: PLC0415
    if client is None:
        client = _default_client()

    # Synthetic root proof node so the children attach somewhere inspectable.
    root_contract = TaskContract(
        goal=f"Chain executor: {QUESTION}",
        inputs={"kind": "chain_executor", "slot_order": SLOT_ORDER,
                "final_answer_slot": FINAL_ANSWER_SLOT},
        budget=Budget(max_llm_calls=0, max_children=len(SLOT_ORDER), max_depth=2),
        repair_policy=RepairPolicy(enabled=False),
    )
    root = ProofNode.new(root_contract)

    resolved: dict = {}
    all_triple_ids: list[str] = []

    for slot_name in SLOT_ORDER:
        spec = SLOT_SPECS[slot_name]
        consumed = {
            s: resolved[s] for s in (spec.get("consumes") or []) if s in resolved
        }

        # Route 1: mechanical EntityConstraintResolutionExecutor for
        # enumerate_filter slots with a known candidate list. This was the
        # successful path for the justice slot in the diagnostic run.
        if (spec.get("preferred_method") == "enumerate_filter"
                and spec.get("candidates")):
            exec_result = resolve_entity_constraint_slot(
                plugin=plugin, slot_name=slot_name, slot_spec=spec,
                candidates=spec["candidates"],
            )
            # Build a synthetic ProofNode for the slot so the proof tree
            # records what happened.
            slot_node = ProofNode.new(
                TaskContract(
                    goal=f"executor: resolve {slot_name!r} slot",
                    inputs={"kind": "executor_slot", "slot_name": slot_name,
                            "consumed_slots": consumed},
                    budget=Budget(max_llm_calls=0, max_children=0, max_depth=1),
                    repair_policy=RepairPolicy(enabled=False),
                ),
                parent_id=root.task_id,
            )
            slot_value = exec_result.get("value", "") or ""
            slot_node.final_result = TaskResult(
                status="success" if slot_value else "missing",
                output={
                    "value": slot_value,
                    "certificate": {
                        "method": exec_result.get("method", ""),
                        "cited_obligation_triples":
                            exec_result.get("cited_obligation_triples", {}),
                        "candidate_table":
                            exec_result.get("candidate_table", []),
                    },
                    "supporting_triple_ids":
                        exec_result.get("supporting_triple_ids", []),
                },
                notes=(f"executor selected {slot_value!r}"
                       if slot_value else "executor found no satisfying candidate"),
            )
            if slot_value:
                slot_node.final_verdict = Accept(
                    reason=f"executor: {slot_name!r} = {slot_value!r}",
                    record={"verifier": "entity_constraint_executor"},
                )
            else:
                slot_node.final_verdict = Escalate(
                    reason=f"executor failed to certify {slot_name!r}"
                )
            root.children.append(slot_node)

            if not slot_value:
                root.final_result = TaskResult(
                    status="missing",
                    output={"resolved": resolved, "stopped_at": slot_name,
                            "all_triple_ids": all_triple_ids},
                    notes=f"chain executor stopped: executor couldn't certify "
                          f"{slot_name!r}",
                )
                root.final_verdict = Escalate(
                    reason=f"chain stopped at slot {slot_name!r}: executor failure"
                )
                return root

            resolved[slot_name] = slot_value
            root.slot_table[slot_name] = slot_value
            all_triple_ids.extend(
                exec_result.get("supporting_triple_ids", []) or []
            )
            continue

        # Route 2: graph-backed relation_follow slots (e.g. college, founder).
        # Acquisition: spawn an LLM research task to populate the graph.
        # Certification: mechanical scan of the graph for triples satisfying
        # the slot's relation pattern. The runtime no longer depends on the
        # LLM child packaging the answer into exactly the right finish()
        # shape — if the graph contains an accepted triple that matches,
        # the slot is certified directly.
        if spec.get("operation") == "relation_follow":
            # First: try the existing graph (cheap — no LLM needed).
            rf_first = resolve_relation_follow_slot(
                plugin=plugin, slot_name=slot_name, slot_spec=spec,
                consumed_slots=consumed,
            )
            if rf_first.get("value") and not rf_first.get("ambiguous"):
                # Already satisfied by existing triples.
                slot_node = ProofNode.new(
                    TaskContract(
                        goal=f"relation_follow: {slot_name!r} from existing graph",
                        inputs={"kind": "relation_follow_slot",
                                "slot_name": slot_name,
                                "consumed_slots": consumed},
                        budget=Budget(max_llm_calls=0, max_children=0, max_depth=1),
                        repair_policy=RepairPolicy(enabled=False),
                    ),
                    parent_id=root.task_id,
                )
                slot_node.final_result = TaskResult(
                    status="success",
                    output={
                        "value": rf_first["value"],
                        "certificate": {
                            "method": "relation_follow",
                            "cited_obligation_triples":
                                rf_first["cited_obligation_triples"],
                            "candidate_table": rf_first["candidate_table"],
                        },
                        "supporting_triple_ids": rf_first["supporting_triple_ids"],
                    },
                    notes=rf_first["selection_reason"],
                )
                slot_node.final_verdict = Accept(
                    reason=f"relation_follow (graph-first): {slot_name!r} = "
                           f"{rf_first['value']!r}",
                    record={"verifier": "relation_follow_graph_first"},
                )
                root.children.append(slot_node)
                resolved[slot_name] = rf_first["value"]
                root.slot_table[slot_name] = rf_first["value"]
                all_triple_ids.extend(rf_first["supporting_triple_ids"])
                continue

            # Otherwise: spawn LLM acquisition, then re-attempt graph-backed
            # certification (graph-salvage pattern).
            search_hints = spec.get("acquisition_search_terms") or []
            acquisition_goal = (
                f"Acquire evidence for the {slot_name!r} slot.\n\n"
                f"{spec['description']}\n\n"
                f"Use propose_triples (not add_triple) so canonical/mention "
                f"resolution runs. Focus your wiki_windows_around searches "
                f"on these terms: {search_hints}. "
                f"Consumed slots already resolved: {consumed}. "
                f"You do NOT need to package the answer perfectly — once you "
                f"have added a relevant accepted triple to the graph, the "
                f"runtime will mechanically certify the slot. Aim to add "
                f"at least one well-supported triple linking {list(consumed.values())} "
                f"via an allowed relation to the next entity."
            )
            acquisition_contract = TaskContract(
                goal=acquisition_goal,
                output_schema=SLOT_RESEARCH_OUTPUT_SCHEMA,
                verifier="slot_certificate",
                allowed_tools=[
                    "propose_triples", "query_graph",
                    "wiki_search", "wiki_read", "wiki_find",
                    "wiki_windows_around",
                ],
                inputs={
                    "kind": "slot_research_task",
                    "slot_name": slot_name,
                    "slot_spec": spec,
                    "consumed_slots": consumed,
                },
                budget=Budget(max_llm_calls=20, max_children=0, max_depth=1),
                repair_policy=RepairPolicy(enabled=True, max_attempts=2),
            )
            child_node = run_task(
                acquisition_contract, plugin,
                parent_id=root.task_id, depth=0, client=client,
            )
            root.children.append(child_node)

            # If the child itself was accepted AND produced a value, use it.
            # The relation_follow salvage is for when the child FAILED — it
            # shouldn't second-guess a successful slot certification (which
            # would re-fail on edge cases like wrong-direction triples for
            # 'founded': the founder is the SUBJECT of that predicate, not
            # the object).
            if child_node.accepted and child_node.final_result:
                child_out = child_node.final_result.output
                if isinstance(child_out, dict) and child_out.get("value"):
                    val = str(child_out["value"])
                    resolved[slot_name] = val
                    root.slot_table[slot_name] = val
                    all_triple_ids.extend(
                        child_out.get("supporting_triple_ids", []) or []
                    )
                    continue

            # Graph-salvage: only after the child failed.
            rf_after = resolve_relation_follow_slot(
                plugin=plugin, slot_name=slot_name, slot_spec=spec,
                consumed_slots=consumed,
            )
            if rf_after.get("value") and not rf_after.get("ambiguous"):
                # Synthesize a salvage node so the proof tree shows the
                # graph-derived certification.
                salvage_node = ProofNode.new(
                    TaskContract(
                        goal=f"relation_follow salvage: {slot_name!r}",
                        inputs={"kind": "relation_follow_salvage",
                                "slot_name": slot_name},
                        budget=Budget(max_llm_calls=0, max_children=0, max_depth=1),
                        repair_policy=RepairPolicy(enabled=False),
                    ),
                    parent_id=root.task_id,
                )
                salvage_node.final_result = TaskResult(
                    status="success",
                    output={
                        "value": rf_after["value"],
                        "certificate": {
                            "method": "relation_follow",
                            "cited_obligation_triples":
                                rf_after["cited_obligation_triples"],
                            "candidate_table": rf_after["candidate_table"],
                        },
                        "supporting_triple_ids": rf_after["supporting_triple_ids"],
                    },
                    notes=f"salvaged after child: {rf_after['selection_reason']}",
                )
                salvage_node.final_verdict = Accept(
                    reason=f"relation_follow (salvage): {slot_name!r} = "
                           f"{rf_after['value']!r}",
                    record={"verifier": "relation_follow_graph_salvage"},
                )
                root.children.append(salvage_node)
                resolved[slot_name] = rf_after["value"]
                root.slot_table[slot_name] = rf_after["value"]
                all_triple_ids.extend(rf_after["supporting_triple_ids"])
                continue

            # Neither graph-first nor salvage worked. Stop.
            root.final_result = TaskResult(
                status="missing",
                output={"resolved": resolved, "stopped_at": slot_name,
                        "all_triple_ids": all_triple_ids},
                notes=(
                    f"chain executor stopped: relation_follow couldn't "
                    f"certify {slot_name!r} before or after acquisition. "
                    f"Reason: {rf_after.get('selection_reason', '')[:200]}"
                ),
            )
            root.final_verdict = Escalate(
                reason=f"chain stopped at slot {slot_name!r}: "
                       f"relation_follow failed; "
                       f"{rf_after.get('selection_reason','')[:160]}"
            )
            return root

        # Route 3: legacy LLM-driven slot_research_task (slots without an
        # operation specified — used as a generic fallback).
        child_contract = TaskContract(
            goal=(
                f"Research the {slot_name!r} slot of the mock-trial chain.\n\n"
                f"{spec['description']}\n\n"
                f"You must return a value AND a certificate proving the value "
                f"satisfies the slot's proof obligations. Use propose_triples "
                f"(NOT add_triple) so canonical/surface-mention resolution "
                f"runs automatically. Cite real triple ids."
            ),
            output_schema=SLOT_RESEARCH_OUTPUT_SCHEMA,
            verifier="slot_certificate",
            allowed_tools=[
                "propose_triples", "query_graph",
                "wiki_search", "wiki_read", "wiki_find", "wiki_windows_around",
            ],
            inputs={
                "kind": "slot_research_task",
                "slot_name": slot_name,
                "slot_spec": spec,
                "consumed_slots": consumed,
            },
            success_criteria=[
                "value is a specific named entity",
                "certificate.cited_obligation_triples covers every obligation",
                "no cited triple uses a disallowed predicate",
            ],
            budget=Budget(max_llm_calls=25, max_children=0, max_depth=1),
            repair_policy=RepairPolicy(enabled=True, max_attempts=3),
        )
        child_node = run_task(
            child_contract, plugin, parent_id=root.task_id, depth=0, client=client,
        )
        root.children.append(child_node)

        if not child_node.accepted:
            root.final_result = TaskResult(
                status="missing",
                output={"resolved": resolved, "stopped_at": slot_name,
                        "all_triple_ids": all_triple_ids},
                notes=(
                    f"chain executor stopped: slot {slot_name!r} not certified "
                    f"({type(child_node.final_verdict).__name__})"
                ),
            )
            root.final_verdict = Escalate(
                reason=f"chain stopped at slot {slot_name!r}: "
                       f"{getattr(child_node.final_verdict, 'reason', '')[:200]}"
            )
            return root

        out = child_node.final_result.output or {}
        value = str(out.get("value", "") or "")
        resolved[slot_name] = value
        root.slot_table[slot_name] = value
        all_triple_ids.extend(out.get("supporting_triple_ids", []) or [])

    # Whole chain resolved.
    final_value = resolved.get(FINAL_ANSWER_SLOT, "")
    root.final_result = TaskResult(
        status="success",
        output={
            "answer": final_value,
            "slot_values": dict(resolved),
            "supporting_triple_ids": all_triple_ids,
        },
        notes=f"chain executor: all {len(SLOT_ORDER)} slots certified",
    )
    root.final_verdict = Accept(
        reason=f"chain executor: every slot certified; answer={final_value!r}",
        record={
            "verifier": "chain_executor",
            "resolved_slots": dict(resolved),
            "total_triples_cited": len(set(all_triple_ids)),
        },
    )
    return root


CHAIN_MODES = {"oracle_chain", "llm_chain", "llm_chain_repair"}
CHAIN_EXECUTOR_MODES = {"oracle_chain_executor"}


def _contract_for(mode: str) -> TaskContract:
    goal = {
        "flat":              FLAT_GOAL,
        "oracle":            ORACLE_GOAL,
        "llm":               LLM_GOAL,
        "llm_repair":        LLM_REPAIR_GOAL,
        "oracle_chain":      ORACLE_CHAIN_GOAL,
        "llm_chain":         LLM_CHAIN_GOAL,
        "llm_chain_repair":  LLM_CHAIN_REPAIR_GOAL,
    }[mode]
    if mode in CHAIN_MODES:
        # Typed chain contract: plugin renders slot/edge guidance, runtime
        # threads consumed slots, chain_completeness verifier checks every
        # edge has a supporting triple before accepting.
        return TaskContract(
            goal=goal,
            output_schema=CHAIN_OUTPUT_SCHEMA,
            verifier="chain_completeness",
            inputs=CHAIN_INPUTS,
            success_criteria=[
                "every chain slot is populated in slot_values",
                "every chain edge is supported by a triple in the graph",
                "answer equals slot_values['founder']",
            ],
            budget=Budget(max_llm_calls=25, max_children=5, max_depth=2),
            repair_policy=RepairPolicy(enabled=True, max_attempts=3),
        )
    # Legacy modes: flat / oracle / llm / llm_repair.
    return TaskContract(
        goal=goal,
        output_schema=OUTPUT_SCHEMA,
        verifier="claim_evidence_alignment",
        success_criteria=[
            "answer identifies a specific person",
            "supporting_triple_ids are non-empty and exist in the shared graph",
        ],
        budget=Budget(max_llm_calls=25, max_children=5, max_depth=2),
        repair_policy=RepairPolicy(enabled=True, max_attempts=3),
    )


def run_one(mode: str, run_id: int, out_root: Path) -> RunResult:
    plugin = KnowledgeQAPlugin()  # fresh graph per run
    proof_dir = out_root / mode / f"run_{run_id}"
    proof_dir.mkdir(parents=True, exist_ok=True)

    try:
        if mode in CHAIN_EXECUTOR_MODES:
            # The chain executor drives the chain mechanically — no LLM root.
            proof = chain_executor(plugin)
        else:
            contract = _contract_for(mode)
            proof = run_task(contract, plugin)
    except Exception as e:  # noqa: BLE001
        return RunResult(
            mode=mode, run_id=run_id,
            verdict_kind="Exception",
            answer="",
            path={"justice_found": False, "college_found": False,
                  "founder_found": False, "answer_correct": False, "coverage": 0},
            metrics={},
            proof_dir=proof_dir,
            error=f"{type(e).__name__}: {e}",
        )

    answer = ""
    if proof.final_result and isinstance(proof.final_result.output, dict):
        answer = str(proof.final_result.output.get("answer", ""))

    path = path_score(plugin, answer)
    metrics = aggregate_metrics(proof)

    (proof_dir / "proof_tree.json").write_text(
        json.dumps(proof.to_dict(), indent=2, default=str), encoding="utf-8",
    )
    (proof_dir / "graph.json").write_text(
        json.dumps([t.to_dict() for t in plugin.triples], indent=2), encoding="utf-8",
    )

    return RunResult(
        mode=mode, run_id=run_id,
        verdict_kind=type(proof.final_verdict).__name__ if proof.final_verdict else "None",
        answer=answer,
        path=path,
        metrics=metrics,
        proof_dir=proof_dir,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def _fmt_row(r: RunResult) -> str:
    m = r.metrics
    return (
        f"  run {r.run_id}: "
        f"verdict={r.verdict_kind:<12} "
        f"correct={'Y' if r.path['answer_correct'] else 'n'} "
        f"cov={r.path['coverage']}/3 "
        f"calls={m.get('llm_calls','?'):>3} "
        f"tools={m.get('tool_calls','?'):>3} "
        f"children={m.get('direct_children','?')} "
        f"attempts={m.get('attempts','?')} "
        f"repairs={m.get('repairs','?')} "
        f"answer={(r.answer or '(none)')[:60]!r}"
    )


def _summary(results: list[RunResult]) -> None:
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'mode':<10} {'n':>3} {'correct':>8} {'cov_avg':>8} "
          f"{'llm_avg':>8} {'tool_avg':>8} {'children_avg':>13}")
    for mode in ["flat", "oracle", "llm", "llm_repair",
                 "oracle_chain", "llm_chain", "llm_chain_repair",
                 "oracle_chain_executor"]:
        rs = [r for r in results if r.mode == mode]
        if not rs:
            continue
        n = len(rs)
        correct = sum(1 for r in rs if r.path["answer_correct"])
        cov = sum(r.path["coverage"] for r in rs) / n
        llm_avg = sum(r.metrics.get("llm_calls", 0) for r in rs) / n
        tool_avg = sum(r.metrics.get("tool_calls", 0) for r in rs) / n
        child_avg = sum(r.metrics.get("direct_children", 0) for r in rs) / n
        print(f"{mode:<10} {n:>3} {correct:>4}/{n:<3} {cov:>7.2f} "
              f"{llm_avg:>8.1f} {tool_avg:>8.1f} {child_avg:>13.2f}")


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
    parser.add_argument("--runs", type=int, default=2,
                        help="Runs per mode (default 2)")
    parser.add_argument(
        "--modes", default="oracle_chain,llm_chain,llm_chain_repair",
        help=("Comma-separated subset of {flat,oracle,llm,llm_repair,"
              "oracle_chain,llm_chain,llm_chain_repair}. Default focuses on "
              "the chain modes — the user's recommended next experiment."),
    )
    args = parser.parse_args()

    _load_api_key()
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set (and no API_KEY= in .env).", file=sys.stderr)
        return 2

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out_root = Path(__file__).resolve().parent / "_mock_trial_output"
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True)

    results: list[RunResult] = []
    for mode in modes:
        print()
        print("-" * 78)
        print(f"MODE: {mode}  ({args.runs} run(s))")
        print("-" * 78)
        for i in range(1, args.runs + 1):
            print(f"  starting run {i}...")
            r = run_one(mode, i, out_root)
            results.append(r)
            print(_fmt_row(r))
            if r.error:
                print(f"    exception: {r.error}")

    # Per-mode summary table.
    _summary(results)

    # JSON dump of all results for downstream analysis.
    (out_root / "results.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2),
        encoding="utf-8",
    )
    print()
    print(f"All proof trees + summary: {out_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
