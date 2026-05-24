"""The execution loop.

  run_task(contract, plugin):
    1. Coherence-check the contract (plugin may reject malformed contracts).
    2. For each repair attempt up to budget:
         a. Drive an LLM tool-use loop until the model calls finish() or
            runs out of LLM calls.
         b. Run the configured verifier on the structured result.
         c. If Accept: stop, attach to proof tree, return.
         d. If RejectWithRepairHint: fold the hint into the next user message
            and retry (within max_repair_attempts).
         e. If Escalate or IncoherentContract: stop, attach, return.
    3. The model can call spawn_subtasks() to create child tasks; each child
       runs its own full run_task() (recursive), with its own verifier loop,
       and its proof node attaches under this task's children.

The runtime does NOT know about triples, pytest, or any specific domain. All
domain knowledge lives in the plugin's tools/verifiers/context renderer.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Callable

from .core import (
    Accept,
    Attempt,
    Budget,
    Escalate,
    IncoherentContract,
    Plugin,
    ProofNode,
    RejectWithRepairHint,
    RepairPolicy,
    TaskContract,
    TaskResult,
    Verdict,
    validate_output_schema,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openai import OpenAI


MODEL = os.environ.get("TASK_RUNTIME_MODEL", "gpt-4o-mini")


def _child_allowed_tools(
    parent_allowed: list[str] | None,
    requested: list[str] | None,
) -> list[str] | None:
    """Compute a child's allowed_tools as a true subset of the parent's.

    Semantics:
      requested=None         → inherit parent's restriction unchanged
      parent_allowed=None    → child may freely choose any subset (parent
                                imposes no restriction)
      both restricted        → child's choices MUST be a subset of parent's;
                                requesting any tool outside parent's set
                                raises ValueError so the runtime can attach
                                a clear error to the spawn response rather
                                than silently expanding access.

    This makes the safety boundary compose recursively — a parent's tool
    restriction is never circumvented by a child's spawn-time override.
    """
    if requested is None:
        return parent_allowed
    if parent_allowed is None:
        return list(requested)
    disallowed = sorted(set(requested) - set(parent_allowed))
    if disallowed:
        raise ValueError(
            f"child requested tools outside parent allowance: {disallowed}"
        )
    return list(requested)


def _default_client() -> "OpenAI":
    """Construct an OpenAI client lazily.

    Kept lazy so `task_runtime.core` types (contracts, proof nodes, verifiers)
    can be imported and used without `openai` installed — useful for offline
    proof-tree reading, schema inspection, and plugin unit tests.
    """
    try:
        from openai import OpenAI  # noqa: PLC0415 — intentional lazy import
    except ImportError as e:
        raise RuntimeError(
            "OpenAI client required to run LLM tasks. "
            "Install with `pip install openai`, or pass a compatible client to run_task()."
        ) from e
    return OpenAI()


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_task(
    contract: TaskContract,
    plugin: Plugin,
    *,
    parent_id: str | None = None,
    depth: int = 0,
    client: "OpenAI | None" = None,
) -> ProofNode:
    """Run one task end-to-end and return its proof node."""
    node = ProofNode.new(contract, parent_id=parent_id)

    # Coherence check is purely offline — runs before any LLM client is needed.
    coh = _coherence_check(plugin, contract)
    if isinstance(coh, IncoherentContract):
        node.final_verdict = coh
        return node

    if depth >= contract.budget.max_depth:
        node.final_verdict = Escalate(reason=f"depth budget exhausted at depth={depth}")
        return node

    # NOW we need an LLM client. Constructing it earlier would force users
    # to install `openai` even to get an offline IncoherentContract rejection.
    client = client or _default_client()

    repair_hint: str | None = None
    max_attempts = (
        contract.repair_policy.max_attempts if contract.repair_policy.enabled else 1
    )

    for attempt_no in range(1, max_attempts + 1):
        result, cost = _execute_attempt(
            contract, plugin, client, repair_hint, node, depth,
        )
        verdict = _verify(contract, result, plugin)
        node.attempts.append(Attempt(
            attempt_no=attempt_no,
            result=result,
            verdict=verdict,
            repair_hint=repair_hint,
            cost=cost,
        ))

        if isinstance(verdict, Accept):
            node.final_result = result
            node.final_verdict = verdict
            return node

        if isinstance(verdict, RejectWithRepairHint) and contract.repair_policy.enabled:
            repair_hint = verdict.hint
            if verdict.missing_requirements:
                repair_hint += "\nMissing: " + ", ".join(verdict.missing_requirements)
            continue

        # Escalate / non-repairable rejection → stop.
        node.final_result = result
        node.final_verdict = verdict
        return node

    # Repair attempts exhausted: last attempt's outcome becomes final.
    last = node.attempts[-1]
    node.final_result = last.result
    node.final_verdict = last.verdict
    return node


# -----------------------------------------------------------------------------
# Inner: one attempt
# -----------------------------------------------------------------------------

def _execute_attempt(
    contract: TaskContract,
    plugin: Plugin,
    client: "OpenAI",
    repair_hint: str | None,
    node: ProofNode,
    depth: int,
) -> tuple[TaskResult, dict]:
    """Run a single LLM tool-use loop until finish() or budget exhaustion."""

    # Spawn callback closes over `node`, `depth`, `plugin`, `client` so
    # children attach under this proof node. max_children is enforced
    # per-attempt: a failed attempt that spawned children shouldn't starve
    # the subsequent repair attempt of its own budget. node.children still
    # accumulates across attempts so the proof tree shows every spawn.
    children_this_attempt = 0

    def spawn_subtasks(subtasks: list[dict]) -> dict:
        nonlocal children_this_attempt
        out = []
        for st in subtasks:
            if children_this_attempt >= contract.budget.max_children:
                out.append({
                    "task_id": None,
                    "verdict_kind": "BudgetExhausted",
                    "error": f"max_children={contract.budget.max_children} "
                             f"reached for this attempt",
                })
                continue
            try:
                child_allowed = _child_allowed_tools(
                    contract.allowed_tools,
                    st.get("allowed_tools") if "allowed_tools" in st else None,
                )
            except ValueError as exc:
                out.append({
                    "task_id": None,
                    "verdict_kind": "IncoherentContract",
                    "error": str(exc),
                })
                continue

            # Dataflow threading: pull `consumes` slot values from the parent's
            # slot_table into the child's inputs so the child agent sees the
            # concrete values it must build on. Also surface relation_lock /
            # expected_produces so the rendered context can emphasize them.
            child_inputs = dict(st.get("inputs", {}) or {})
            consumed_slots: dict = {}
            for slot in (st.get("consumes") or []):
                if slot in node.slot_table:
                    consumed_slots[slot] = node.slot_table[slot]
            if consumed_slots:
                child_inputs["consumed_slots"] = consumed_slots
            if st.get("produces"):
                child_inputs["expected_produces"] = list(st["produces"])
            if st.get("relation_lock"):
                child_inputs["relation_lock"] = st["relation_lock"]

            child = TaskContract(
                goal=st.get("goal", ""),
                output_schema=st.get("output_schema"),
                verifier=st.get("verifier"),
                inputs=child_inputs,
                success_criteria=st.get("success_criteria", []),
                allowed_tools=child_allowed,
                # Inherit parent's depth budget (cap is global); LLM-call
                # budget and repair attempts may be overridden per child.
                budget=Budget(
                    max_llm_calls=st.get("max_llm_calls", contract.budget.max_llm_calls),
                    max_children=contract.budget.max_children,
                    max_depth=contract.budget.max_depth,
                ),
                repair_policy=RepairPolicy(
                    enabled=contract.repair_policy.enabled,
                    max_attempts=st.get(
                        "max_repair_attempts", contract.repair_policy.max_attempts,
                    ),
                ),
            )
            children_this_attempt += 1
            child_node = run_task(
                child, plugin,
                parent_id=node.task_id, depth=depth + 1, client=client,
            )
            node.children.append(child_node)

            # Dataflow: if the child was accepted and its output declares
            # slot values, merge them into the parent's slot_table. We accept
            # two shapes for robustness (models reliably get one or the other
            # but not both):
            #   1. output["slot_values"] = {slot: value, ...}   (canonical)
            #   2. output[slot] = value, for each slot in `produces` (top-level)
            # The NEXT child whose `consumes` mentions one of those slots
            # will see the concrete value injected into its inputs.
            produced_slots: dict = {}
            declared_produces = list(st.get("produces") or [])
            if child_node.accepted and child_node.final_result:
                out_obj = child_node.final_result.output
                if isinstance(out_obj, dict):
                    # Shape 1: nested slot_values / produces dict.
                    nested = out_obj.get("slot_values") or out_obj.get("produces") or {}
                    if isinstance(nested, dict):
                        for k, v in nested.items():
                            if v:
                                node.slot_table[k] = v
                                produced_slots[k] = v
                    # Shape 2: top-level keys matching declared produces.
                    for slot in declared_produces:
                        if slot in produced_slots:
                            continue  # already captured from nested
                        v = out_obj.get(slot)
                        if v:
                            node.slot_table[slot] = v
                            produced_slots[slot] = v

            out.append({
                "task_id": child_node.task_id,
                "verdict_kind": (
                    type(child_node.final_verdict).__name__
                    if child_node.final_verdict else None
                ),
                "accepted": child_node.accepted,
                "output": (
                    child_node.final_result.output
                    if child_node.final_result else None
                ),
                "notes": (
                    child_node.final_result.notes
                    if child_node.final_result else ""
                ),
                "produced_slots": produced_slots,
                "parent_slot_table": dict(node.slot_table),
            })
        return {"children": out, "parent_slot_table": dict(node.slot_table)}

    tools_spec, tool_fns = _build_tool_specs(plugin, contract, spawn_subtasks)
    system = _render_system(contract, plugin)
    user = _render_user(contract, repair_hint)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    cost = {"llm_calls": 0, "tool_calls": 0}
    finished: dict | None = None

    for _ in range(contract.budget.max_llm_calls):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools_spec, tool_choice="auto",
        )
        cost["llm_calls"] += 1
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # Free-text reply when we expected finish() — fail this attempt cleanly.
            return TaskResult(
                status="failed",
                output=msg.content or "",
                notes="model returned prose instead of calling finish()",
                cost=cost,
            ), cost

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args = {}
                tool_out: Any = {"error": f"could not parse tool args: {e}"}
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps(tool_out),
                })
                continue

            cost["tool_calls"] += 1

            if name == "finish":
                finished = args
                # Acknowledge the finish call so the message log is consistent.
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps({"ok": True}),
                })
                break

            fn = tool_fns.get(name)
            if fn is None:
                tool_out = {"error": f"unknown tool: {name}"}
            else:
                try:
                    tool_out = fn(**args)
                except Exception as e:  # noqa: BLE001 — surface to the model
                    tool_out = {"error": f"{type(e).__name__}: {e}"}

            messages.append({
                "role": "tool", "tool_call_id": call.id,
                "content": json.dumps(tool_out, default=str)[:8000],
            })

        if finished is not None:
            break

    if finished is None:
        return TaskResult(
            status="failed",
            output=None,
            notes=f"LLM-call budget ({contract.budget.max_llm_calls}) exhausted without finish()",
            cost=cost,
        ), cost

    return TaskResult(
        status=finished.get("status", "success"),
        output=finished.get("output"),
        artifacts=finished.get("artifacts", {}) or {},
        notes=finished.get("notes", "") or "",
        cost=cost,
    ), cost


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _coherence_check(plugin: Plugin, contract: TaskContract) -> Verdict | None:
    fn = getattr(plugin, "coherence_check", None)
    if fn is None:
        return None
    try:
        return fn(contract)
    except Exception as e:  # noqa: BLE001
        return IncoherentContract(reason=f"coherence_check raised: {e}")


def _verify(contract: TaskContract, result: TaskResult, plugin: Plugin) -> Verdict:
    """Three-stage verification: runtime sanity → schema → plugin."""
    # Stage 0: runtime/protocol sanity.
    # A TaskResult the runtime itself flagged as failed must NEVER be accepted
    # just because the contract has no verifier. Contract elasticity means
    # "you can opt out of schema/plugin verification" — not "accept runtime
    # failure". The runtime's own failure modes (no finish, prose-only reply,
    # budget exhaustion) get a repair hint; explicit model-reported failure
    # escalates.
    if result.status == "failed" and (
        result.output is None
        or "without finish" in (result.notes or "")
        or "prose instead of calling finish" in (result.notes or "")
    ):
        return RejectWithRepairHint(
            reason="task attempt did not produce a valid finish() result",
            hint=(
                "You MUST call the `finish` tool to return a structured result. "
                "Do not stop after running tools and do not return prose only. "
                "Call finish(status='success', output=...) if you completed the task, "
                "or finish(status='missing'/'failed', output=..., notes='why') if you "
                "could not."
            ),
            missing_requirements=["valid finish() call"],
        )
    if result.status in {"missing", "failed"}:
        return Escalate(
            reason=f"task self-reported status={result.status!r}: "
                   f"{result.notes or '(no notes)'}"
        )
    # status='partial' is acceptable only if a plugin verifier explicitly
    # ratifies it. Under a bare contract (no verifier), partial must escalate;
    # otherwise `proof.accepted` would be true while final_result.status is
    # "partial" — the status/verdict disagreement the framework was meant to
    # avoid. Plugins that want to accept partials should configure a verifier
    # that does so explicitly.
    if result.status == "partial" and contract.verifier is None:
        return Escalate(
            reason="status='partial' under a bare contract; no verifier configured to ratify partial success"
        )

    # Stage 1: runtime-level schema validation.
    # If the contract declared an output_schema, the runtime enforces it
    # *before* any plugin verifier runs. This makes contract level 2 (schema)
    # mean something, not just appear in the prompt.
    if contract.output_schema is not None:
        schema_errors = validate_output_schema(result.output, contract.output_schema)
        if schema_errors:
            preview = "\n  - ".join(schema_errors[:10])
            return RejectWithRepairHint(
                reason=f"output failed schema validation ({len(schema_errors)} error(s))",
                hint=(
                    "Your finish() output did not conform to the contract's output_schema. "
                    "Errors:\n  - " + preview + "\n"
                    "Re-call finish() with output shaped per OUTPUT SCHEMA."
                ),
                missing_requirements=schema_errors[:10],
            )

    # Stage 2: plugin verifier (if configured).
    if contract.verifier is None:
        return Accept(reason="schema validated; no plugin verifier configured")
    verifiers = plugin.verifiers()
    v = verifiers.get(contract.verifier)
    if v is None:
        return Escalate(
            reason=f"verifier '{contract.verifier}' not found in plugin '{plugin.name}'"
        )
    try:
        return v.check(contract, result, workspace={})
    except Exception as e:  # noqa: BLE001
        return Escalate(reason=f"verifier raised: {type(e).__name__}: {e}")


def _build_tool_specs(
    plugin: Plugin, contract: TaskContract, spawn_callback: Callable,
):
    """OpenAI tool specs + name→callable map. Always includes finish + spawn."""
    tool_fns: dict[str, Callable] = {}
    specs: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": (
                    "Return the final structured result for this task. "
                    "Call this exactly once when done."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["success", "partial", "missing", "failed"],
                        },
                        "output": {
                            "description": (
                                "Final result. Must conform to output_schema if given."
                            ),
                        },
                        "artifacts": {
                            "type": "object",
                            "description": "Named artifacts produced (paths, IDs, ...).",
                        },
                        "notes": {"type": "string"},
                    },
                    "required": ["status", "output"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_subtasks",
                "description": (
                    "Spawn child tasks. Each runs its own full verifier loop and "
                    "returns a structured result. Use for decomposition."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subtasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "goal": {"type": "string"},
                                    "output_schema": {"type": "object"},
                                    "verifier": {"type": "string"},
                                    "inputs": {"type": "object"},
                                    "success_criteria": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "max_llm_calls": {"type": "integer"},
                                    "max_repair_attempts": {"type": "integer"},
                                    "allowed_tools": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": (
                                            "Narrow the child's tool access. "
                                            "Omit to inherit the parent's restriction."
                                        ),
                                    },
                                    "consumes": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": (
                                            "Names of slots from the parent's "
                                            "slot_table this child needs. The "
                                            "runtime auto-injects their values "
                                            "into the child's inputs["
                                            "'consumed_slots']."
                                        ),
                                    },
                                    "produces": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": (
                                            "Names of slots this child will "
                                            "populate. The child's finish() "
                                            "output should include "
                                            "slot_values={slot: value, ...} "
                                            "for these. The runtime merges "
                                            "them into the parent's "
                                            "slot_table on accept."
                                        ),
                                    },
                                    "relation_lock": {
                                        "type": "string",
                                        "description": (
                                            "A semantic relation the child "
                                            "must preserve (e.g. "
                                            "'undergraduate college "
                                            "attended', NOT 'law school "
                                            "attended'). Surfaced in the "
                                            "child's rendered context."
                                        ),
                                    },
                                },
                                "required": ["goal"],
                            },
                        },
                    },
                    "required": ["subtasks"],
                },
            },
        },
    ]
    tool_fns["spawn_subtasks"] = spawn_callback

    raw_tools = plugin.tools()
    allowed = contract.allowed_tools
    for name, fn in raw_tools.items():
        if allowed is not None and name not in allowed:
            continue
        tool_fns[name] = fn
        spec = getattr(fn, "_tool_spec", None)
        if spec is None:
            spec = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (fn.__doc__ or name).strip().splitlines()[0],
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        specs.append(spec)

    return specs, tool_fns


def _render_system(contract: TaskContract, plugin: Plugin) -> str:
    lines = [
        "You are a task-execution agent operating under a structured contract.",
        "You MUST call the `finish` tool exactly once with your final structured result.",
        "If the task needs decomposition, call `spawn_subtasks` to create child tasks; "
        "each child has its own verifier and budget, and returns a structured result.",
        "Use the plugin's tools to interact with the workspace; do not invent results.",
        "If you cannot complete the task, call finish() with status='missing' or 'failed' "
        "and explain in notes — do NOT fabricate output.",
        "",
        plugin.render_contract_context(contract),
    ]
    return "\n".join(lines)


def _render_user(contract: TaskContract, repair_hint: str | None) -> str:
    parts = [f"GOAL: {contract.goal}"]
    if contract.inputs:
        parts.append("INPUTS: " + json.dumps(contract.inputs))
    if contract.output_schema:
        parts.append("OUTPUT SCHEMA: " + json.dumps(contract.output_schema))
    if contract.success_criteria:
        parts.append("SUCCESS CRITERIA:")
        parts.extend(f"  - {c}" for c in contract.success_criteria)
    if repair_hint:
        parts.append("")
        parts.append("REPAIR — your previous result failed verification.")
        parts.append("REPAIR HINT:")
        parts.append(repair_hint)
        parts.append("")
        parts.append("Address the issue above and call finish() again with the corrected result.")
    return "\n".join(parts)
