"""Core types: contracts, results, verdicts, proof nodes, plugin protocol.

The runtime is contract-elastic: a TaskContract can be anywhere from a bare
goal string (level 0) to a fully typed, verified, repair-policied contract
(level 4). The runtime does not require any particular level.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Protocol, runtime_checkable


# -----------------------------------------------------------------------------
# Budgets and repair policy
# -----------------------------------------------------------------------------

@dataclass
class Budget:
    """How much the runtime is allowed to spend on one task attempt.

    Repair attempts are budgeted separately via RepairPolicy.max_attempts so
    there is a single source of truth for retry count.
    """
    max_llm_calls: int = 10
    max_children: int = 5
    max_depth: int = 3


@dataclass
class RepairPolicy:
    """How the runtime reacts to RejectWithRepairHint verdicts.

    Each repair attempt starts a fresh LLM conversation (the prior attempt's
    failure is folded in as a REPAIR HINT in the new user message). Carry-
    forward mode (preserving prior messages) is deliberately not implemented
    in v1.
    """
    enabled: bool = True
    max_attempts: int = 3


# -----------------------------------------------------------------------------
# Contract and result
# -----------------------------------------------------------------------------

@dataclass
class TaskContract:
    """A declarative description of what a task should accomplish."""
    goal: str
    output_schema: dict | None = None
    verifier: str | None = None          # name of a verifier registered on the plugin
    success_criteria: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    allowed_tools: list[str] | None = None  # None = all plugin tools
    budget: Budget = field(default_factory=Budget)
    repair_policy: RepairPolicy = field(default_factory=RepairPolicy)
    deps: list[str] = field(default_factory=list)  # IDs of sibling tasks this depends on

    @property
    def level(self) -> int:
        """Coarse description of contract strictness (0..4) — for telemetry/proof tree."""
        n = 0
        if self.budget != Budget():
            n = max(n, 1)
        if self.output_schema is not None:
            n = max(n, 2)
        if self.verifier is not None:
            n = max(n, 3)
        if self.success_criteria or self.deps or not self.repair_policy.enabled:
            n = max(n, 4)
        return n


@dataclass
class TaskResult:
    """What a task attempt actually produced."""
    status: str  # "success" | "partial" | "missing" | "failed"
    output: Any
    artifacts: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, int] = field(default_factory=dict)
    notes: str = ""


# -----------------------------------------------------------------------------
# Verdicts (verifier output)
# -----------------------------------------------------------------------------

class Verdict:
    """Base sentinel; concrete verdicts are the dataclasses below."""
    pass


@dataclass
class Accept(Verdict):
    reason: str = ""
    # Verifier-supplied evidence: command, exit code, hashes, etc. Used to
    # turn the proof tree into an independently re-checkable certificate.
    record: dict = field(default_factory=dict)


@dataclass
class RejectWithRepairHint(Verdict):
    reason: str
    hint: str
    missing_requirements: list[str] = field(default_factory=list)
    record: dict = field(default_factory=dict)


@dataclass
class Escalate(Verdict):
    """Stop trying; surface this for parent/human decision."""
    reason: str
    record: dict = field(default_factory=dict)


@dataclass
class IncoherentContract(Verdict):
    """The contract itself is malformed (e.g., verifier name doesn't exist;
    evidence verifier on an essentially-subjective goal). No record — the
    contract never executed."""
    reason: str


# -----------------------------------------------------------------------------
# Proof tree
# -----------------------------------------------------------------------------

@dataclass
class Attempt:
    """One execution pass: LLM run + verifier check."""
    attempt_no: int
    result: TaskResult | None
    verdict: Verdict | None
    repair_hint: str | None = None  # the hint that produced this attempt (None for first)
    cost: dict = field(default_factory=dict)


@dataclass
class ProofNode:
    task_id: str
    parent_id: str | None
    contract: TaskContract
    attempts: list[Attempt] = field(default_factory=list)
    children: list["ProofNode"] = field(default_factory=list)
    final_result: TaskResult | None = None
    final_verdict: Verdict | None = None
    # Slot table: tracks dataflow values produced by children and consumed
    # by siblings. Populated by the runtime when a child whose contract
    # declared `produces` returns slot_values in its output. Persists across
    # repair attempts of the same task so a successful child's contribution
    # isn't lost if a sibling fails.
    slot_table: dict = field(default_factory=dict)

    @classmethod
    def new(cls, contract: TaskContract, parent_id: str | None = None) -> "ProofNode":
        return cls(
            task_id=f"T{uuid.uuid4().hex[:8]}",
            parent_id=parent_id,
            contract=contract,
        )

    @property
    def accepted(self) -> bool:
        return isinstance(self.final_verdict, Accept)

    def to_dict(self) -> dict:
        """JSON-friendly serialization of the whole subtree."""

        def enc_verdict(v: Verdict | None) -> dict | None:
            if v is None:
                return None
            return {"kind": type(v).__name__, **v.__dict__}

        def enc_attempt(a: Attempt) -> dict:
            return {
                "attempt_no": a.attempt_no,
                "result": asdict(a.result) if a.result else None,
                "verdict": enc_verdict(a.verdict),
                "repair_hint": a.repair_hint,
                "cost": a.cost,
            }

        def enc_node(n: ProofNode) -> dict:
            return {
                "task_id": n.task_id,
                "parent_id": n.parent_id,
                "contract": asdict(n.contract),
                "contract_level": n.contract.level,
                "attempts": [enc_attempt(a) for a in n.attempts],
                "children": [enc_node(c) for c in n.children],
                "final_result": asdict(n.final_result) if n.final_result else None,
                "final_verdict": enc_verdict(n.final_verdict),
                "slot_table": dict(n.slot_table),
            }

        return enc_node(self)


# -----------------------------------------------------------------------------
# Minimal output-schema validator (runtime-level, no external deps)
# -----------------------------------------------------------------------------

def validate_output_schema(value: Any, schema: dict, path: str = "") -> list[str]:
    """Validate `value` against a tiny subset of JSON Schema.

    Supported keywords: type (object/array/string/integer/number/boolean/null),
    required, properties, items. Unsupported keywords are ignored. Returns a
    list of human-readable error messages (empty list = valid).
    """
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []
    expected = schema.get("type")
    here = path or "<root>"

    def _is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    if expected == "object":
        if not isinstance(value, dict):
            errors.append(f"{here}: expected object, got {type(value).__name__}")
            return errors
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{here}: missing required field '{req}'")
        for k, sub in (schema.get("properties") or {}).items():
            if k in value:
                errors.extend(
                    validate_output_schema(value[k], sub, f"{path}.{k}" if path else k)
                )
    elif expected == "array":
        if not isinstance(value, list):
            errors.append(f"{here}: expected array, got {type(value).__name__}")
            return errors
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors.extend(
                    validate_output_schema(item, item_schema, f"{path}[{i}]")
                )
    elif expected == "string":
        if not isinstance(value, str):
            errors.append(f"{here}: expected string, got {type(value).__name__}")
    elif expected == "integer":
        if not _is_int(value):
            errors.append(f"{here}: expected integer, got {type(value).__name__}")
    elif expected == "number":
        if not (_is_int(value) or isinstance(value, float)):
            errors.append(f"{here}: expected number, got {type(value).__name__}")
    elif expected == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{here}: expected boolean, got {type(value).__name__}")
    elif expected == "null":
        if value is not None:
            errors.append(f"{here}: expected null, got {type(value).__name__}")
    # Unknown / missing type: no constraint.
    return errors


# -----------------------------------------------------------------------------
# Plugin & verifier protocols
# -----------------------------------------------------------------------------

@runtime_checkable
class Verifier(Protocol):
    def check(self, contract: TaskContract, result: TaskResult, workspace: dict) -> Verdict: ...


@runtime_checkable
class Plugin(Protocol):
    name: str

    def tools(self) -> dict[str, Callable]: ...
    def verifiers(self) -> dict[str, Verifier]: ...
    def render_contract_context(self, contract: TaskContract) -> str: ...
    # Optional; default to "no check".
    def coherence_check(self, contract: TaskContract) -> Verdict | None: ...
