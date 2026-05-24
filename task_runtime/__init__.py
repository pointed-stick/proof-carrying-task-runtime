"""Proof-Carrying Recursive Task Runtime.

A lightweight runtime where an LLM accomplishes a task by:
  1. operating under an explicit (elastic) contract,
  2. spawning sub-task contracts recursively,
  3. having results checked by a verifier,
  4. repairing failures via hints folded into a retry,
  5. emitting a proof tree (not just a debug log).

Domain-specific machinery (knowledge graphs, code sandboxes, document tools)
lives in plugins. The runtime knows nothing about triples, pytest, or Wikipedia.
"""

from .core import (
    TaskContract, TaskResult, Budget, RepairPolicy,
    Verdict, Accept, RejectWithRepairHint, Escalate, IncoherentContract,
    ProofNode, Attempt,
    Plugin, Verifier,
)
from .runtime import run_task

__all__ = [
    "TaskContract", "TaskResult", "Budget", "RepairPolicy",
    "Verdict", "Accept", "RejectWithRepairHint", "Escalate", "IncoherentContract",
    "ProofNode", "Attempt",
    "Plugin", "Verifier",
    "run_task",
]
