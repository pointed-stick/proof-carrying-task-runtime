"""Prove the runtime's recursive proof-tree plumbing — no live LLM.

A scripted FakeClient drives a deterministic two-level task tree:

  root  →  spawn_subtasks([child_1, child_2])  →  finish
  child_1 → finish
  child_2 → finish

Asserts the proof-tree shape:
  * root proof node accepted
  * children list has 2 entries
  * each child's parent_id is the root task_id
  * each child accepted with the scripted output
  * root's final output is the integrated summary

Run:
  python -m task_runtime.tests.test_recursion
  pytest task_runtime/tests/test_recursion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import (  # noqa: E402
    Accept, Budget, ProofNode, RepairPolicy, TaskContract,
)
from task_runtime.runtime import run_task  # noqa: E402


# -----------------------------------------------------------------------------
# Minimal OpenAI-shaped fakes
# -----------------------------------------------------------------------------

class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class FakeMessage:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content

    def model_dump(self, exclude_none: bool = True) -> dict:
        d: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


class FakeChatCompletions:
    """Replays a scripted list of FakeMessage responses in order."""

    def __init__(self, script: list[FakeMessage]):
        self.script = list(script)
        self.calls = 0

    def create(self, **kwargs) -> SimpleNamespace:
        if self.calls >= len(self.script):
            raise AssertionError(
                f"FakeChatCompletions: script exhausted at call #{self.calls + 1}"
            )
        msg = self.script[self.calls]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeClient:
    def __init__(self, script: list[FakeMessage]):
        self._completions = FakeChatCompletions(script)
        self.chat = SimpleNamespace(completions=self._completions)


# -----------------------------------------------------------------------------
# A plugin with no tools and no verifier — the test is about the runtime,
# not the plugin.
# -----------------------------------------------------------------------------

class NoopPlugin:
    name = "noop"

    def tools(self):
        return {}

    def verifiers(self):
        return {}

    def render_contract_context(self, contract):
        return "PLUGIN: noop"

    def coherence_check(self, contract):
        return None


# -----------------------------------------------------------------------------
# The test
# -----------------------------------------------------------------------------

def _build_script() -> list[FakeMessage]:
    """Script the model's tool-call sequence for a 2-child recursion."""
    return [
        # Call 1 (root): spawn two children.
        FakeMessage(tool_calls=[FakeToolCall(
            "call_root_spawn", "spawn_subtasks",
            {"subtasks": [
                {"goal": "subtask one"},
                {"goal": "subtask two"},
            ]},
        )]),
        # Call 2 (child 1): finish.
        FakeMessage(tool_calls=[FakeToolCall(
            "call_c1_finish", "finish",
            {"status": "success", "output": {"value": "child1_done"}},
        )]),
        # Call 3 (child 2): finish.
        FakeMessage(tool_calls=[FakeToolCall(
            "call_c2_finish", "finish",
            {"status": "success", "output": {"value": "child2_done"}},
        )]),
        # Call 4 (root, after spawn returns): finish, integrating children.
        FakeMessage(tool_calls=[FakeToolCall(
            "call_root_finish", "finish",
            {
                "status": "success",
                "output": {"summary": "both children done",
                           "children_outputs": ["child1_done", "child2_done"]},
            },
        )]),
    ]


def build_recursion_proof() -> ProofNode:
    """Drive the runtime through the scripted 2-level tree and return the proof.

    Kept separate from the test function so it can return a ProofNode (which
    pytest would otherwise warn about — PytestReturnNotNoneWarning).
    """
    client = FakeClient(_build_script())
    contract = TaskContract(
        goal="root task",
        budget=Budget(max_llm_calls=5, max_children=5, max_depth=3),
        repair_policy=RepairPolicy(enabled=False),
    )
    proof = run_task(contract, NoopPlugin(), client=client)
    # Script consumption check lives here so the helper still validates that
    # the runtime made exactly the expected number of LLM calls.
    assert client._completions.calls == 4, (
        f"expected 4 LLM calls, got {client._completions.calls}"
    )
    return proof


def test_recursion_proof_tree() -> None:
    """Pytest-discoverable assertion-only test."""
    proof = build_recursion_proof()

    # Root accepted.
    assert isinstance(proof, ProofNode), f"expected ProofNode, got {type(proof)}"
    assert isinstance(proof.final_verdict, Accept), (
        f"root not accepted: {type(proof.final_verdict).__name__}"
    )

    # Children attached under root.
    assert len(proof.children) == 2, f"expected 2 children, got {len(proof.children)}"
    for i, child in enumerate(proof.children):
        assert isinstance(child.final_verdict, Accept), (
            f"child {i} not accepted: {type(child.final_verdict).__name__}"
        )
        assert child.parent_id == proof.task_id, (
            f"child {i} parent_id={child.parent_id!r} != root {proof.task_id!r}"
        )
        expected_value = f"child{i + 1}_done"
        assert child.final_result.output == {"value": expected_value}, (
            f"child {i} unexpected output: {child.final_result.output}"
        )

    # Root integrated the children.
    assert proof.final_result.output == {
        "summary": "both children done",
        "children_outputs": ["child1_done", "child2_done"],
    }, f"unexpected root output: {proof.final_result.output}"


def main() -> int:
    proof = build_recursion_proof()
    print("recursion proof: OK")
    print(f"  root task_id:      {proof.task_id}")
    print(f"  root attempts:     {len(proof.attempts)}")
    print(f"  root final output: {proof.final_result.output}")
    print(f"  child count:       {len(proof.children)}")
    for c in proof.children:
        print(
            f"    - {c.task_id} parent={c.parent_id} "
            f"verdict={type(c.final_verdict).__name__} "
            f"output={c.final_result.output}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
