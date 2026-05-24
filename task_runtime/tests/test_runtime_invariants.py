"""Runtime invariants that need scripted FakeClient tests (not live model).

Covers:
  - _child_allowed_tools intersection semantics
  - spawn_subtasks rejects requests for tools outside the parent's allowance
  - max_children is per-attempt (a failed repair doesn't starve the next)
  - max_children still caps across multiple spawn_subtasks calls in one attempt

Run:
  python -m task_runtime.tests.test_runtime_invariants
  pytest task_runtime/tests/test_runtime_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import (  # noqa: E402
    Accept, Budget, RepairPolicy, TaskContract,
)
from task_runtime.runtime import _child_allowed_tools, run_task  # noqa: E402
from task_runtime.tests.test_recursion import (  # noqa: E402
    FakeClient, FakeMessage, FakeToolCall, NoopPlugin,
)


# -----------------------------------------------------------------------------
# _child_allowed_tools — pure helper
# -----------------------------------------------------------------------------

def test_child_allowed_tools_none_requested_inherits_parent() -> None:
    assert _child_allowed_tools(["a", "b"], None) == ["a", "b"]
    assert _child_allowed_tools(None, None) is None
    assert _child_allowed_tools([], None) == []


def test_child_allowed_tools_unrestricted_parent_allows_any_subset() -> None:
    assert _child_allowed_tools(None, ["x"]) == ["x"]
    assert _child_allowed_tools(None, []) == []


def test_child_allowed_tools_restricted_parent_allows_subset() -> None:
    assert _child_allowed_tools(["a", "b", "c"], ["a"]) == ["a"]
    assert _child_allowed_tools(["a", "b", "c"], ["a", "b"]) in (["a", "b"], ["b", "a"])
    assert _child_allowed_tools(["a", "b", "c"], []) == []


def test_child_allowed_tools_rejects_expansion() -> None:
    import pytest
    try:
        _child_allowed_tools(["a"], ["a", "b"])
    except ValueError as e:
        assert "b" in str(e)
        return
    raise AssertionError("expected ValueError for tool outside parent set")


def test_child_allowed_tools_rejects_when_parent_empty() -> None:
    """Parent explicitly forbids tools — child cannot acquire any."""
    try:
        _child_allowed_tools([], ["anything"])
    except ValueError as e:
        assert "anything" in str(e)
        return
    raise AssertionError("expected ValueError for tool outside empty parent set")


# -----------------------------------------------------------------------------
# spawn_subtasks integration: tool inheritance & rejection of expansion
# -----------------------------------------------------------------------------

def test_spawn_rejects_child_requesting_tool_outside_parent() -> None:
    """A child that tries to expand the parent's allowance must be rejected.

    The runtime returns an error in the spawn response (it doesn't crash),
    so the parent can decide what to do. The malformed child does NOT attach
    to the proof tree.
    """
    script = [
        FakeMessage(tool_calls=[FakeToolCall(
            "s1", "spawn_subtasks",
            {"subtasks": [
                {"goal": "good child", "allowed_tools": ["read_file"]},
                {"goal": "bad child", "allowed_tools": ["secret_tool"]},
            ]},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "c1", "finish", {"status": "success", "output": {}},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "root", "finish", {"status": "success", "output": {}},
        )]),
    ]
    proof = run_task(
        TaskContract(
            goal="root", allowed_tools=["read_file", "list_files"],
            budget=Budget(max_llm_calls=10, max_children=5, max_depth=3),
            repair_policy=RepairPolicy(enabled=False),
        ),
        NoopPlugin(),
        client=FakeClient(script),
    )
    # Only the well-formed child should attach.
    assert len(proof.children) == 1, (
        f"expected 1 child (bad one rejected), got {len(proof.children)}"
    )
    assert proof.children[0].contract.allowed_tools == ["read_file"]


def test_spawn_child_with_empty_parent_allowance_cannot_acquire_tools() -> None:
    """Parent with allowed_tools=[] explicitly forbids tools. Children may not
    acquire any."""
    script = [
        FakeMessage(tool_calls=[FakeToolCall(
            "s1", "spawn_subtasks",
            {"subtasks": [
                {"goal": "tries to escalate", "allowed_tools": ["wiki_read"]},
            ]},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "root", "finish", {"status": "success", "output": {}},
        )]),
    ]
    proof = run_task(
        TaskContract(
            goal="root", allowed_tools=[],
            budget=Budget(max_llm_calls=10, max_children=5, max_depth=3),
            repair_policy=RepairPolicy(enabled=False),
        ),
        NoopPlugin(),
        client=FakeClient(script),
    )
    assert len(proof.children) == 0, (
        f"child should have been rejected; got {len(proof.children)} children"
    )


def test_spawn_child_inherits_parent_restriction_when_omitting_allowed_tools() -> None:
    script = [
        FakeMessage(tool_calls=[FakeToolCall(
            "s1", "spawn_subtasks",
            {"subtasks": [{"goal": "inherits"}]},  # no allowed_tools field
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "c1", "finish", {"status": "success", "output": {}},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "root", "finish", {"status": "success", "output": {}},
        )]),
    ]
    proof = run_task(
        TaskContract(
            goal="root", allowed_tools=["read_file", "wiki_search"],
            budget=Budget(max_llm_calls=10, max_children=5, max_depth=3),
            repair_policy=RepairPolicy(enabled=False),
        ),
        NoopPlugin(),
        client=FakeClient(script),
    )
    assert len(proof.children) == 1
    assert proof.children[0].contract.allowed_tools == ["read_file", "wiki_search"]


# -----------------------------------------------------------------------------
# max_children is per-attempt, not cumulative across repair attempts
# -----------------------------------------------------------------------------

def test_max_children_resets_each_repair_attempt() -> None:
    """Attempt 1 spawns max_children children and fails schema validation.
    Attempt 2 must get a FRESH max_children budget — not 0 because of the
    children already attached from attempt 1.
    """
    # Attempt 1: spawn 2 children (the cap), then finish with malformed
    # output that schema validation will reject.
    # Attempt 2: spawn 2 MORE children (must be allowed despite attempt 1's
    # 2 already attached), then finish correctly.
    script = [
        # --- attempt 1 ---
        FakeMessage(tool_calls=[FakeToolCall(
            "s1", "spawn_subtasks",
            {"subtasks": [{"goal": "a1"}, {"goal": "a2"}]},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "c1", "finish", {"status": "success", "output": {}},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "c2", "finish", {"status": "success", "output": {}},
        )]),
        # finish with wrong schema → triggers schema rejection
        FakeMessage(tool_calls=[FakeToolCall(
            "root_bad", "finish", {"status": "success", "output": "wrong type"},
        )]),
        # --- attempt 2 (after repair hint) ---
        FakeMessage(tool_calls=[FakeToolCall(
            "s2", "spawn_subtasks",
            {"subtasks": [{"goal": "b1"}, {"goal": "b2"}]},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "c3", "finish", {"status": "success", "output": {}},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "c4", "finish", {"status": "success", "output": {}},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "root_good", "finish",
            {"status": "success", "output": {"done": True}},
        )]),
    ]
    proof = run_task(
        TaskContract(
            goal="root",
            output_schema={"type": "object"},
            budget=Budget(max_llm_calls=10, max_children=2, max_depth=3),
            repair_policy=RepairPolicy(enabled=True, max_attempts=3),
        ),
        NoopPlugin(),
        client=FakeClient(script),
    )
    # 4 children total across 2 attempts; per-attempt cap was 2.
    assert len(proof.children) == 4, (
        f"expected 4 children across 2 attempts, got {len(proof.children)}"
    )
    assert len(proof.attempts) == 2, f"expected 2 attempts, got {len(proof.attempts)}"
    assert isinstance(proof.final_verdict, Accept), proof.final_verdict


def test_max_children_caps_across_multiple_spawn_calls_in_one_attempt() -> None:
    """Two spawn_subtasks calls in one attempt asking for 3+3 children with
    cap=2 should attach only 2 total (the first call exhausts capacity).
    """
    script = [
        FakeMessage(tool_calls=[FakeToolCall(
            "s1", "spawn_subtasks",
            {"subtasks": [{"goal": "a"}, {"goal": "b"}, {"goal": "c"}]},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "c1", "finish", {"status": "success", "output": {}},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "c2", "finish", {"status": "success", "output": {}},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "s2", "spawn_subtasks",
            {"subtasks": [{"goal": "d"}, {"goal": "e"}, {"goal": "f"}]},
        )]),
        FakeMessage(tool_calls=[FakeToolCall(
            "root", "finish", {"status": "success", "output": {}},
        )]),
    ]
    proof = run_task(
        TaskContract(
            goal="root",
            budget=Budget(max_llm_calls=10, max_children=2, max_depth=3),
            repair_policy=RepairPolicy(enabled=False),
        ),
        NoopPlugin(),
        client=FakeClient(script),
    )
    assert len(proof.children) == 2, (
        f"per-attempt cap should hold across spawn calls; got {len(proof.children)}"
    )


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                print(f"  FAIL {name}: {e}")
                return 1
    print("all runtime invariant tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
