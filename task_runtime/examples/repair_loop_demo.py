"""Forced-repair demo — exercises the verifier → repair → retry loop live.

The slugify tests in `slugify_demo.py` happened to one-shot in our trial; this
demo uses tests specifically chosen to be hostile to the most common first-pass
implementations:

  - `[^\\w]+` (Python `re` default): treats digits and underscore as letters,
    so test_digits_are_separators and test_underscore_is_separator FAIL.
  - `[^a-z]+`: treats unicode letters as non-letters, so
    test_unicode_letters_kept FAILS.

Only an implementation that walks character-by-character with `.isalpha()`
(or an equivalent unicode-letter-aware predicate) passes everything.

The point isn't slugify; the point is to get a proof tree of shape:

  attempt 1 → RejectWithRepairHint(pytest failure)
  attempt 2 → Accept(pytest passed)

If both attempts get Accept, the model one-shotted it again (also fine).
If the loop never converges, the budget caps it and the proof tree records
the dead end honestly.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import Budget, RepairPolicy, TaskContract  # noqa: E402
from task_runtime.plugins.code_tests import (  # noqa: E402
    CodeTestsPlugin, set_workspace, write_file,
)
from task_runtime.runtime import run_task  # noqa: E402


# Tests deliberately chosen to fail naive regex-based implementations.
HOSTILE_TESTS = '''\
from slugify import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_digits_are_separators():
    # Hostile to `re.sub(r"[^\\w]+", "-", s)` — \\w includes digits.
    assert slugify("hello123world") == "hello-world"


def test_underscore_is_separator():
    # Hostile to `[^\\w]+` — \\w includes underscore.
    assert slugify("hello_world") == "hello-world"


def test_unicode_letters_kept():
    # Hostile to `[^a-z]+` — strips accented letters.
    assert slugify("Café Münchner") == "café-münchner"


def test_mixed_collapse():
    # Hostile to anything that doesn't collapse adjacent separators.
    assert slugify("a___---   b") == "a-b"


def test_digits_in_middle_collapse():
    assert slugify("abc123def456ghi") == "abc-def-ghi"


def test_strip_edges():
    assert slugify("  spaced out  ") == "spaced-out"


def test_empty():
    assert slugify("") == ""
'''

CONFTEST = "import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))\n"


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

    workspace = Path(__file__).resolve().parent / "_repair_loop_workspace"
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    set_workspace(workspace)

    write_file("tests/test_slugify.py", HOSTILE_TESTS)
    write_file("conftest.py", CONFTEST)

    plugin = CodeTestsPlugin(verifier_target="tests/test_slugify.py")

    contract = TaskContract(
        goal=(
            "Implement slugify(text: str) -> str in slugify.py at the workspace root "
            "so EVERY test in tests/test_slugify.py passes. Lowercase the input; replace "
            "any run of characters that are NOT unicode letters (a–z plus accented "
            "letters; digits and underscore are NOT letters) with a single hyphen; "
            "strip leading/trailing hyphens. Read the tests first to be sure."
        ),
        output_schema={
            "type": "object",
            "required": ["implementation_path", "summary"],
            "properties": {
                "implementation_path": {"type": "string"},
                "summary": {"type": "string"},
            },
        },
        verifier="pytest",
        success_criteria=[
            "slugify.py exists at the workspace root",
            "all tests in tests/test_slugify.py pass under `pytest -x`",
        ],
        budget=Budget(max_llm_calls=15, max_children=0, max_depth=1),
        repair_policy=RepairPolicy(enabled=True, max_attempts=4),
    )

    print(f"Workspace: {workspace}")
    print(f"Contract level: {contract.level}")
    print(f"Repair budget: {contract.repair_policy.max_attempts} attempts")
    proof = run_task(contract, plugin)

    proof_path = workspace / "proof_tree.json"
    proof_path.write_text(
        json.dumps(proof.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print(f"final verdict: {type(proof.final_verdict).__name__ if proof.final_verdict else 'None'}")
    print(f"attempts:      {len(proof.attempts)}")
    print()
    print("Per-attempt verdicts:")
    for a in proof.attempts:
        vk = type(a.verdict).__name__ if a.verdict else "None"
        had_hint = a.repair_hint is not None
        print(f"  #{a.attempt_no}: {vk}  (repair_hint_in: {had_hint}, cost: {a.cost})")

    print()
    print(f"proof tree: {proof_path}")

    return 0 if proof.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
