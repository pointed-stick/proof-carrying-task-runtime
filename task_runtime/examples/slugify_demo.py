"""Falsification demo for the universality claim.

If a runtime designed for "recursive QA agents" can run an unrelated code-tests
task end-to-end with no QA-specific code in the runtime, the abstraction is
at least binary-real. If it can't, the framework was QA-shaped all along.

This demo:
  1. Seeds tests/test_slugify.py into a workspace.
  2. Hands a contract (goal + output_schema + verifier='pytest') to the runtime.
  3. The runtime drives the LLM through write_file/run_pytest until finish().
  4. The PytestVerifier re-runs pytest. If it fails, the failure output becomes
     a repair hint for the next attempt.
  5. Final answer + proof tree are written to the workspace.

Run:
  python -m task_runtime.examples.slugify_demo
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# Allow `python path/to/slugify_demo.py` too.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import Budget, RepairPolicy, TaskContract  # noqa: E402
from task_runtime.plugins.code_tests import (  # noqa: E402
    CodeTestsPlugin, set_workspace, write_file,
)
from task_runtime.runtime import run_task  # noqa: E402


SLUGIFY_TESTS = '''\
from slugify import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation():
    assert slugify("Hello, World!") == "hello-world"


def test_collapse_spaces():
    assert slugify("a   b   c") == "a-b-c"


def test_strip_edges():
    assert slugify("  spaced out  ") == "spaced-out"


def test_unicode_letters_kept():
    # Unicode letters are preserved (lowercased); non-letters become hyphens.
    assert slugify("Café Münchner") == "café-münchner"


def test_empty():
    assert slugify("") == ""


def test_only_punctuation():
    assert slugify("!!!---!!!") == ""
'''

CONFTEST = "import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))\n"


def _load_api_key() -> None:
    """Match decompose_graph_agent's convention: read API_KEY from .env.
    The openai SDK reads OPENAI_API_KEY from the environment, so we set that.
    """
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

    workspace = Path(__file__).resolve().parent / "_slugify_workspace"
    # Always start from a clean slate so the demo is a real falsification test.
    # reset_workspace() only nukes the currently-registered workspace; in a
    # fresh process that's None, so we delete the target path explicitly.
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    set_workspace(workspace)

    # Pre-seed the tests + conftest so the model can read them.
    write_file("tests/test_slugify.py", SLUGIFY_TESTS)
    write_file("conftest.py", CONFTEST)

    plugin = CodeTestsPlugin(verifier_target="tests/test_slugify.py")

    contract = TaskContract(
        goal=(
            "Implement slugify(text: str) -> str in slugify.py at the workspace root, "
            "so that every test in tests/test_slugify.py passes. Lowercase the input, "
            "replace runs of non-letter characters with a single hyphen, strip "
            "leading/trailing hyphens, and preserve unicode letters."
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
        inputs={"tests_path": "tests/test_slugify.py"},
        budget=Budget(max_llm_calls=15, max_children=0, max_depth=1),
        repair_policy=RepairPolicy(enabled=True, max_attempts=4),
    )

    print(f"Workspace: {workspace}")
    print(f"Contract level: {contract.level}  (verifier={contract.verifier})")
    proof = run_task(contract, plugin)

    proof_path = workspace / "proof_tree.json"
    proof_path.write_text(
        json.dumps(proof.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    final = proof.final_verdict
    kind = type(final).__name__ if final else "None"
    print()
    print("=" * 60)
    print(f"final verdict: {kind}")
    print(f"attempts:      {len(proof.attempts)}")
    print(f"proof tree:    {proof_path}")
    if proof.final_result and proof.final_result.output:
        print("output:")
        print(json.dumps(proof.final_result.output, indent=2)[:800])
    if final is not None and not proof.accepted:
        print(f"reason: {getattr(final, 'reason', '')}")

    return 0 if proof.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
