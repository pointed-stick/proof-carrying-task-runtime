"""Code-tests plugin: write source files in a workspace and verify with pytest.

Tools:
  write_file(path, contents)
  read_file(path)
  list_files()
  run_pytest(target?)

Verifier:
  pytest  —  re-runs pytest at verification time (does not trust the model's
             claim of having run tests successfully)
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from ..core import (
    Accept, IncoherentContract, RejectWithRepairHint,
    TaskContract, TaskResult, Verdict,
)


# Module-level workspace state. A single run uses one workspace; if you want
# isolated per-task workspaces, refactor this to a class. Kept simple for v1.
_WORKSPACE: dict = {"dir": None}


def _ws() -> Path:
    if _WORKSPACE["dir"] is None:
        _WORKSPACE["dir"] = Path(tempfile.mkdtemp(prefix="task_runtime_"))
    return Path(_WORKSPACE["dir"])


def set_workspace(path: str | Path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    _WORKSPACE["dir"] = p


def reset_workspace() -> None:
    d = _WORKSPACE["dir"]
    if d and Path(d).exists():
        shutil.rmtree(d, ignore_errors=True)
    _WORKSPACE["dir"] = None


def _safe_path(path: str) -> Path:
    """Resolve `path` relative to the workspace, refusing escapes.

    Cross-platform: even on POSIX we reject Windows-style absolute paths
    (`C:\\...`, `\\foo`) and backslash traversal — otherwise those would be
    treated as ordinary filenames and silently allowed. Then we resolve
    relative to the workspace root and require containment.
    """
    if path is None or path == "":
        raise ValueError("path required")
    # Treat the input through Windows-path semantics regardless of host OS so
    # `..` traversal, drive letters, and root-relative `\foo` all get rejected
    # consistently. PureWindowsPath understands both `/` and `\` as separators.
    #
    # NOTE: `win.root` is checked separately from `win.is_absolute()` because
    # PureWindowsPath considers a path absolute only when it has BOTH a drive
    # AND a root. So `\foo\bar.txt` has root=='\\' but is_absolute()==False.
    # Without the explicit `win.root` check, root-relative Windows paths leak.
    win = PureWindowsPath(path)
    if win.is_absolute() or win.drive or win.root or ".." in win.parts:
        raise ValueError(f"path escapes workspace: {path!r}")
    root = _ws().resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"path escapes workspace: {path!r}") from None
    return candidate


def _workspace_hashes() -> dict[str, str]:
    """SHA-256 hex of every .py file in the workspace (excluding __pycache__).

    Becomes part of the verifier `record` so a verdict is independently
    re-checkable: anyone can recompute these hashes against the same files
    and re-run pytest to confirm the verdict.
    """
    root = _ws().resolve()
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts or ".pytest_cache" in p.parts:
            continue
        try:
            out[str(p.relative_to(root)).replace("\\", "/")] = (
                "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
            )
        except OSError:
            pass
    return out


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

def write_file(path: str, contents: str) -> dict:
    """Write a UTF-8 text file in the workspace, creating parent dirs."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(contents, encoding="utf-8")
    return {"ok": True, "path": str(p.relative_to(_ws().resolve())), "bytes": len(contents)}

write_file._tool_spec = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a UTF-8 text file in the workspace, creating parent dirs.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "contents": {"type": "string"},
            },
            "required": ["path", "contents"],
        },
    },
}


def read_file(path: str) -> dict:
    """Read a workspace file."""
    try:
        p = _safe_path(path)
    except ValueError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"no such file: {path}"}
    return {"path": path, "contents": p.read_text(encoding="utf-8")}

read_file._tool_spec = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


def list_files() -> dict:
    """List all files in the workspace."""
    return {
        "files": sorted(
            str(p.relative_to(_ws()))
            for p in _ws().rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    }

list_files._tool_spec = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List all files in the workspace (excluding __pycache__).",
        "parameters": {"type": "object", "properties": {}},
    },
}


def run_pytest(target: str = "") -> dict:
    """Run pytest in the workspace (-x -q --tb=short). Returns exit code + output."""
    args = ["python", "-m", "pytest", "-x", "-q", "--tb=short"]
    if target:
        args.append(target)
    try:
        proc = subprocess.run(
            args, cwd=str(_ws()), capture_output=True, text=True, timeout=60,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "pytest timeout (60s)"}
    except FileNotFoundError as e:
        return {"exit_code": -2, "stdout": "", "stderr": f"pytest not runnable: {e}"}

run_pytest._tool_spec = {
    "type": "function",
    "function": {
        "name": "run_pytest",
        "description": "Run pytest in the workspace. Optional target path.",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string"}},
        },
    },
}


# -----------------------------------------------------------------------------
# Verifier
# -----------------------------------------------------------------------------

@dataclass
class PytestVerifier:
    target: str = ""

    def _command(self) -> list[str]:
        args = ["python", "-m", "pytest", "-x", "-q", "--tb=short"]
        if self.target:
            args.append(self.target)
        return args

    def check(self, contract: TaskContract, result: TaskResult, workspace: dict) -> Verdict:
        # Re-run pytest; do not trust the model's prior run.
        out = run_pytest(target=self.target)
        record = {
            "verifier": "pytest",
            "command": " ".join(self._command()),
            "cwd": str(_ws().resolve()),
            "exit_code": out["exit_code"],
            "stdout_tail": out["stdout"],
            "stderr_tail": out["stderr"],
            "workspace_hashes": _workspace_hashes(),
        }
        if out["exit_code"] == 0:
            return Accept(
                reason=f"pytest passed: {out['stdout'][:200].strip()}",
                record=record,
            )
        return RejectWithRepairHint(
            reason=f"pytest exit {out['exit_code']}",
            hint=(
                "pytest is failing. Read the failure output below and fix your implementation. "
                "Then call run_pytest yourself to confirm, then call finish().\n"
                f"--- STDOUT ---\n{out['stdout']}\n--- STDERR ---\n{out['stderr']}"
            ),
            missing_requirements=["all pytest tests must pass"],
            record=record,
        )


# -----------------------------------------------------------------------------
# Plugin
# -----------------------------------------------------------------------------

class CodeTestsPlugin:
    name = "code_tests"

    def __init__(self, verifier_target: str = ""):
        self._verifiers = {"pytest": PytestVerifier(target=verifier_target)}

    def tools(self):
        return {
            "write_file": write_file,
            "read_file": read_file,
            "list_files": list_files,
            "run_pytest": run_pytest,
        }

    def verifiers(self):
        return self._verifiers

    def render_contract_context(self, contract: TaskContract) -> str:
        return (
            f"PLUGIN: code_tests\n"
            f"Workspace: {_ws()}\n"
            f"Tools: write_file, read_file, list_files, run_pytest.\n"
            f"Workflow: read existing files to understand the contract; write your "
            f"implementation; run_pytest to check; iterate; call finish().\n"
            f"The configured verifier ('{contract.verifier}') will re-run pytest at "
            f"verification time, so you must actually make the tests pass."
        )

    def coherence_check(self, contract: TaskContract):
        if contract.verifier and contract.verifier not in self._verifiers:
            return IncoherentContract(
                reason=(
                    f"verifier '{contract.verifier}' not provided by code_tests "
                    f"(available: {sorted(self._verifiers)})"
                )
            )
        return None
