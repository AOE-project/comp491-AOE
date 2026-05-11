"""
solver/runner.py — SolverRunner.

Writes the generated Gurobi script to a temporary file, runs it inside the
session folder, and returns a structured result dict.

The result dict is stored in GraphState["solver_result"] and has the shape:
    {
        "status":          "optimal" | "infeasible" | "unbounded" |
                           "other:<code>" | "runtime_error" | "timeout",
        "objective_value": float | None,
        "variables":       {"<name>[<idx>]": <value>, ...},
        "iis_path":        str | None,        # path to iis.ilp on infeasibility
        "stdout":          str,
        "stderr":          str,
        "returncode":      int,
    }
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from core.state import GraphState

_SESSIONS_ROOT = Path(__file__).resolve().parent.parent / "sessions"
_TIMEOUT_SECONDS = 60


def _empty_payload(extra: dict) -> dict:
    base = {
        "status":          "runtime_error",
        "stdout":          "",
        "stderr":          "",
        "returncode":      -1,
        "objective_value": None,
        "variables":       {},
        "iis_path":        None,
    }
    base.update(extra)
    return base


def run_generated_code(code: str, cwd: Path | None = None) -> dict:
    """
    Execute the generated Gurobi script in a subprocess. The script is
    executed with cwd set to the session folder (when provided), so artefacts
    it writes — result.json, iis.ilp — land alongside the session's other
    files.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        tmp_path = Path(f.name)

    if cwd is not None:
        work_dir = Path(cwd)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = tmp_path.parent

    # Clear any stale result.json from a previous run so we never report
    # last-run output if the new script fails before writing.
    stale = work_dir / "result.json"
    if stale.exists():
        stale.unlink()

    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            cwd=str(work_dir),
        )
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        return _empty_payload({
            "status": "timeout",
            "stderr": f"Script exceeded {_TIMEOUT_SECONDS}s timeout.",
        })
    finally:
        tmp_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        return _empty_payload({
            "status":     "runtime_error",
            "stdout":     proc.stdout,
            "stderr":     proc.stderr,
            "returncode": proc.returncode,
        })

    parsed = {}
    result_json = work_dir / "result.json"
    if result_json.exists():
        try:
            parsed = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parsed = {}

    iis_path = work_dir / "iis.ilp"
    iis_str = str(iis_path) if iis_path.exists() else None

    return {
        "status":          parsed.get("status", "other:unknown"),
        "objective_value": parsed.get("objective_value"),
        "variables":       parsed.get("variables") or {},
        "iis_path":        iis_str,
        "stdout":          proc.stdout,
        "stderr":          proc.stderr,
        "returncode":      proc.returncode,
    }


def solver_node(state: GraphState) -> dict:
    """LangGraph node — runs the generated Gurobi script and stores the result."""
    code = (state.get("generated_code") or "").strip()
    if not code:
        return {
            "solver_result": _empty_payload({
                "stderr": "No generated code found in state.",
            })
        }

    session_id = state.get("session_id")
    session_dir = _SESSIONS_ROOT / session_id if session_id else None

    return {"solver_result": run_generated_code(code, cwd=session_dir)}
