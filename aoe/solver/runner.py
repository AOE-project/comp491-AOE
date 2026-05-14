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

from core.config import load_settings
from core.state import GraphState

_SESSIONS_ROOT = Path(__file__).resolve().parent.parent / "sessions"
_TIMEOUT_SECONDS = 60
_settings = load_settings()

_INJECTED_ERROR_MESSAGES = {
    "syntax_error": "SyntaxError: invalid syntax at line 5",
    "runtime_error": "NameError: name 'x' is not defined",
    "modeling_error": "GurobiPy error: model.addvars() received invalid argument",
    "unknown_error": "UnexpectedError: Something went wrong",
}


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


# ---------------------------------------------------------------------------
# Node helpers (from ko_uni_map)
# ---------------------------------------------------------------------------

def _get_injected_test_error_result(state: GraphState) -> dict | None:
    """Return injected solver output in test mode; otherwise return None."""
    regeneration_attempts = state.get("regeneration_attempts", 0)
    should_inject_error = (
        _settings.test_error_injection
        and regeneration_attempts == 0  # Only inject on 1st attempt (no regenerations yet)
    )
    print(
        f"[DEBUG] solver_node: regeneration_attempts={regeneration_attempts}, "
        f"should_inject_error={should_inject_error}, "
        f"test_error_injection={_settings.test_error_injection}"
    )

    if not should_inject_error:
        return None

    error_type = getattr(_settings, "test_error_type", "syntax_error")
    error_msg = _INJECTED_ERROR_MESSAGES.get(error_type, _INJECTED_ERROR_MESSAGES["syntax_error"])
    print(f"[INJECTING ERROR] Type: {error_type.upper()}")
    print(f"[ERROR MESSAGE] {error_msg}")

    # Return an injected failure using the rich ardil payload shape so that
    # downstream consumers (UI, debug nodes) see a consistent structure.
    return {
        "solver_result": _empty_payload({
            "status": "runtime_error",
            "stderr": error_msg,
            "returncode": 1,
        }),
        "last_execution_error": error_msg,
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def solver_node(state: GraphState) -> dict:
    """LangGraph node — runs the generated Gurobi script and stores the result."""

    # Test-mode error injection (ko_uni_map feature, kept as requested)
    injected_result = _get_injected_test_error_result(state)
    if injected_result is not None:
        return injected_result

    code = (state.get("generated_code") or "").strip()
    if not code:
        return {
            "solver_result": _empty_payload({
                "stderr": "No generated code found in state.",
            }),
            "last_execution_error": "No generated code found in state.",
        }

    session_id = state.get("session_id")
    session_dir = _SESSIONS_ROOT / session_id if session_id else None

    result = run_generated_code(code, cwd=session_dir)

    # Capture execution error for debug routing (ko_uni_map feature)
    last_execution_error = None
    if result["status"] not in ("optimal", "infeasible", "unbounded"):
        last_execution_error = result.get("stderr") or f"Solver status: {result['status']}"

    return {
        "solver_result":        result,
        "last_execution_error": last_execution_error,
    }