"""
solver/runner.py — SolverRunner.

Writes the generated Gurobi script to a temporary file, executes it in a
subprocess, and returns a structured result dict.

The result dict is stored in GraphState["solver_result"] and has the form:
    {
        "status":     "success" | "runtime_error" | "timeout",
        "stdout":     str,   # everything the script printed
        "stderr":     str,   # error output — used by the retry mechanism
        "returncode": int,   # process exit code (0 = clean exit)
    }
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from core.state import GraphState

_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_generated_code(code: str) -> dict:
    """
    Write code to a temp .py file, run it, and return a result dict.
    The temp file is deleted automatically after the run.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        tmp_path = Path(f.name)

    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        status = "success" if proc.returncode == 0 else "runtime_error"
        return {
            "status":     status,
            "stdout":     proc.stdout,
            "stderr":     proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "status":     "timeout",
            "stdout":     "",
            "stderr":     f"Script exceeded {_TIMEOUT_SECONDS}s timeout.",
            "returncode": -1,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def solver_node(state: GraphState) -> dict:
    """
    LangGraph node — runs the generated Gurobi script and stores the result.
    """
    code = (state.get("generated_code") or "").strip()
    if not code:
        return {
            "solver_result": {
                "status":     "runtime_error",
                "stdout":     "",
                "stderr":     "No generated code found in state.",
                "returncode": -1,
            }
        }

    result = run_generated_code(code)
    return {"solver_result": result}