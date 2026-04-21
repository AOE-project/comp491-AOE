"""
solver/runner.py — SolverRunner.

Writes the generated Gurobi script to a temporary file, executes it in a
subprocess, and returns a structured result dict.

The result dict is stored in GraphState["solver_result"] and has the form:
    {
        "status":     "success" | "failed" | "timeout",
        "stdout":     str,   # everything the script printed
        "stderr":     str,   # error output — used by the retry mechanism
        "returncode": int,   # process exit code (0 = clean exit)
    }
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from core.config import load_settings
from core.state import GraphState

_TIMEOUT_SECONDS = 60
_settings = load_settings()

_INJECTED_ERROR_MESSAGES = {
    "syntax_error": "SyntaxError: invalid syntax at line 5",
    "runtime_error": "NameError: name 'x' is not defined",
    "modeling_error": "GurobiPy error: model.addvars() received invalid argument",
    "unknown_error": "UnexpectedError: Something went wrong",
}


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
        status = "success" if proc.returncode == 0 else "failed"
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

    return {
        "solver_result": {
            "status": "failed",
            "stdout": "",
            "stderr": error_msg,
            "returncode": 1,
        },
        "last_execution_error": error_msg,
    }

def solver_node(state: GraphState) -> dict:
    """
    LangGraph node — runs the generated Gurobi script and stores the result.
    """
    
    injected_result = _get_injected_test_error_result(state)
    if injected_result is not None:
        return injected_result
    
    code = (state.get("generated_code") or "").strip()
    if not code:
        error_msg = "No generated code found in state."
        return {
            "solver_result": {
                "status":     "failed",
                "stdout":     "",
                "stderr":     error_msg,
                "returncode": -1,
            },
            "last_execution_error": error_msg,
        }

    result = run_generated_code(code)
    
    # Capture execution error for debug routing
    last_execution_error = None
    if result["status"] != "success":
        # Combine stderr and return code into a descriptive error message
        last_execution_error = f"{result['status']}: {result['stderr']}"
    
    return {
        "solver_result": result,
        "last_execution_error": last_execution_error,
    }