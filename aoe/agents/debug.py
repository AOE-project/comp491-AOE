"""
agents/debug.py — DebugAgent: classifies errors and enriches context for recovery.

Uses ErrorContext to deeply analyze failures, classify into categories,
and provide strategic prompting for LLM recovery agents.

Architecture:
1. Catch & Inspect: Extract rich error context from exception
2. Classification: Identify error category with analysis
3. Prompting Strategy: Structure LLM prompt with hints
4. Retry Loop: Track attempts with accumulating context
"""

from datetime import datetime
from core.state import GraphState
from core.error_analysis import ErrorContext


class DebugLimitExceeded(Exception):
    """Raised when max_debug_attempts is exceeded."""
    pass


# debug agent
def debug_node(state: GraphState) -> dict:
    """
    Classify execution error with rich context analysis.
    Does not perform regeneration; routing logic decides the action.
    
    Creates ErrorContext for:
    - Deep error classification
    - Cause analysis with hints
    - Strategic LLM prompting
    
    If max_debug_attempts exceeded, sets error_type to "max_retries_exceeded"
    so router can handle gracefully (e.g., fail or ask user).
    """
    error = state.get("last_execution_error")
    if not error:
        return {}  # no error, skip
    
    # Extract context for rich analysis
    generated_code = state.get("generated_code", "")
    milp_model = state.get("milp_model", {})
    
    # Create rich error context
    error_context = ErrorContext(
        error_message=error,
        generated_code=generated_code,
        milp_model=milp_model
    )
    
    # Track attempt
    attempts = state.get("debug_attempts", [])
    attempt_count = len(attempts) + 1
    print(f"[DEBUG] debug_node: attempt_count={attempt_count}, regeneration_attempts={state.get('regeneration_attempts', 0)}, max_debug={state.get('max_debug_attempts', 5)}")
    previous_failed_attempts = [
        {
            "error_message": a.get("error", ""),
            "identified_cause": a.get("analysis", {}).get("primary_cause", ""),
            "code": a.get("code_attempted", "")
        }
        for a in attempts
    ]
    
    if attempt_count > state.get("max_debug_attempts", 5):
        # Max retries exceeded — don't crash, just mark it so router can handle
        return {
            "last_error_type": "max_retries_exceeded",
            "error_context": error_context.to_dict(),
            "debug_attempts": attempts + [{
                "attempt": attempt_count,
                "error": error,
                "error_type": error_context.error_type,
                "analysis": error_context.analysis,
                "timestamp": datetime.now().isoformat()
            }]
        }
    
    return {
        "last_error_type": error_context.error_type,
        "error_context": error_context.to_dict(),
        "error_analysis": error_context.analysis,
        "debug_attempts": attempts + [{
            "attempt": attempt_count,
            "error": error,
            "error_type": error_context.error_type,
            "analysis": error_context.analysis,
            "timestamp": datetime.now().isoformat()
        }]
    }

