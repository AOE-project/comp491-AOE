"""
middleware/handle.py — AOEHandle.

Sole entry point for all UIs. Each call to run() represents one user message:
the graph runs once and returns, the caller collects the next message and calls
run() again with the updated state.
"""

import uuid
from pathlib import Path

from core.state import GraphState
from core.logger import SessionLogger
from graph.workflow import compiled_graph


class AOEHandle:

    def __init__(self, sessions_root: Path | None = None):
        self._sessions_root = sessions_root
        self._logger: SessionLogger | None = None

    def run(self, user_message: str, state: GraphState | None = None) -> GraphState:
        if state is None:
            session_id = str(uuid.uuid4())
            state: GraphState = {
                "session_id": session_id,
                "problem_description": user_message,
                "history": [{"role": "user", "content": user_message}],
                "iteration_count": 0,
                "analyser_output": None,
                "milp_model": {},
                "confirmed_assumptions": [],
                "unconfirmed_assumptions": [],
                "open_questions": [],
                "analysis_summary": "",
                "technical_summary": "",
                "analyser_approved": False,
                "raw_data": {},
                "input_retrieval_queue": [],
                "input_retrieval_cursor": 0,
                "current_input_spec": {},
                "generated_code": "",
                "code_syntax_error": None,
                "last_execution_error": None,
                "debug_attempts": [],
                "last_error_type": None,
                "max_debug_attempts": 5,
                "regeneration_attempts": 0,
                "solver_result": {},
                "explanation": "",
                "token_usage": {},
            }
            self._logger = SessionLogger(session_id, self._sessions_root)
            self._logger.log_event("session_started", {"problem": user_message})
        else:
            state["history"].append({"role": "user", "content": user_message})

        state = compiled_graph.invoke(state)
        self._logger.checkpoint(state)
        return state

    @classmethod
    def resume(
        cls,
        session_id: str,
        sessions_root: Path | None = None,
    ) -> tuple["AOEHandle", GraphState]:
        """Restore a previous session. Returns (handle, state)."""
        handle = cls(sessions_root=sessions_root)
        handle._logger, state = SessionLogger.load(session_id, sessions_root)
        return handle, state
