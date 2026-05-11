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
        is_new = state is None
        if is_new:
            session_id = str(uuid.uuid4())
            state: GraphState = {
                "session_id": session_id,
                "problem_description": user_message,
                "history": [{"role": "user", "content": user_message}],
                "chat_history": [],
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

    def save(self, state: GraphState) -> None:
        """Persist the current state to disk without running the graph.

        Used by the UI after it finalises chat_history (which can only be
        completed once the assistant turn has been rendered).
        """
        if self._logger is not None:
            self._logger.checkpoint(state)

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
