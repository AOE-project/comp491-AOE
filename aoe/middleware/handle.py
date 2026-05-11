"""
middleware/handle.py — AOEHandle.

Sole entry point for all UIs. Each call to run() represents one user message:
the graph runs once and returns, the caller collects the next message and calls
run() again with the updated state.
"""

import uuid
from pathlib import Path

from agents.chat_agent import chat_agent_node
from core.state import GraphState
from core.logger import SessionLogger
from graph.workflow import compiled_graph, re_optimization_graph


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
                "solver_result": {},
                "explanation": "",
                "chat_mode": False,
                "chat_history": [],
                "chat_response": "",
                "chat_pending_modification": None,
                "chat_modification_approved": False,
                "token_usage": {},
            }
            self._logger = SessionLogger(session_id, self._sessions_root)
            self._logger.log_event("session_started", {"problem": user_message})
        else:
            state["history"].append({"role": "user", "content": user_message})

        if state.get("chat_mode"):
            # Solver has already run — send the message directly to the chat agent.
            updates = chat_agent_node(state)
            state = {**state, **updates}
            # The chat agent updated milp_model — regenerate code and re-solve.
            if state.get("chat_modification_approved"):
                state = re_optimization_graph.invoke(state)
                state["chat_modification_approved"] = False
                state["chat_response"] = ""  # clear so UI shows the new solver result
        else:
            state = compiled_graph.invoke(state)
            # Switch to chat mode once the solver result is available.
            if state.get("solver_result"):
                state["chat_mode"] = True

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
