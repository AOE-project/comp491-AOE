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
                "last_execution_error": None,
                "debug_attempts": [],
                "last_error_type": None,
                "max_debug_attempts": 5,
                "regeneration_attempts": 0,
                "solver_result": {},
                "explanation": "",
                "chat_mode": False,
                "chat_response": "",
                "chat_pending_modification": None,
                "chat_modification_approved": False,
                "is_regeneration": False,
                "regen_set_changes": False,
                "regen_force_ask": [],
                "token_usage": {},
                "costs": {"total": 0.0, "by_agent": {}},
            }
            self._logger = SessionLogger(session_id, self._sessions_root)
            self._logger.log_event("session_started", {"problem": user_message})
        else:
            state["history"].append({"role": "user", "content": user_message})

        if state.get("chat_mode"):
            # Save pending modification BEFORE chat_agent clears it on confirmation.
            saved_pending = state.get("chat_pending_modification") or {}

            # Solver has already run — send the message to the chat agent.
            updates = chat_agent_node(state)
            state = {**state, **updates}

            if state.get("chat_modification_approved"):
                # Forward the chat agent's description to the analyser so it can
                # update the MILP model, then collect any new data and re-solve.
                original_request = (
                    saved_pending.get("user_request")
                    or saved_pending.get("description")
                    or ""
                )
                if original_request:
                    new_message = (
                        f"Apply the following structural change to the existing model "
                        f"and return the updated model with open_questions set to []: "
                        f"{original_request}"
                    )
                    print(f"  New message added to history: {new_message}")
                    state["history"].append({
                        "role": "user",
                        "content": new_message,
                    })

                state["analyser_approved"] = False
                state["is_regeneration"] = True
                state["regen_set_changes"] = saved_pending.get("set_changes", False)
                state["regen_force_ask"] = saved_pending.get("regen_force_ask") or []
                # Temporarily leave chat mode so data-collection messages are
                # routed through compiled_graph (not the chat agent).
                state["chat_mode"] = False
                # Clear stale artefacts so the pipeline regenerates them.
                state["input_retrieval_queue"] = []
                state["input_retrieval_cursor"] = 0
                state["current_input_spec"] = {}
                state["generated_code"] = ""
                state["solver_result"] = {}
                state["explanation"] = ""
                state["latex_model"] = ""
                state["latex_png_path"] = ""

                state = compiled_graph.invoke(state)

                state["chat_modification_approved"] = False
                state["regen_set_changes"] = False
                state["regen_force_ask"] = []
                # After analyser has updated the model, pass-through on future calls.
                state["analyser_approved"] = True
                # Move any pending assumptions to confirmed — the user already
                # approved the modification through the chat agent flow.
                state["confirmed_assumptions"] = (
                    state.get("confirmed_assumptions") or []
                ) + (state.get("unconfirmed_assumptions") or [])
                state["unconfirmed_assumptions"] = []
                state["open_questions"] = []

                if state.get("solver_result"):
                    # No new data was needed — re-solve completed in one shot.
                    state["chat_mode"] = True
                    state["is_regeneration"] = False
                    state["chat_response"] = ""  # show new solver result
        else:
            state = compiled_graph.invoke(state)
            # Switch to chat mode once the solver result is available.
            if state.get("solver_result"):
                state["chat_mode"] = True
                state["is_regeneration"] = False
                state["chat_response"] = ""  # clear stale chat response so UI shows solver result card

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
