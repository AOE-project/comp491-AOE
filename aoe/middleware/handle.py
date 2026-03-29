"""
middleware/handle.py — AOEHandle.

Entry point for all UIs. Runs the LangGraph workflow, returns Events,
and delegates session persistence to SessionManager.
Returns (updated_state, list[Event]) to the caller.
"""

import uuid
from core.state import GraphState
from graph.workflow import compiled_graph


class AOEHandle:
    def run(self, problem_description: str) -> GraphState:
        initial_state: GraphState = {
            "session_id": str(uuid.uuid4()),
            "problem_description": problem_description,
            "history": [],
            "milp_model": {},
            "generated_code": "",
            "debug_history": [],
            "solver_result": {},
            "explanation": "",
            "token_usage": {},
        }
        return compiled_graph.invoke(initial_state)
