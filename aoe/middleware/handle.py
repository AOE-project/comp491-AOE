"""
middleware/handle.py — AOEHandle.

Sole entry point for all UIs. Manages the multi-turn analyser loop:
each call to run() represents one user message. The graph runs once per call
and returns; the caller is responsible for collecting the next user message
and calling run() again with the updated state.
"""

import uuid
from core.state import GraphState
from graph.workflow import compiled_graph


class AOEHandle:
    def run(self, user_message: str, state: GraphState | None = None) -> GraphState:
        """
        Run one turn of the graph.

        On the first call pass state=None; the initial state is built from user_message.
        On next calls pass the state returned by the previous call and set
        user_message to the user's latest reply.
        """
        if state is None:
            state: GraphState = {
                "session_id": str(uuid.uuid4()),
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
                "generated_code": "",
                "solver_result": {},
                "explanation": "",
                "token_usage": {},
            }
        else:
            state["history"].append({"role": "user", "content": user_message})

        return compiled_graph.invoke(state)
