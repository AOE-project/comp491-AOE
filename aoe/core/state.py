"""
core/state.py — GraphState, MILPModel, SolverResult, exceptions etc.
Shared across the whole system to pass data and results between agents, the graph, and the UI.
"""

from typing import TypedDict


class GraphState(TypedDict):
    session_id: str
    problem_description: str
    history: list
    milp_model: dict        # output of Analyser
    generated_code: str     # output of CodeGenerator
    debug_history: list     # output of Debug (future)
    solver_result: dict     # output of SolverRunner (future)
    explanation: str        # output of Explainer
    token_usage: dict