"""
core/state.py — GraphState, shared across agents, graph, and UI.
All fields are optional at initialisation; agents populate them progressively.
"""

from typing import TypedDict


class GraphState(TypedDict):
    # Session
    session_id: str
    problem_description: str        # initial natural language user message
    history: list                   # full conversation history (role, content pairs)

    # Analyser
    iteration_count: int            # number of analyser turns so far (max 10)
    analyser_output: dict           # full Option-B JSON from the last analyser call
    milp_model: dict                # milp_model portion extracted from analyser_output
    confirmed_assumptions: list
    unconfirmed_assumptions: list
    open_questions: list            # surfaced to the user
    analysis_summary: str           # plain-English summary shown to the user
    technical_summary: str          # symbolic summary for internal use / CodeGenerator
    analyser_approved: bool         # True once the user explicitly approves the model

    # Data collection (after analyser approval)
    raw_data: dict                  # keyed by parameter data_key; filled by user after approval
    input_retrieval_queue: list     # ordered collection tasks: sets first, then parameters
    input_retrieval_cursor: int     # index of the current task in the queue
    current_input_spec: dict        # describes what input is needed now; None when collection is done

    # Code generation
    generated_code: str             # output of CodeGenerator

    # Solver
    solver_result: dict             # output of SolverRunner

    # Explanation
    explanation: str                # output of Explainer

    # Token tracking
    token_usage: dict
