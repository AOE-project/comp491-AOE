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
    chat_history: list              # Gradio chatbot transcript (rendered messages, written by UI)

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
    code_syntax_error: str | None   # syntax error message from CodeGenerator, None if valid

    # Solver
    solver_result: dict             # output of SolverRunner

    # LaTeX formulation
    latex_model: str                # LaTeX body source; populated by LaTeXGeneratorAgent
    latex_png_path: str             # absolute path to the compiled PNG; populated by LaTeXGeneratorAgent

    # Explanation
    explanation: str                # output of Explainer

    # Chat Agent (post-solver interactive Q&A)
    chat_mode: bool                 # True once the solver has run and chat is active
    chat_history: list              # chat-specific turns: [{"role": ..., "content": ...}, ...]
    chat_response: str              # last response text from the chat agent
    chat_pending_modification: dict # proposed model change awaiting user approval
    chat_modification_approved: bool  # True when user approves a pending modification

    # Token tracking
    token_usage: dict
